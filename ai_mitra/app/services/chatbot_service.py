from typing import Dict, Any, List, Optional
import json
import re
from datetime import datetime

from app.services.claude_service import ClaudeService
from app.services.language_service import LanguageService
from app.services.navigation_service import NavigationService
from app.models.response_models import ChatResponse, TagsModel
from app.config import load_app_context, DEFAULT_LANGUAGE

# In-memory storage for chat history
chat_memory = {}

class ChatbotService:
    """Main service for processing chat messages and generating responses"""
    
    def __init__(self):
        self.claude_service = ClaudeService()
        self.language_service = LanguageService()
        self.navigation_service = NavigationService()
        self.app_context = load_app_context()
    
    async def process_message(self, message: str, language: str = None, user_id: str = None) -> ChatResponse:
        """Process a chat message and return a response with navigation suggestions"""
        # Detect language if not provided
        detected_language = self.language_service.detect_language(message)
        request_language = language or detected_language or DEFAULT_LANGUAGE
        
        # Clean up the message by removing language prefix if present
        clean_message = self.language_service.clean_language_prefix(message)
        
        # Get chat history for this user if user_id is provided
        user_history = None
        if user_id and user_id in chat_memory:
            user_history = chat_memory.get(user_id, [])
        
        # Add current message to chat history if user_id is provided
        if user_id:
            self._add_to_chat_history(user_id, "user", clean_message)
        
        # Generate farming advice and extract tags using Claude, including chat history
        advice_message, tags = await self.claude_service.generate_farming_response(
            clean_message, request_language, self.app_context, chat_history=user_history
        )
        
        # Add response to chat history if user_id is provided
        if user_id:
            self._add_to_chat_history(user_id, "assistant", advice_message)
        
        # Suggest navigation routes based on the message and extracted tags
        # Pass the actual tags extracted from Claude's response
        suggested_routes = self.navigation_service.suggest_navigations(clean_message, tags)
        
        # Create and return the response with the extracted tags properly included
        return ChatResponse(
            message=advice_message,
            navigations=suggested_routes,
            tags=TagsModel(**tags) if tags else TagsModel(),
            language=request_language,
            source_language=detected_language
        )
    
    def _add_to_chat_history(self, user_id: str, role: str, message: str) -> None:
        """Add a message to the user's chat history"""
        if user_id not in chat_memory:
            chat_memory[user_id] = []
        
        chat_memory[user_id].append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the last 10 messages to avoid memory issues
        if len(chat_memory[user_id]) > 10:
            chat_memory[user_id] = chat_memory[user_id][-10:]
    
    def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get the chat history for a specific user"""
        return chat_memory.get(user_id, [])
    
    def clear_chat_history(self, user_id: str) -> bool:
        """Clear the chat history for a specific user"""
        if user_id in chat_memory:
            chat_memory[user_id] = []
            return True
        return False