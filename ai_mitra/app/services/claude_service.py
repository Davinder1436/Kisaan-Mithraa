import json
import os
import httpx
from typing import Dict, Any, Tuple

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE

class ClaudeService:
    """Service for interacting with the Claude API directly"""
    
    def __init__(self):
        # API settings
        self.api_key = ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    async def generate_farming_response(
        self, message: str, language: str, app_context: Dict[str, Any], chat_history=None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate farming advice and extract tags from the user's message"""
        
        # Create system prompt with app context and task description
        system_prompt = self._create_system_prompt(app_context, language)
        
        # Prepare the request payload
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": CLAUDE_TEMPERATURE,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }
        
        # Make the direct API call to Claude
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                
                # Check if the request was successful
                if response.status_code != 200:
                    return f"API Error: {response.status_code}", self._default_tags()
                
                response_data = response.json()
                
                # Extract the text content
                response_text = ""
                if 'content' in response_data and len(response_data['content']) > 0:
                    response_text = response_data['content'][0]['text']
                
                # Try to parse the response as JSON
                try:
                    response_json = json.loads(response_text)
                    
                    # Check if it has the expected structure
                    if isinstance(response_json, dict) and "message" in response_json and "tags" in response_json:
                        return response_json["message"], response_json["tags"]
                    else:
                        # If it's JSON but doesn't have the expected structure
                        return response_text, self._default_tags()
                except json.JSONDecodeError:
                    # If it's not valid JSON, return the text as is
                    return response_text, self._default_tags()
                
        except Exception as e:
            error_msg = f"Error communicating with Claude API: {str(e)}"
            print(error_msg)
            return error_msg, self._default_tags()
    
    def _create_system_prompt(self, app_context: Dict[str, Any], language: str) -> str:
        """Create system prompt with app context and task description"""
        # Language name mapping
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "pa": "Punjabi",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi"
        }
        language_name = language_names.get(language, "English")
        
        # Extract key navigation info for the system prompt
        navigation_summary = ""
        for nav in app_context.get("navigation_contexts", [])[:10]:
            navigation_summary += f"- {nav.get('route')}: {nav.get('title')} - {nav.get('description')}\n"
        
        # Adjusted instructions to specify English tags even for non-English content
        return f"""You are the agricultural assistant for the CropConnect app, a platform that connects farmers through cooperatives, shared resources, and knowledge exchange.

Your task is to:
1. Provide helpful, practical farming advice based on the user's question
2. Respond in {language_name} language for the main message content
3. Analyze the query to extract relevant entities like crops, locations, farming issues, etc.

The app has the following navigation options:
{navigation_summary}

CRITICAL INSTRUCTIONS - YOUR RESPONSE FORMAT:
You MUST return your response as a VALID JSON object with exactly this structure:
{{
    "message": "Your detailed farming advice here in {language_name}",
    "tags": {{
        "crops": ["crop1", "crop2"],  // List of crops mentioned IN ENGLISH ONLY
        "city": "city_name",          // Location mentioned IN ENGLISH ONLY
        "topics": ["topic1", "topic2"], // Agricultural topics detected IN ENGLISH ONLY
        "issues": ["issue1", "issue2"], // Farming problems detected IN ENGLISH ONLY
        "seasons": ["season1", "season2"] // Seasons mentioned IN ENGLISH ONLY
    }}
}}

IMPORTANT: Even when writing the message in {language_name}, all tag values MUST be in English only.
For example, if responding in Hindi about "गेहूं" (wheat), the tag should be in English as "wheat".

DO NOT include any text outside of this JSON structure.
DO NOT include markdown code blocks or any wrapping syntax.
DO NOT explain the JSON format in your response.
ONLY return the valid JSON object itself.

Make sure your response is practical, specific, and actionable for farmers in India.
"""
    
    def _default_tags(self) -> Dict[str, Any]:
        """Default empty tags structure"""
        return {
            "crops": [],
            "city": None,
            "topics": [],
            "issues": [],
            "seasons": []
        }