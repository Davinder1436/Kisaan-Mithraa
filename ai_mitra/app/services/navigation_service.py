from typing import List, Dict, Any
import json
import re

from app.config import load_app_context

class NavigationService:
    """Service for suggesting relevant app navigation based on user queries"""
    
    def __init__(self):
        # Load the app context
        self.app_context = load_app_context()
        self.navigation_contexts = self.app_context.get("navigation_contexts", [])
        self.meta = self.app_context.get("meta", {})
        
        # Extract keywords for efficient matching
        self._extract_keywords()
    
    def _extract_keywords(self):
        """Extract keywords from navigation contexts for efficient matching"""
        self.context_keywords = {}
        self.navigation_by_route = {}
        
        for context in self.navigation_contexts:
            route = context.get("route")
            if not route:
                continue
                
            self.navigation_by_route[route] = context
            
            # Gather keywords from intent keywords and user needs
            keywords = []
            keywords.extend(context.get("intent_keywords", []))
            keywords.extend(context.get("user_needs", []))
            
            # Store unique keywords for this route
            self.context_keywords[route] = list(set(keywords))
    
    def suggest_navigations(self, message: str, extracted_entities: Dict[str, Any]) -> List[str]:
        """Suggest app navigation routes based on the user message and extracted entities"""
        suggestions = []
        
        # Score each navigation context based on relevance to the message
        route_scores = {}
        
        for route, keywords in self.context_keywords.items():
            score = 0
            
            # Check for direct keyword matches
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', message.lower()):
                    score += 2
            
            # Check entity matches
            context = self.navigation_by_route[route]
            
            # Check for crop relevance
            if "crops" in extracted_entities and extracted_entities["crops"]:
                if route in ["/resource-pool", "/podcasts", "/chatbot"]:
                    score += 3
            
            # Check for city/location relevance
            if "city" in extracted_entities and extracted_entities["city"]:
                if route in ["/search-cooperatives", "/nearby-cooperatives"]:
                    score += 3
            
            # Special cases for specific routes
            if route == "/podcasts" and ("learn" in message.lower() or "information" in message.lower() or 
                                        any(topic in ["cultivation", "farming techniques", "agricultural education"] 
                                           for topic in extracted_entities.get("topics", []))):
                score += 3
                
            if route == "/resource-pool" and ("equipment" in message.lower() or "tools" in message.lower()):
                score += 2
                
            if route == "/chatbot" and (any(term in message.lower() for term in ["help", "advice", "problem", "issue"]) or
                                       extracted_entities.get("issues", [])):
                score += 4
                
            if route == "/community" and any(term in message.lower() for term in ["discuss", "talk", "other farmers"]):
                score += 2
                
            # Specific rules for crops and issues
            if extracted_entities.get("crops", []) and route == "/chatbot":
                score += 3
                
            if extracted_entities.get("issues", []) and route == "/community":
                score += 2
            
            # Store the score
            route_scores[route] = score
        
        # Get top scoring routes (max 3)
        sorted_routes = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
        suggestions = [route for route, score in sorted_routes if score > 0][:3]
        
        # Always include chatbot as a navigation option if not already included and we have other suggestions
        if "/chatbot" not in suggestions and suggestions:
            suggestions.append("/chatbot")
            
        # If no routes have scores, provide some default suggestions
        if not suggestions:
            suggestions = ["/chatbot", "/podcasts", "/community"]
        
        return suggestions