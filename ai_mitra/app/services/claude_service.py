import json
import os
import httpx
import logging
from typing import Dict, Any, Tuple, List, Optional

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE

# Set up logger
logger = logging.getLogger(__name__)

class ClaudeService:
    """Service for interacting with the Claude API directly with enhanced error handling"""
    
    def __init__(self):
        # API settings
        self.api_key = ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Log initialization status (without exposing the actual key)
        if self.api_key:
            logger.info(f"ClaudeService initialized with API key (length: {len(self.api_key)})")
        else:
            logger.error("ClaudeService initialized with empty API key!")
    
    async def generate_farming_response(
        self, message: str, language: str, app_context: Dict[str, Any], chat_history=None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate farming advice and extract tags from the user's message"""
        
        # Create system prompt with app context and task description
        system_prompt = self._create_system_prompt(app_context, language)
        
        # Make the direct API call to Claude
        message_response = await self.get_claude_response(
            message,
            system_prompt,
            CLAUDE_MODEL,
            CLAUDE_MAX_TOKENS,
            CLAUDE_TEMPERATURE
        )
        
        # Check if there was an error with the API call
        if message_response.startswith("Error:") or message_response.startswith("Network error:"):
            logger.error(f"Error in main response: {message_response}")
            return message_response, self._default_tags()
        
        # Try to parse the response as JSON
        try:
            response_json = json.loads(message_response)
            
            # Check if it has the expected structure
            if isinstance(response_json, dict) and "message" in response_json and "tags" in response_json:
                logger.info("Successfully parsed response with expected structure")
                return response_json["message"], response_json["tags"]
            else:
                # If it's JSON but doesn't have the expected structure
                logger.warning(f"Response is valid JSON but missing expected structure: {list(response_json.keys())}")
                return message_response, self._default_tags()
        except json.JSONDecodeError:
            # If it's not valid JSON, return the text as is
            logger.warning("Response is not valid JSON, using as plain text")
            return message_response, self._default_tags()
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return message_response, self._default_tags()
    
    async def get_claude_response(
        self, 
        message: str,
        system_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Get a response from Claude API with enhanced error handling and debugging"""
        try:
            # Log request parameters for debugging (without showing the full system prompt)
            logger.debug(f"Claude API Request - Model: {model}, Max Tokens: {max_tokens}, Temperature: {temperature}")
            logger.debug(f"Message length: {len(message)} chars")
            logger.debug(f"System prompt length: {len(system_prompt)} chars")
            
            # Check if API key is available
            if not self.api_key:
                logger.error("API key is not set or empty")
                return "Error: API key is not configured properly"
            
            # Prepare the request payload
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": message}]
            }
            
            # Log payload structure (for debugging)
            logger.debug(f"Payload structure: {json.dumps({k: '...' if k in ['system', 'messages'] else v for k, v in payload.items()})}")
            
            # Make API call to Claude
            async with httpx.AsyncClient() as client:
                # Log API call attempt
                logger.info(f"Making API call to Claude with model {model}")
                
                # Send request to Claude API
                claude_response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                
                # Log API response status
                logger.info(f"Claude API response status: {claude_response.status_code}")
                
                # Check if the request was successful
                if claude_response.status_code != 200:
                    # Log the error response
                    error_body = claude_response.text
                    logger.error(f"API Error: Status {claude_response.status_code}, Response: {error_body[:200]}...")
                    return f"Error: API returned status code {claude_response.status_code}. Details: {error_body[:200]}"
                
                # Parse the response
                response_data = claude_response.json()
                
                # Log successful response structure
                logger.debug(f"Response structure: {list(response_data.keys())}")
                
                # Extract the text content
                if 'content' in response_data and len(response_data['content']) > 0:
                    content_item = response_data['content'][0]
                    if 'text' in content_item:
                        return content_item['text']
                    else:
                        logger.warning("Response content item does not contain 'text' field")
                        logger.debug(f"Content item structure: {content_item}")
                        return "Sorry, received unexpected response format from Claude API"
                else:
                    logger.warning("Response does not contain expected 'content' field or it's empty")
                    logger.debug(f"Response data: {json.dumps(response_data)[:200]}...")
                    return "Sorry, I couldn't generate a response due to unexpected API response format"
        except json.JSONDecodeError as jde:
            # Specific handling for JSON parse errors
            error_detail = str(jde)
            logger.error(f"Error parsing Claude API response: {error_detail}")
            return f"Error parsing response from Claude API: {error_detail}"
        except httpx.RequestError as re:
            # Network/connection related errors
            error_detail = str(re)
            logger.error(f"Network error when calling Claude API: {error_detail}")
            return f"Network error when communicating with Claude API: {error_detail}"
        except httpx.TimeoutException as te:
            # Timeout errors
            logger.error(f"Request to Claude API timed out: {str(te)}")
            return "Request to Claude API timed out. Please try again."
        except Exception as e:
            # General exception handling with detailed logging
            error_detail = str(e)
            error_type = type(e).__name__
            logger.error(f"Unexpected error ({error_type}) calling Claude API: {error_detail}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {error_type} - {error_detail}"
    
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
        prompt = f"""You are the agricultural assistant for the CropConnect app, a platform that connects farmers through cooperatives, shared resources, and knowledge exchange.

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
        logger.debug(f"Created system prompt for language: {language_name}")
        return prompt
    
    def _default_tags(self) -> Dict[str, Any]:
        """Default empty tags structure"""
        return {
            "crops": [],
            "city": None,
            "topics": [],
            "issues": [],
            "seasons": []
        }