import os
import json
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

from app.services.language_service import LanguageService
from app.services.navigation_service import NavigationService
from app.services.user_memory_service import UserMemoryService
from app.models.user_models import UserProfile, UserMessage

# Load environment variables
load_dotenv()

# Configure settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1000"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))

# Initialize FastAPI app
app = FastAPI(
    title="CropConnect Chatbot API",
    description="API for the CropConnect agricultural chatbot with user memory",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    user_id: Optional[str] = None

# Response model
class ChatResponse(BaseModel):
    message: str
    navigations: List[str] = []
    tags: Dict[str, Any] = {}
    language: str
    source_language: Optional[str] = None
    profile_tags: List[Dict[str, Any]] = []

# Create service instances
language_service = LanguageService()
navigation_service = NavigationService()
user_memory_service = UserMemoryService()

# Load app context
try:
    with open("app/data/app_context.json", "r") as f:
        APP_CONTEXT = json.load(f)
except Exception as e:
    print(f"Warning: Could not load app_context.json: {e}")
    APP_CONTEXT = {
        "navigation_contexts": [],
        "meta": {}
    }

@app.get("/")
async def root():
    return {"message": "Welcome to CropConnect Chatbot API with User Memory"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response with user memory and profile tags
    Using a two-step approach for more reliable parsing with fixed navigation
    """
    try:
        # Detect source language
        source_language = language_service.detect_language(request.message)
        
        # Get user profile and chat history if user_id is provided
        user_context = []
        user_tags = []
        
        if request.user_id:
            # Get recent chat history
            user_context = user_memory_service.get_recent_chat_context(request.user_id)
            
            # Get current profile for the user
            user_profile = user_memory_service.get_user_profile(request.user_id)
            
            # Store user message in history
            user_memory_service.add_message_to_history(
                request.user_id, 
                "user", 
                request.message
            )
            
            # Analyze message for new profile tags
            new_tags = user_memory_service.analyze_message_for_tags(
                request.message,
                user_profile.profile_tags
            )
            
            # Update user profile with new tags
            if new_tags:
                user_memory_service.update_profile_tags(request.user_id, new_tags)
        
        # STEP 1: Generate the main response message
        message_prompt = create_message_prompt(
            request.language, 
            APP_CONTEXT, 
            user_context, 
            request.user_id
        )
        
        # Make first API call to get the main message
        message_response = await get_claude_response(
            request.message,
            message_prompt,
            CLAUDE_MODEL,
            CLAUDE_MAX_TOKENS,
            CLAUDE_TEMPERATURE
        )
        
        # STEP 2: Generate the tags in a separate call for reliability
        tags_prompt = create_tags_prompt(
            request.language,
            message_response
        )
        
        # Make second API call to get just the tags
        tags_response = await get_claude_response(
            request.message,
            tags_prompt,
            CLAUDE_MODEL,
            500,  # Fewer tokens needed for tags
            0.2    # Lower temperature for more consistent tags
        )
        
        # Parse the tags from the response
        tags = parse_tags_from_response(tags_response)
        
        # Store assistant response in user history if user_id is provided
        if request.user_id:
            user_memory_service.add_message_to_history(
                request.user_id, 
                "assistant", 
                message_response
            )
            
            # Update user profile with inferred tags from response
            inferred_tags = infer_tags_from_response(tags)
            if inferred_tags:
                user_memory_service.update_profile_tags(request.user_id, inferred_tags)
            
            # Get representative tags for this user (only high-confidence ones)
            user_tags = user_memory_service.select_representative_tags(request.user_id)
        
        # Fixed navigation options as requested
        navigations = ["/podcasts", "/community"]
        
        # Create the response
        response = ChatResponse(
            message=message_response,
            navigations=navigations,
            tags=tags,
            language=request.language,
            source_language=source_language,
            profile_tags=user_tags
        )
        
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

async def get_claude_response(
    message: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float
) -> str:
    """Get a response from Claude API with error handling"""
    try:
        # Prepare the request payload
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }
        
        # Make API call to Claude
        async with httpx.AsyncClient() as client:
            claude_response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json=payload,
                timeout=30.0
            )
            
            # Check if the request was successful
            if claude_response.status_code != 200:
                return f"Error: API returned status code {claude_response.status_code}"
            
            # Parse the response
            response_data = claude_response.json()
            
            # Extract the text content
            if 'content' in response_data and len(response_data['content']) > 0:
                return response_data['content'][0]['text']
            else:
                return "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return f"Error generating response: {str(e)}"

def create_message_prompt(language: str, app_context: Dict[str, Any], user_context: List[Dict[str, Any]], user_id: Optional[str]) -> str:
    """Create a system prompt focused on generating the main message"""
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
    
    # Generate base prompt
    base_prompt = f"""You are the agricultural assistant for the CropConnect app, a platform that connects farmers through cooperatives, shared resources, and knowledge exchange.

Your task is to:
1. Provide helpful, practical farming advice based on the user's question
2. Respond in {language_name} language

IMPORTANT INSTRUCTIONS:
- Provide a detailed, informative response to the user's query
- Include specific, actionable advice when possible
- If appropriate, mention potential follow-up resources or actions
- DO NOT include any JSON formatting in your response
- Respond ONLY with the content of your message

Format your response as a normal conversational message without any special formatting.
"""
    
    # Add user context if available
    if user_context:
        context_text = "\nPrevious conversation history:\n"
        for msg in user_context:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content']}\n"
        
        base_prompt += f"\n{context_text}\n"
        base_prompt += "Use this conversation history to provide a more personalized response.\n"
    
    # Add user profile information if available
    if user_id:
        user_profile = user_memory_service.get_user_profile(user_id)
        
        if user_profile.profile_tags:
            profile_text = "\nUser profile information:\n"
            
            for tag in user_profile.profile_tags.values():
                profile_text += f"- {tag.category}: {tag.value}\n"
            
            base_prompt += f"\n{profile_text}\n"
            base_prompt += "Use this profile information to tailor your advice to this specific farmer's context.\n"
    
    return base_prompt

def create_tags_prompt(language: str, message_content: str) -> str:
    """Create a system prompt focused on generating tags"""
    return f"""Analyze the following agricultural advice and extract relevant entities and tags from it.

Advice:
{message_content}

Return ONLY a JSON object with the following structure:
{{
    "tags": {{
        "crops": ["crop1", "crop2"],  // List of crops mentioned IN ENGLISH ONLY
        "city": "location_name",      // Location mentioned IN ENGLISH ONLY
        "topics": ["topic1", "topic2"], // Agricultural topics IN ENGLISH ONLY
        "issues": ["issue1", "issue2"], // Farming problems IN ENGLISH ONLY
        "seasons": ["season1", "season2"] // Seasons IN ENGLISH ONLY
    }}
}}

DO NOT include any explanation, just the JSON object itself.
Focus on extracting entities present in the message.
"""

def parse_tags_from_response(response_text: str) -> Dict[str, Any]:
    """Parse tags JSON from Claude's response with improved error handling"""
    # Default empty tags structure
    default_tags = {
        "crops": [],
        "city": None,
        "topics": [],
        "issues": [],
        "seasons": []
    }
    
    try:
        # Find JSON object in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            # Extract the JSON string
            json_str = response_text[json_start:json_end]
            
            # Parse JSON
            parsed_json = json.loads(json_str)
            
            # Check if it has the expected structure
            if "tags" in parsed_json:
                tags = parsed_json["tags"]
                
                # Ensure tags have the expected structure
                for key in default_tags:
                    if key not in tags:
                        tags[key] = default_tags[key]
                    elif tags[key] is None and key != "city":
                        tags[key] = []
                
                return tags
            
            # If we got JSON but without the expected structure
            return default_tags
            
        # If no JSON found
        return default_tags
        
    except json.JSONDecodeError:
        # If JSON parsing fails
        return default_tags
    except Exception as e:
        print(f"Error parsing tags: {e}")
        return default_tags
async def get_claude_response(
    message: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float
) -> str:
    """Get a response from Claude API with error handling"""
    try:
        # Prepare the request payload
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }
        
        # Make API call to Claude
        async with httpx.AsyncClient() as client:
            claude_response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json=payload,
                timeout=30.0
            )
            
            # Check if the request was successful
            if claude_response.status_code != 200:
                return f"Error: API returned status code {claude_response.status_code}"
            
            # Parse the response
            response_data = claude_response.json()
            
            # Extract the text content
            if 'content' in response_data and len(response_data['content']) > 0:
                return response_data['content'][0]['text']
            else:
                return "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return f"Error generating response: {str(e)}"

def create_system_prompt_for_message(language: str, app_context: Dict[str, Any], user_context: List[Dict[str, Any]], user_id: Optional[str]) -> str:
    """Create a system prompt focused on generating the main message"""
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
    
    # Generate base prompt
    base_prompt = f"""You are the agricultural assistant for the CropConnect app, a platform that connects farmers through cooperatives, shared resources, and knowledge exchange.

Your task is to:
1. Provide helpful, practical farming advice based on the user's question
2. Respond in {language_name} language

IMPORTANT INSTRUCTIONS:
- Provide a detailed, informative response to the user's query
- Include specific, actionable advice when possible
- If appropriate, mention potential follow-up resources or actions
- DO NOT include any JSON formatting in your response
- Respond ONLY with the content of your message

Format your response as a normal conversational message without any special formatting.
"""
    
    # Add user context if available
    if user_context:
        context_text = "\nPrevious conversation history:\n"
        for msg in user_context:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content']}\n"
        
        base_prompt += f"\n{context_text}\n"
        base_prompt += "Use this conversation history to provide a more personalized response.\n"
    
    # Add user profile information if available
    if user_id:
        user_profile = user_memory_service.get_user_profile(user_id)
        
        if user_profile.profile_tags:
            profile_text = "\nUser profile information:\n"
            
            for tag in user_profile.profile_tags.values():
                profile_text += f"- {tag.category}: {tag.value}\n"
            
            base_prompt += f"\n{profile_text}\n"
            base_prompt += "Use this profile information to tailor your advice to this specific farmer's context.\n"
    
    return base_prompt

def create_system_prompt_for_tags(language: str, message_content: str) -> str:
    """Create a system prompt focused on generating tags"""
    return f"""Analyze the following agricultural advice and extract relevant entities and tags from it.

Advice:
{message_content}

Return ONLY a JSON object with the following structure:
{{
    "tags": {{
        "crops": ["crop1", "crop2"],  // List of crops mentioned IN ENGLISH ONLY
        "city": "location_name",      // Location mentioned IN ENGLISH ONLY
        "topics": ["topic1", "topic2"], // Agricultural topics IN ENGLISH ONLY
        "issues": ["issue1", "issue2"], // Farming problems IN ENGLISH ONLY
        "seasons": ["season1", "season2"] // Seasons IN ENGLISH ONLY
    }}
}}

DO NOT include any explanation, just the JSON object itself.
Focus on extracting entities present in the message.
"""

def parse_tags_from_response(response_text: str) -> Dict[str, Any]:
    """Parse tags JSON from Claude's response with improved error handling"""
    # Default empty tags structure
    default_tags = {
        "crops": [],
        "city": None,
        "topics": [],
        "issues": [],
        "seasons": []
    }
    
    try:
        # Find JSON object in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            # Extract the JSON string
            json_str = response_text[json_start:json_end]
            
            # Parse JSON
            parsed_json = json.loads(json_str)
            
            # Check if it has the expected structure
            if "tags" in parsed_json:
                tags = parsed_json["tags"]
                
                # Ensure tags have the expected structure
                for key in default_tags:
                    if key not in tags:
                        tags[key] = default_tags[key]
                    elif tags[key] is None and key != "city":
                        tags[key] = []
                
                return tags
            
            # If we got JSON but without the expected structure
            return default_tags
            
        # If no JSON found
        return default_tags
        
    except json.JSONDecodeError:
        # If JSON parsing fails
        return default_tags
    except Exception as e:
        print(f"Error parsing tags: {e}")
        return default_tags
@app.get("/api/v1/user-profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get a user's profile information"""
    try:
        profile = user_memory_service.get_user_profile(user_id)
        
        # Convert to a more API-friendly format
        profile_data = {
            "user_id": profile.user_id,
            "created_at": profile.created_at,
            "last_active": profile.last_active,
            "conversation_count": profile.conversation_count,
            "profile_tags": [
                {
                    "category": tag.category,
                    "value": tag.value,
                    "confidence": tag.confidence,
                    "source": tag.source,
                    "last_updated": tag.last_updated
                }
                for tag in profile.profile_tags.values()
            ],
            "chat_history_length": len(profile.chat_history),
            "selected_tags": profile.selected_tags
        }
        
        return profile_data
    except Exception as e:
        print(f"Error retrieving user profile: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving user profile: {str(e)}"}
        )

@app.get("/api/v1/user-history/{user_id}")
async def get_user_history(user_id: str, limit: int = 10):
    """Get a user's chat history"""
    try:
        profile = user_memory_service.get_user_profile(user_id)
        
        # Get the most recent messages
        recent_messages = profile.chat_history[-limit:] if profile.chat_history else []
        
        # Format for API response
        history = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in recent_messages
        ]
        
        return {
            "user_id": user_id,
            "history": history,
            "total_messages": len(profile.chat_history)
        }
    except Exception as e:
        print(f"Error retrieving user history: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving user history: {str(e)}"}
        )

@app.delete("/api/v1/user-history/{user_id}")
async def clear_user_history(user_id: str):
    """Clear a user's chat history"""
    try:
        profile = user_memory_service.get_user_profile(user_id)
        profile.chat_history = []
        
        # Save the updated profile
        user_memory_service.save_user_profile(profile)
        
        return {"user_id": user_id, "status": "cleared"}
    except Exception as e:
        print(f"Error clearing user history: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error clearing user history: {str(e)}"}
        )

@app.post("/api/v1/user-tags/{user_id}")
async def update_user_tags(user_id: str, tags: Dict[str, Dict[str, Any]]):
    """Manually update tags for a user"""
    try:
        success = user_memory_service.update_profile_tags(user_id, tags)
        
        if success:
            return {"user_id": user_id, "status": "updated", "tags": tags}
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to update user tags"}
            )
    except Exception as e:
        print(f"Error updating user tags: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error updating user tags: {str(e)}"}
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

def create_system_prompt(language: str, app_context: Dict[str, Any], user_context: List[Dict[str, Any]], user_id: Optional[str]) -> str:
    """Create a system prompt with user context and app information"""
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
    
    # Extract navigation info
    navigation_summary = ""
    for nav in app_context.get("navigation_contexts", [])[:10]:
        navigation_summary += f"- {nav.get('route')}: {nav.get('title')} - {nav.get('description')}\n"
    
    # Generate base prompt
    base_prompt = f"""You are the agricultural assistant for the CropConnect app, a platform that connects farmers through cooperatives, shared resources, and knowledge exchange.

Your task is to:
1. Provide helpful, practical farming advice based on the user's question
2. Respond in {language_name} language for the main message content
3. Analyze the query to extract relevant entities like crops, locations, farming issues, etc.

The app has the following navigation options:
{navigation_summary}

RESPONSE FORMAT - YOUR RESPONSE MUST BE A VALID JSON OBJECT:
{{
    "message": "Your detailed farming advice here in {language_name}",
    "tags": {{
        "crops": ["crop1", "crop2"],  // List of crops mentioned IN ENGLISH ONLY
        "city": "city_name",          // Location mentioned IN ENGLISH ONLY
        "topics": ["topic1", "topic2"], // Agricultural topics IN ENGLISH ONLY
        "issues": ["issue1", "issue2"], // Farming problems IN ENGLISH ONLY
        "seasons": ["season1", "season2"] // Seasons IN ENGLISH ONLY
    }}
}}

Return ONLY the JSON object, nothing else.
"""
    
    # Add user context if available
    if user_context:
        context_text = "\nPrevious conversation history:\n"
        for msg in user_context:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content']}\n"
        
        base_prompt += f"\n{context_text}\n"
        base_prompt += "Use this conversation history to provide a more personalized response.\n"
    
    # Add user profile information if available
    if user_id:
        user_profile = user_memory_service.get_user_profile(user_id)
        
        if user_profile.profile_tags:
            profile_text = "\nUser profile information:\n"
            
            for tag in user_profile.profile_tags.values():
                profile_text += f"- {tag.category}: {tag.value}\n"
            
            base_prompt += f"\n{profile_text}\n"
            base_prompt += "Use this profile information to tailor your advice to this specific farmer's context.\n"
    
    return base_prompt

def infer_tags_from_response(response_tags: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Infer user profile tags from the response tags with improved accuracy"""
    inferred_tags = {}
    
    # Infer farmer type from crops and topics
    crops = response_tags.get("crops", [])
    topics = response_tags.get("topics", [])
    
    # Infer farmer_type based on topics and crops
    if any(topic in ["organic farming", "natural farming", "organic"] for topic in topics):
        inferred_tags["farmer_types"] = {
            "value": "organic_farmer",
            "confidence": 0.7,
            "source": "inferred_from_interests"
        }
    elif any(topic in ["commercial farming", "market", "profit", "business"] for topic in topics):
        inferred_tags["farmer_types"] = {
            "value": "commercial_farmer",
            "confidence": 0.7,
            "source": "inferred_from_interests"
        }
    elif len(crops) >= 3:
        inferred_tags["farmer_types"] = {
            "value": "multi_crop_farmer",
            "confidence": 0.65,
            "source": "inferred_from_interests"
        }
    
    # Infer technology adoption based on topics
    tech_topics = [
        "tractors", "farm mechanization", "farm equipment", "machinery", 
        "automation", "sensor", "digital", "smart farming", "precision agriculture"
    ]
    
    if any(tech in topics for tech in tech_topics):
        inferred_tags["technology_adoption"] = {
            "value": "tech_adopter",
            "confidence": 0.7,
            "source": "inferred_from_topics"
        }
        
        # If looking at advanced technologies, upgrade to tech_innovator
        advanced_tech = ["precision agriculture", "smart farming", "automation", "sensor", "IoT", "drone"]
        if any(adv_tech in topic for topic in topics for adv_tech in advanced_tech):
            inferred_tags["technology_adoption"] = {
                "value": "tech_innovator",
                "confidence": 0.7,
                "source": "inferred_from_topics"
            }
    
    # Infer farming_region if city is present
    city = response_tags.get("city")
    if city:
        # Simple mapping of states/regions to farming regions
        region_mapping = {
            "Punjab": "northern_plains",
            "Haryana": "northern_plains",
            "Uttar Pradesh": "gangetic_plains",
            "Bihar": "gangetic_plains",
            "West Bengal": "eastern_region",
            "Assam": "northeast_region",
            "Madhya Pradesh": "central_highlands",
            "Maharashtra": "western_ghats",
            "Kerala": "coastal_region",
            "Tamil Nadu": "coastal_region",
            "Rajasthan": "arid_region",
            "Gujarat": "arid_region"
        }
        
        # Check if the city matches any state/region
        for state, region in region_mapping.items():
            if state.lower() in city.lower():
                inferred_tags["farming_regions"] = {
                    "value": region,
                    "confidence": 0.7,
                    "source": "inferred_from_location"
                }
                break
    
    # Infer farming challenges based on issues
    issues = response_tags.get("issues", [])
    if issues:
        challenge_mapping = {
            "pest": "pest_management",
            "disease": "pest_management",
            "water": "water_scarcity",
            "drought": "water_scarcity",
            "irrigation": "water_scarcity",
            "soil": "soil_health",
            "fertility": "soil_health",
            "erosion": "soil_health",
            "market": "market_access",
            "price": "price_fluctuation",
            "labor": "labor_shortage",
            "cost": "input_costs",
            "storage": "storage_facilities",
            "transport": "transportation"
        }
        
        for issue in issues:
            issue_lower = issue.lower()
            for keyword, challenge in challenge_mapping.items():
                if keyword in issue_lower:
                    inferred_tags["farming_challenges"] = {
                        "value": challenge,
                        "confidence": 0.7,
                        "source": "inferred_from_issues"
                    }
                    break
    
    # Infer irrigation methods from topics
    irrigation_topics = ["irrigation", "water management", "drip irrigation", "sprinkler", "flood irrigation"]
    if any(irr_topic in topics for irr_topic in irrigation_topics):
        for topic in topics:
            if "drip" in topic.lower():
                inferred_tags["irrigation_methods"] = {
                    "value": "drip",
                    "confidence": 0.7,
                    "source": "inferred_from_topics"
                }
                break
            elif "sprinkler" in topic.lower():
                inferred_tags["irrigation_methods"] = {
                    "value": "sprinkler",
                    "confidence": 0.7,
                    "source": "inferred_from_topics"
                }
                break
            elif "canal" in topic.lower():
                inferred_tags["irrigation_methods"] = {
                    "value": "canal",
                    "confidence": 0.7,
                    "source": "inferred_from_topics"
                }
                break
            elif "tube well" in topic.lower() or "borewell" in topic.lower():
                inferred_tags["irrigation_methods"] = {
                    "value": "tube_well",
                    "confidence": 0.7,
                    "source": "inferred_from_topics"
                }
                break
    
    # Infer seasons
    seasons = response_tags.get("seasons", [])
    if seasons:
        season_mapping = {
            "kharif": "kharif",
            "rabi": "rabi",
            "zaid": "zaid",
            "summer": "zaid",
            "monsoon": "kharif",
            "winter": "rabi"
        }
        
        for season in seasons:
            season_lower = season.lower()
            for keyword, mapped_season in season_mapping.items():
                if keyword in season_lower:
                    inferred_tags["crop_cycle_focus"] = {
                        "value": mapped_season,
                        "confidence": 0.7,
                        "source": "inferred_from_seasons"
                    }
                    break
    
    # Infer equipment usage from topics related to farm machinery
    equipment_topics = ["tractor", "mechanization", "farm equipment", "machinery", "harvester", "thresher"]
    if any(equip_topic in " ".join(topics).lower() for equip_topic in equipment_topics):
        if "tractor" in " ".join(topics).lower():
            inferred_tags["equipment_usage"] = {
                "value": "tractor_owner",
                "confidence": 0.65,
                "source": "inferred_from_topics"
            }
        else:
            inferred_tags["equipment_usage"] = {
                "value": "small_machinery",
                "confidence": 0.65,
                "source": "inferred_from_topics"
            }
    
    return inferred_tags
if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = os.getenv("APP_RELOAD", "true").lower() == "true"
    
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)