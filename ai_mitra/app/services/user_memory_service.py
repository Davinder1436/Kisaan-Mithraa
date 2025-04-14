# app/services/user_memory_service.py
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import random

from app.models.user_models import UserProfile, UserMessage, UserProfileTag
import re
from typing import Dict, List, Any
from app.models.user_models import UserProfileTag

logger = logging.getLogger(__name__)

class UserMemoryService:
    """Service for managing user memory and profile tags"""
    
    def __init__(self, profile_tags_path: str = "app/data/profile_tags.json", user_profiles_dir: str = "app/data/user_profiles"):
        """Initialize the User Memory Service"""
        self.profile_tags_path = profile_tags_path
        self.user_profiles_dir = user_profiles_dir
        self.profile_tags = self._load_profile_tags()
        
        # Ensure user profiles directory exists
        os.makedirs(self.user_profiles_dir, exist_ok=True)
    
    def _load_profile_tags(self) -> Dict[str, List[str]]:
        """Load available profile tags from the JSON store"""
        try:
            with open(self.profile_tags_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading profile tags: {e}")
            # Return empty dictionary if file doesn't exist or has errors
            return {}
    
    def _get_user_profile_path(self, user_id: str) -> str:
        """Get the path to a user's profile file"""
        return os.path.join(self.user_profiles_dir, f"{user_id}.json")
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get a user's profile, creating it if it doesn't exist"""
        profile_path = self._get_user_profile_path(user_id)
        
        # Try to load existing profile
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                    return UserProfile(**profile_data)
            except Exception as e:
                logger.error(f"Error loading user profile for {user_id}: {e}")
                # Create new profile if loading fails
                return UserProfile(user_id=user_id)
        
        # Create new profile if it doesn't exist
        return UserProfile(user_id=user_id)
    
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save a user's profile to storage"""
        profile_path = self._get_user_profile_path(profile.user_id)
        
        try:
            # Update last_active timestamp
            profile.last_active = datetime.now().isoformat()
            
            # Save profile to file
            with open(profile_path, 'w') as f:
                json.dump(profile.dict(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving user profile for {profile.user_id}: {e}")
            return False
    
    def add_message_to_history(self, user_id: str, role: str, content: str) -> bool:
        """Add a message to the user's chat history"""
        profile = self.get_user_profile(user_id)
        
        # Create new message
        new_message = UserMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to history
        profile.chat_history.append(new_message)
        
        # Increment conversation count if it's a user message
        if role == "user":
            profile.conversation_count += 1
        
        # Save updated profile
        return self.save_user_profile(profile)
    
    def update_profile_tags(self, user_id: str, new_tags: Dict[str, Dict[str, Any]]) -> bool:
        """Update profile tags for a user"""
        profile = self.get_user_profile(user_id)
        
        # Process and add each new tag
        for category, tag_info in new_tags.items():
            # Create a new tag object
            new_tag = UserProfileTag(
                category=category,
                value=tag_info["value"],
                confidence=tag_info.get("confidence", 0.7),
                source=tag_info.get("source", "inferred"),
                last_updated=datetime.now().isoformat()
            )
            
            # Add or update tag in profile
            profile.profile_tags[category] = new_tag
        
        # Save updated profile
        return self.save_user_profile(profile)
    
    def select_representative_tags(self, user_id: str) -> List[Dict[str, Any]]:
        """Select only high-confidence tags for a user without a fixed count"""
        profile = self.get_user_profile(user_id)
        
        # Filter for high-confidence tags only (0.6 or higher)
        high_confidence_tags = [
            {
                "category": tag.category,
                "value": tag.value,
                "confidence": tag.confidence
            }
            for tag in profile.profile_tags.values()
            if tag.confidence >= 0.6  # Only include tags with significant confidence
        ]
        
        # Sort by confidence (highest first)
        sorted_tags = sorted(high_confidence_tags, key=lambda x: x["confidence"], reverse=True)
        
        # Prioritize diversity in categories
        selected_tags = []
        categories_used = set()
        
        # First pass: select highest confidence tag from each category
        for tag in sorted_tags:
            if tag["category"] not in categories_used:
                selected_tags.append(tag)
                categories_used.add(tag["category"])
        
        # Second pass: add any remaining high confidence tags (0.8 or higher)
        for tag in sorted_tags:
            if tag["confidence"] >= 0.8 and tag not in selected_tags:
                selected_tags.append(tag)
        
        # Update selected tags in profile
        profile.selected_tags = [f"{tag['category']}:{tag['value']}" for tag in selected_tags]
        self.save_user_profile(profile)
        
        return selected_tags
    def analyze_message_for_tags(self, message: str, current_tags: Dict[str, UserProfileTag] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a message to identify potential profile tags with more robust accuracy
        """
        if not current_tags:
            current_tags = {}
        
        new_tags = {}
        message_lower = message.lower()
        
        # -------- Extract location/region information --------
        # Check for Indian states and regions
        indian_states = {
            "Punjab": "northern_plains",
            "Haryana": "northern_plains", 
            "Uttar Pradesh": "gangetic_plains",
            "Bihar": "gangetic_plains",
            "West Bengal": "eastern_region",
            "Assam": "northeast_region",
            "Arunachal Pradesh": "northeast_region",
            "Madhya Pradesh": "central_highlands",
            "Chhattisgarh": "central_highlands",
            "Maharashtra": "western_ghats",
            "Karnataka": "western_ghats",
            "Kerala": "coastal_region",
            "Tamil Nadu": "coastal_region",
            "Andhra Pradesh": "coastal_region",
            "Telangana": "deccan_plateau",
            "Rajasthan": "arid_region",
            "Gujarat": "arid_region"
        }
        
        for state, region in indian_states.items():
            state_lower = state.lower()
            if state_lower in message_lower:
                new_tags["farming_regions"] = {
                    "value": region,
                    "confidence": 0.7,
                    "source": "message_mention"
                }
                break
        
        # -------- Extract language preferences --------
        # Check for language indicators
        language_indicators = {
            "english": ["english", "speak english", "english speaking", "communicate in english"],
            "hindi": ["hindi", "speak hindi", "hindi speaking", "communicate in hindi"],
            "punjabi": ["punjabi", "speak punjabi", "punjabi speaking"],
            "marathi": ["marathi", "speak marathi", "marathi speaking"],
            "tamil": ["tamil", "speak tamil", "tamil speaking"],
            "telugu": ["telugu", "speak telugu", "telugu speaking"]
        }
        
        for lang, indicators in language_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                new_tags["language_preference"] = {
                    "value": lang,
                    "confidence": 0.8,
                    "source": "message_mention"
                }
                break
        
        # Check for Hindi script directly
        hindi_chars = ['\u0900', '\u0901', '\u0902', '\u0903']
        if any(char in message for char in hindi_chars):
            new_tags["language_preference"] = {
                "value": "hindi",
                "confidence": 0.9,
                "source": "message_language"
            }
        # Default to English for ASCII text
        elif message.isascii() and len(message) > 20 and "language_preference" not in new_tags:
            new_tags["language_preference"] = {
                "value": "english",
                "confidence": 0.7,
                "source": "message_language"
            }
        
        # -------- Extract farming type information --------
        # Check for specific farming types
        farming_type_indicators = {
            "organic_farmer": ["organic", "natural farming", "no chemicals", "without pesticides"],
            "commercial_farmer": ["commercial", "market", "selling", "profit", "business"],
            "progressive_farmer": ["new techniques", "modern", "technology", "innovative"],
            "contract_farmer": ["contract", "agreement", "company", "corporate"]
        }
        
        for farmer_type, indicators in farming_type_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                new_tags["farmer_types"] = {
                    "value": farmer_type,
                    "confidence": 0.7,
                    "source": "message_mention"
                }
                break
        
        # -------- Extract farm size and estimate farmer type --------
        # Look for explicit farm size mentions
        farm_size_pattern = r'(\d+)\s*(acre|hectare|bigha|beegha)'
        farm_size_match = re.search(farm_size_pattern, message_lower)
        
        if farm_size_match:
            size = int(farm_size_match.group(1))
            unit = farm_size_match.group(2)
            
            # Convert to acres if needed
            if unit == "hectare":
                size = size * 2.47  # 1 hectare = 2.47 acres
            elif unit in ["bigha", "beegha"]:
                size = size * 0.625  # Using North Indian bigha on average
            
            # Determine farmer type based on land size
            if size <= 2.5:
                new_tags["farmer_types"] = {
                    "value": "marginal_farmer",
                    "confidence": 0.75,
                    "source": "inferred_from_size"
                }
            elif size <= 5:
                new_tags["farmer_types"] = {
                    "value": "small_farmer",
                    "confidence": 0.75,
                    "source": "inferred_from_size"
                }
            elif size <= 25:
                new_tags["farmer_types"] = {
                    "value": "medium_farmer",
                    "confidence": 0.75,
                    "source": "inferred_from_size"
                }
            else:
                new_tags["farmer_types"] = {
                    "value": "large_farmer",
                    "confidence": 0.75,
                    "source": "inferred_from_size"
                }
        
        # -------- Extract technology adoption level --------
        # Check for technology indicators
        tech_indicators = {
            "tech_resistant": ["traditional only", "not comfortable with technology", "avoid technology"],
            "tech_curious": ["interested in learning", "want to try", "new technologies"],
            "tech_adopter": ["using technology", "adopted", "digital", "mobile app", "smartphone"],
            "tech_innovator": ["latest technology", "cutting edge", "precision agriculture", "farm automation"]
        }
        
        for tech_level, indicators in tech_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                new_tags["technology_adoption"] = {
                    "value": tech_level,
                    "confidence": 0.7,
                    "source": "message_mention"
                }
                break
        
        # If we see mentions of tractors, mechanization, etc., infer at least tech_adopter
        if any(term in message_lower for term in ["tractor", "mechanization", "machine", "equipment", "technology", "app"]):
            if "technology_adoption" not in new_tags:
                new_tags["technology_adoption"] = {
                    "value": "tech_adopter", 
                    "confidence": 0.65,
                    "source": "inferred_from_context"
                }
        
        # -------- Extract irrigation methods --------
        irrigation_methods = {
            "drip": ["drip irrigation", "drip system", "drip lines", "drippers"],
            "sprinkler": ["sprinkler", "sprinkle system", "spray irrigation"],
            "canal": ["canal irrigation", "canal water", "water channels"],
            "tube_well": ["tube well", "borewell", "ground water", "pump"],
            "rainfed": ["rainfed", "rain dependent", "monsoon dependent", "without irrigation"]
        }
        
        for method, indicators in irrigation_methods.items():
            if any(indicator in message_lower for indicator in indicators):
                new_tags["irrigation_methods"] = {
                    "value": method,
                    "confidence": 0.75,
                    "source": "message_mention"
                }
                break
        
        # -------- Return all new tags, but don't override high-confidence existing tags with lower confidence ones --------
        filtered_new_tags = {}
        for category, tag_info in new_tags.items():
            # Only include if we don't already have this category with higher confidence
            if (category not in current_tags or 
                tag_info["confidence"] >= current_tags[category].confidence or
                current_tags[category].value != tag_info["value"]):
                
                filtered_new_tags[category] = tag_info
        
        return filtered_new_tags
    def get_recent_chat_context(self, user_id: str, message_count: int = 5) -> List[Dict[str, Any]]:
        """Get recent chat history for context"""
        profile = self.get_user_profile(user_id)
        
        # Get the most recent messages
        recent_messages = profile.chat_history[-message_count:] if profile.chat_history else []
        
        # Convert to format expected by the chatbot
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in recent_messages
        ]