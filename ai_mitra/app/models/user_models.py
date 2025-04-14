# app/models/user_models.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class UserMessage(BaseModel):
    """Individual message in a user's chat history"""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of the message")


class UserProfileTag(BaseModel):
    """Tag representing an aspect of the user's profile"""
    category: str = Field(..., description="Category of the tag (e.g., farmer_type, region)")
    value: str = Field(..., description="Value of the tag")
    confidence: float = Field(default=0.5, description="Confidence score for this tag (0.0-1.0)")
    source: str = Field(default="inferred", description="How the tag was determined (explicit/inferred)")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the tag was last updated")


class UserProfile(BaseModel):
    """Complete user profile with chat history and profile tags"""
    user_id: str = Field(..., description="Unique identifier for the user")
    chat_history: List[UserMessage] = Field(default_factory=list, description="History of all chat messages")
    profile_tags: Dict[str, UserProfileTag] = Field(default_factory=dict, description="Profile tags for the user")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the profile was created")
    last_active: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the user was last active")
    conversation_count: int = Field(default=0, description="Count of conversations")
    selected_tags: List[str] = Field(default_factory=list, description="Tags selected to represent the user in current chat")