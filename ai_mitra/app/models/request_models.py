from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Model for chat requests"""
    message: str = Field(..., description="User's message")
    language: Optional[str] = Field("en", description="Language code (en, hi, pa, ta, te, mr)")
    user_id: Optional[str] = Field(None, description="User ID for tracking conversation history")
    location: Optional[str] = Field(None, description="User's location")
    context: Optional[dict] = Field(None, description="Additional context for the request")
