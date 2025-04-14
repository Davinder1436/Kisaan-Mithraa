from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

class TagsModel(BaseModel):
    """Model for response tags"""
    crops: List[str] = Field(default_factory=list, description="Detected crop types")
    city: Optional[str] = Field(None, description="Detected city or location")
    topics: List[str] = Field(default_factory=list, description="Detected agricultural topics")
    issues: List[str] = Field(default_factory=list, description="Detected farming issues or problems")
    seasons: List[str] = Field(default_factory=list, description="Detected seasons")

    class Config:
        # This allows extra fields in the JSON
        extra = "allow"
        
        # Sample structure
        schema_extra = {
            "example": {
                "crops": ["tomato", "potato"],
                "city": "Maharashtra",
                "topics": ["irrigation", "fertilization"],
                "issues": ["pest infestation"],
                "seasons": ["kharif"]
            }
        }

    def dict(self, *args, **kwargs):
        # Ensure all list fields are initialized
        result = super().dict(*args, **kwargs)
        for field in ["crops", "topics", "issues", "seasons"]:
            if field not in result or result[field] is None:
                result[field] = []
        return result

class ChatResponse(BaseModel):
    """Model for chat responses"""
    message: str = Field(..., description="Response message")
    navigations: List[str] = Field(default_factory=list, description="Suggested app navigation routes")
    tags: TagsModel = Field(default_factory=TagsModel, description="Detected tags")
    language: str = Field(..., description="Language of the response")
    source_language: Optional[str] = Field(None, description="Detected source language of the query")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "1. *जलवायु और समय*: मिर्ची के लिए गर्म जलवायु आवश्यक है...",
                "navigations": ["/podcasts", "/search-cooperatives"],
                "tags": {
                    "crops": ["chillies"],
                    "city": "jalandhar",
                    "topics": ["cultivation"],
                    "issues": [],
                    "seasons": ["summer"]
                },
                "language": "hi",
                "source_language": "hi"
            }
        }