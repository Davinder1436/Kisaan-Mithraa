# app/services/language_service.py
import re
from typing import Optional

class LanguageService:
    """Service for language detection and processing"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of the input text"""
        # First check if the user explicitly specified a language in the text
        explicit_lang = LanguageService._extract_explicit_language(text)
        if explicit_lang:
            return explicit_lang
        
        # Simple detection based on character sets
        # Check for Hindi (Devanagari script)
        hindi_range = range(0x0900, 0x097F)
        if any(ord(char) in hindi_range for char in text):
            return "hi"
        
        # Check for Punjabi (Gurmukhi script)
        punjabi_range = range(0x0A00, 0x0A7F)
        if any(ord(char) in punjabi_range for char in text):
            return "pa"
        
        # Check for Tamil script
        tamil_range = range(0x0B80, 0x0BFF)
        if any(ord(char) in tamil_range for char in text):
            return "ta"
        
        # Check for Telugu script
        telugu_range = range(0x0C00, 0x0C7F)
        if any(ord(char) in telugu_range for char in text):
            return "te"
        
        # Check for Marathi (uses Devanagari script like Hindi)
        # This is a simplification as we would need more contextual analysis to distinguish from Hindi
        if any(ord(char) in hindi_range for char in text) and ("marathi" in text.lower() or "maharashtra" in text.lower()):
            return "mr"
        
        # Default to English
        return "en"
    
    @staticmethod
    def _extract_explicit_language(text: str) -> Optional[str]:
        """Extract explicitly mentioned language from the text"""
        # Pattern to match "in language:" or "in language -" at the beginning
        pattern = r"^in\s+([a-zA-Z]+)[\s:-]"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            lang_name = match.group(1).lower()
            
            # Map language names to codes
            language_mapping = {
                "hindi": "hi",
                "english": "en",
                "punjabi": "pa",
                "tamil": "ta",
                "telugu": "te",
                "marathi": "mr"
            }
            
            return language_mapping.get(lang_name)
        
        return None
    
    @staticmethod
    def clean_language_prefix(text: str) -> str:
        """Remove language prefix from the text"""
        pattern = r"^in\s+([a-zA-Z]+)[\s:-]\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get the full language name from the code"""
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "pa": "Punjabi",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi"
        }
        return language_names.get(lang_code, "English")