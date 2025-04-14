import re
from langdetect import detect, LangDetectException

from app.config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE

class LanguageService:
    """Service for language detection and processing"""
    
    @staticmethod
    def detect_language(text):
        """Detect the language of the input text"""
        # First check if the user explicitly specified a language in the text
        explicit_lang = LanguageService._extract_explicit_language(text)
        if explicit_lang:
            return explicit_lang
        
        # Use langdetect for automatic detection
        try:
            detected = detect(text)
            # Map to our supported language codes
            if detected in SUPPORTED_LANGUAGES:
                return detected
            # Map 'en-us', 'en-gb', etc. to 'en'
            if '-' in detected:
                base_lang = detected.split('-')[0]
                if base_lang in SUPPORTED_LANGUAGES:
                    return base_lang
                
            # Special handling for languages that might be confused
            if detected == 'ur' and 'hi' in SUPPORTED_LANGUAGES:
                return 'hi'  # Urdu might be confused with Hindi
                
            return DEFAULT_LANGUAGE
        except LangDetectException:
            return DEFAULT_LANGUAGE
    
    @staticmethod
    def _extract_explicit_language(text):
        """Extract explicitly mentioned language from the text"""
        # Pattern to match "in language:" or "in language -" at the beginning
        pattern = r"^in\s+([a-zA-Z]+)[\s:-]"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            lang_name = match.group(1).lower()
            
            # Direct match with language codes
            if lang_name in SUPPORTED_LANGUAGES:
                return lang_name
                
            # Match with language names
            for code, name in SUPPORTED_LANGUAGES.items():
                if lang_name in name.lower():
                    return code
            
        return None
    
    @staticmethod
    def clean_language_prefix(text):
        """Remove language prefix from the text"""
        pattern = r"^in\s+([a-zA-Z]+)[\s:-]\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    @staticmethod
    def get_language_name(lang_code):
        """Get the full language name from the code"""
        return SUPPORTED_LANGUAGES.get(lang_code, "English")
