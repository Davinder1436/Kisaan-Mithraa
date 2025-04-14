import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Claude Model Configuration
CLAUDE_MODEL = "claude-3-opus-20240229"
CLAUDE_MAX_TOKENS = 1000
CLAUDE_TEMPERATURE = 0.7

# App Configuration
APP_CONTEXT_PATH = Path(__file__).parent / "data" / "app_context.json"

def load_app_context():
    """Load the app navigation context from JSON file"""
    try:
        with open(APP_CONTEXT_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading app context: {e}")
        return {}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi"
}

# Default language
DEFAULT_LANGUAGE = "en"
