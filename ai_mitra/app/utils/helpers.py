import json
import re
from typing import Dict, Any

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from text that might contain other content"""
    # Find the first JSON-like pattern in the text
    json_pattern = r'({[\s\S]*})'
    matches = re.findall(json_pattern, text)
    
    if not matches:
        return {}
    
    # Try to parse each match until we find valid JSON
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return {}
