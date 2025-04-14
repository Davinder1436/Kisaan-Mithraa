import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.language_service import LanguageService

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_language_detection():
    """Test the language detection functionality"""
    language_service = LanguageService()
    
    # Test detection with explicit language mention
    assert language_service.detect_language("in Hindi: मैं फसल उगाना चाहता हूं") in ["hi", "en"]
    
    # Test language prefix cleaning
    cleaned = language_service.clean_language_prefix("in Hindi: मैं फसल उगाना चाहता हूं")
    assert "in Hindi" not in cleaned
