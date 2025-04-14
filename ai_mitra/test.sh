#!/bin/bash

# Test script for enhanced CropConnect chatbot with user memory and profile tagging

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Enhanced CropConnect Chatbot API...${NC}"
echo

# Test 1: First message from a new user (in English)
echo -e "${YELLOW}Test 1: First message from a new user (English)${NC}"
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have a 5-acre farm in Punjab where I grow wheat and rice. I use traditional irrigation methods but want to try drip irrigation. Can you advise me?", 
    "language": "en",
    "user_id": "test_user_1"
  }'

echo -e "\n\n"

# Test 2: Follow-up message from the same user (in English)
echo -e "${YELLOW}Test 2: Follow-up message from the same user (English)${NC}"
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What would be the cost of setting up drip irrigation for my wheat crop?", 
    "language": "en",
    "user_id": "test_user_1"
  }'

echo -e "\n\n"

# Test 3: Message from a different user (in Hindi)
echo -e "${YELLOW}Test 3: Message from a different user (Hindi)${NC}"
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "मैं हरियाणा में कपास की खेती करता हूं। कीटों से बचाव के लिए क्या करूं?", 
    "language": "hi",
    "user_id": "test_user_2"
  }'

echo -e "\n\n"

# Test 4: Get user profile for the first user
echo -e "${YELLOW}Test 4: Get user profile for the first user${NC}"
curl -X GET http://localhost:8000/api/v1/user-profile/test_user_1

echo -e "\n\n"

# Test 5: Get chat history for the first user
echo -e "${YELLOW}Test 5: Get chat history for the first user${NC}"
curl -X GET http://localhost:8000/api/v1/user-history/test_user_1

echo -e "\n\n"

# Test 6: Try different crop topic to see tag updates
echo -e "${YELLOW}Test 6: User asks about a different crop${NC}"
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to try growing some organic vegetables next to my wheat field. What are some good options for Punjab?", 
    "language": "en",
    "user_id": "test_user_1"
  }'

echo -e "\n\n"

# Test 7: Check updated profile tags
echo -e "${YELLOW}Test 7: Check updated profile tags after organic mention${NC}"
curl -X GET http://localhost:8000/api/v1/user-profile/test_user_1

echo -e "\n\n"

# Test 8: Manually update user tags
echo -e "${YELLOW}Test 8: Manually update user tags${NC}"
curl -X POST http://localhost:8000/api/v1/user-tags/test_user_1 \
  -H "Content-Type: application/json" \
  -d '{
    "income_levels": {
      "value": "above_average_income",
      "confidence": 0.9,
      "source": "manual"
    },
    "education_levels": {
      "value": "agricultural_degree",
      "confidence": 0.9,
      "source": "manual"
    }
  }'

echo -e "\n\n"

# Test 9: Health check
echo -e "${YELLOW}Test 9: Health check endpoint${NC}"
curl -X GET http://localhost:8000/api/v1/health

echo -e "\n\n${GREEN}Testing complete!${NC}"