# CropConnect Chatbot

A minimal implementation of the CropConnect agricultural chatbot that provides farming advice and contextual navigation suggestions.

## Setup

1. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

2. Update your .env file with your Anthropic API key.

3. Start the server:
   ```
   python3 -m app.main
   ```

The API will be available at http://localhost:8000

## API Usage

### Chat Endpoint

Send POST requests to `/api/v1/chat` with the following JSON:

```json
{
  "message": "How do I grow tomatoes in Maharashtra?",
  "language": "en"  // Optional: en, hi, pa, ta, te, mr
}
```

The response will include farming advice, navigation suggestions, and extracted tags.
