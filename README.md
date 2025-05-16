# Algerian Real Estate Chatbot - Training & Deployment

This project provides a comprehensive solution for training and deploying a multilingual chatbot specialized in Algerian real estate. The chatbot can understand Arabic, French, English, and especially Algerian dialect (Darija).

## Features

- Multi-language intent recognition
- Data augmentation for Algerian dialect
- Context-aware conversation flow
- User preference tracking
- TensorFlow-based neural network
- Deployment-ready setup for Heroku

## Directory Structure

```
.
├── .bolt/                  # Configuration files
│   └── config.json         # Intent definitions
├── app.py                  # Flask API server
├── model/                  # Directory for trained model
├── model_trainer.py        # Script to train the model
├── test_chatbot.py         # Script to test the model
├── requirements.txt        # Dependencies
├── Procfile                # Heroku deployment file
└── README.md               # This file
```

## Setup & Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Training the Model

To train the chatbot model, run:

```bash
python model_trainer.py --intents .bolt/config.json
```

This will:
1. Load the intents from the specified file
2. Augment the data with additional patterns
3. Preprocess the text data
4. Train a neural network model
5. Save the model and metadata to the `model/` directory

## Testing the Model

After training, you can test the model with:

```bash
# Test with example queries
python test_chatbot.py

# Or use interactive mode
python test_chatbot.py --interactive
```

In interactive mode, you can chat with the bot to test its responses.

## Running the API Server

To run the Flask API server locally:

```bash
python app.py
```

The server will be available at http://localhost:5000

## API Endpoints

- `POST /api/message`: Send a message to the chatbot
  - Request body: `{ "user_id": "unique-user-id", "message": "Your message here" }`
  - Response: `{ "response": "Bot response", "context": "current_context", "preferences": {...} }`

- `POST /api/reset`: Reset a conversation
  - Request body: `{ "user_id": "unique-user-id" }`
  - Response: `{ "status": "success", "message": "Conversation reset successfully" }`

## Deployment to Heroku

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku: `heroku login`
3. Create a new Heroku app: `heroku create your-app-name`
4. Initialize a Git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
5. Deploy to Heroku: `git push heroku master`
6. Scale the app: `heroku ps:scale web=1`

## Customizing the Chatbot

To customize the chatbot, edit the intents in `.bolt/config.json`. Each intent has:
- `tag`: The intent identifier
- `patterns`: Example phrases that trigger the intent
- `responses`: Possible bot responses
- `context`: The next context to set after this intent

After modifying intents, retrain the model using `model_trainer.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.