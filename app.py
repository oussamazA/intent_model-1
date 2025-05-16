import os
import numpy as np
import pickle
import random
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if model directory exists
if not os.path.exists('model'):
    os.makedirs('model', exist_ok=True)

# Global variables to store model and metadata
model = None
words = None
classes = None
intents = None
preprocessor = None

# Load the model and metadata if they exist
def load_model_and_metadata():
    global model, words, classes, intents, preprocessor
    
    try:
        # Try loading Keras v3 format first
        keras_v3_path = os.path.join('model', 'chatbot_model.keras')
        if os.path.exists(keras_v3_path):
            model = tf.keras.models.load_model(keras_v3_path)
        else:
            # Fallback to SavedModel format
            model = tf.keras.models.load_model(os.path.join('model', 'chatbot_model'))
        
        # Load metadata
        metadata_path = os.path.join('model', 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        words = metadata['words']
        classes = metadata['classes']
        intents = metadata['intents']
        
        # Load preprocessor
        preprocessor_path = os.path.join('model', 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
            
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# User sessions to store context
user_sessions = {}

# Function to convert a sentence to a bag of words
def sentence_to_bag_of_words(sentence, words, preprocessor):
    # Tokenize and preprocess the sentence
    tokens = preprocessor.preprocess(sentence)
    
    # Initialize bag of words
    bag = [0] * len(words)
    
    # Fill the bag of words array
    for token in tokens:
        for i, word in enumerate(words):
            if word == token:
                bag[i] = 1
    
    return np.array([bag])

# Function to predict the intent of a sentence
def predict_intent(sentence, model, words, classes, preprocessor, threshold=0.25):
    # Convert sentence to bag of words
    bow = sentence_to_bag_of_words(sentence, words, preprocessor)
    
    # Make prediction
    results = model.predict(bow)[0]
    
    # Filter out predictions below threshold
    results = [[i, r] for i, r in enumerate(results) if r > threshold]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return list of intents and probabilities
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': float(r[1])})
    
    return return_list

# Function to get a response based on the predicted intent
def get_response(intents_list, intents_data, user_context=None, user_preferences=None):
    if not intents_list:
        return {
            'text': "I'm not sure I understand. Could you rephrase that?",
            'context': user_context,
            'new_context': user_context,
            'preferences': user_preferences or {}
        }
    
    tag = intents_list[0]['intent']
    probability = intents_list[0]['probability']
    
    # Extract user preferences if applicable
    preferences = extract_preferences(tag)
    if user_preferences:
        preferences.update(user_preferences)
    
    # Find the intent in intents_data
    for intent in intents_data:
        if intent['tag'] == tag:
            # Get a random response
            response = random.choice(intent['responses'])
            
            # Replace placeholders with user preferences
            response = replace_placeholders(response, preferences)
            
            # Get the new context if available
            new_context = None
            if 'context' in intent and intent['context']:
                new_context = intent['context'][0]
            
            return {
                'text': response,
                'context': user_context,
                'new_context': new_context,
                'preferences': preferences,
                'confidence': probability
            }
    
    return {
        'text': "I'm not sure I understand. Could you rephrase that?",
        'context': user_context,
        'new_context': user_context,
        'preferences': user_preferences or {}
    }

# Helper function to extract preferences from intent tag and message
def extract_preferences(tag):
    # This is a simplified version - in a real app, you would extract
    # actual preferences from the user's message based on the intent tag
    preferences = {}
    
    # Example extraction based on tag
    if 'property_type' in tag:
        preferences['propertyType'] = 'property type'
    elif 'location' in tag:
        preferences['location'] = 'location'
    elif 'price' in tag:
        preferences['price'] = 'price range'
    
    return preferences

# Helper function to replace placeholders in response
def replace_placeholders(response, preferences):
    if '%PROPERTY%' in response and 'propertyType' in preferences:
        response = response.replace('%PROPERTY%', preferences['propertyType'])
    if '%LOCATION%' in response and 'location' in preferences:
        response = response.replace('%LOCATION%', preferences['location'])
    if '%PRICE%' in response and 'price' in preferences:
        response = response.replace('%PRICE%', preferences['price'])
    
    return response

@app.route('/api/message', methods=['POST'])
def process_message():
    # Check if model is loaded
    if model is None:
        if not load_model_and_metadata():
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The chatbot model is not available. Please train the model first.'
            }), 500
    
    data = request.json
    user_id = data.get('user_id', 'default_user')
    message = data.get('message', '')
    
    # Get or create user session
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'context': None,
            'preferences': {}
        }
    
    # Get user context and preferences
    user_context = user_sessions[user_id]['context']
    user_preferences = user_sessions[user_id]['preferences']
    
    # Predict intent
    intents_list = predict_intent(message, model, words, classes, preprocessor)
    
    # Get response
    response_data = get_response(intents_list, intents, user_context, user_preferences)
    
    # Update user session
    if response_data['new_context']:
        user_sessions[user_id]['context'] = response_data['new_context']
    
    if response_data['preferences']:
        user_sessions[user_id]['preferences'].update(response_data['preferences'])
    
    return jsonify({
        'response': response_data['text'],
        'context': response_data['new_context'],
        'preferences': user_sessions[user_id]['preferences'],
        'confidence': response_data.get('confidence', 0)
    })

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    
    # Reset user session
    user_sessions[user_id] = {
        'context': None,
        'preferences': {}
    }
    
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })

@app.route('/', methods=['GET'])
def index():
    return "Algerian Real Estate Chatbot API is running!"

if __name__ == '__main__':
    # Try to load model and metadata
    load_model_and_metadata()
    
    # Get port from environment variable for Heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
