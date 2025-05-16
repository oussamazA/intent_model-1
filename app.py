import os
import numpy as np
import pickle
import random
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Global holders
model = None
words = None
classes = None
intents = None
preprocessor = None

# Try loading model + metadata from disk
def load_model_and_metadata():
    global model, words, classes, intents, preprocessor

    model_path = 'model/chatbot_model.keras'
    metadata_path = 'model/metadata.pkl'
    preproc_path = 'model/preprocessor.pkl'

    try:
        print(f"[startup] Loading Keras model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("[startup] Model loaded.")

        print(f"[startup] Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        words = metadata.get('words', [])
        classes = metadata.get('classes', [])
        intents = metadata.get('intents', [])
        print(f"[startup] Loaded metadata: {len(words)} words, {len(classes)} classes, {len(intents)} intents.")

        print(f"[startup] Loading preprocessor from {preproc_path}...")
        with open(preproc_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("[startup] Preprocessor loaded.")

        return True

    except Exception as e:
        print(f"[startup][ERROR] Failed to load model/metadata: {e}")
        model = None
        return False

# Simple user sessions for context/preferences
user_sessions = {}

def sentence_to_bag_of_words(sentence, words, preprocessor):
    tokens = preprocessor.preprocess(sentence)
    bag = [1 if w in tokens else 0 for w in words]
    return np.array([bag])

def predict_intent(sentence, model, words, classes, preprocessor, threshold=0.25):
    bow = sentence_to_bag_of_words(sentence, words, preprocessor)
    probs = model.predict(bow)[0]
    filtered = [(i, p) for i, p in enumerate(probs) if p > threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[i], 'probability': float(p)} for i, p in filtered]

def get_response(intents_list, intents_data, user_context=None, user_preferences=None):
    if not intents_list:
        return {
            'text': "I'm not sure I understand. Could you rephrase that?",
            'context': user_context,
            'new_context': user_context,
            'preferences': user_preferences or {}
        }

    top = intents_list[0]
    tag = top['intent']
    confidence = top['probability']

    # merge preferences
    prefs = user_preferences.copy() if user_preferences else {}
    prefs.update(extract_preferences(tag))

    for intent_def in intents_data:
        if intent_def['tag'] == tag:
            resp = random.choice(intent_def.get('responses', []))
            resp = replace_placeholders(resp, prefs)
            new_ctx = intent_def.get('context', [None])[0]
            return {
                'text': resp,
                'context': user_context,
                'new_context': new_ctx,
                'preferences': prefs,
                'confidence': confidence
            }

    return {
        'text': "I'm not sure I understand. Could you rephrase that?",
        'context': user_context,
        'new_context': user_context,
        'preferences': prefs,
        'confidence': confidence
    }

def extract_preferences(tag):
    prefs = {}
    if 'property_type' in tag:
        prefs['propertyType'] = None
    if 'location' in tag:
        prefs['location'] = None
    if 'price' in tag:
        prefs['price'] = None
    return prefs

def replace_placeholders(text, prefs):
    if '%PROPERTY%' in text and prefs.get('propertyType'):
        text = text.replace('%PROPERTY%', prefs['propertyType'])
    if '%LOCATION%' in text and prefs.get('location'):
        text = text.replace('%LOCATION%', prefs['location'])
    if '%PRICE%' in text and prefs.get('price'):
        text = text.replace('%PRICE%', prefs['price'])
    return text

@app.route('/api/message', methods=['POST'])
def process_message():
    global model, words, classes, intents, preprocessor

    if model is None:
        if not load_model_and_metadata():
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Chatbot model or metadata missing. Please train first.'
            }), 500

    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    msg = data.get('message', '')

    # init session
    session = user_sessions.setdefault(user_id, {'context': None, 'preferences': {}})
    user_context = session['context']
    user_prefs   = session['preferences']

    # run through pipeline
    intents_list = predict_intent(msg, model, words, classes, preprocessor)
    resp_data    = get_response(intents_list, intents, user_context, user_prefs)

    # update session
    if resp_data.get('new_context') is not None:
        session['context'] = resp_data['new_context']
    if resp_data.get('preferences'):
        session['preferences'].update(resp_data['preferences'])

    return jsonify({
        'response': resp_data['text'],
        'context':  resp_data['new_context'],
        'preferences': session['preferences'],
        'confidence': resp_data.get('confidence', 0)
    })

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    user_sessions[user_id] = {'context': None, 'preferences': {}}
    return jsonify({'status': 'reset'})

@app.route('/', methods=['GET'])
def index():
    return "🕌 Algerian Real Estate Chatbot API is up!"

if __name__ == '__main__':
    # pre-load at startup if possible
    load_model_and_metadata()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
