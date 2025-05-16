import pickle
import numpy as np
import tensorflow as tf
import argparse

def load_model_and_metadata():
    """Load the trained model and metadata"""
    try:
        # Load model
        model = tf.keras.models.load_model('model/chatbot_model')
        
        # Load metadata
        with open('model/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load preprocessor
        with open('model/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        return model, metadata, preprocessor
    except Exception as e:
        print(f"Error loading model and metadata: {e}")
        return None, None, None

def sentence_to_bag_of_words(sentence, words, preprocessor):
    """Convert a sentence to a bag of words"""
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

def predict_intent(sentence, model, words, classes, preprocessor, threshold=0.25):
    """Predict the intent of a sentence"""
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

def get_response(intents_list, intents_data, user_context=None):
    """Get a response based on the predicted intent"""
    if not intents_list:
        return {
            'response': "I'm not sure I understand. Could you rephrase that?",
            'context': user_context,
            'new_context': user_context
        }
    
    tag = intents_list[0]['intent']
    probability = intents_list[0]['probability']
    
    # Find the intent in intents_data
    for intent in intents_data:
        if intent['tag'] == tag:
            # Get a random response
            import random
            response = random.choice(intent['responses'])
            
            # Get the new context if available
            new_context = None
            if 'context' in intent and intent['context']:
                new_context = intent['context'][0]
            
            return {
                'response': response,
                'context': user_context,
                'new_context': new_context,
                'confidence': probability
            }
    
    return {
        'response': "I'm not sure I understand. Could you rephrase that?",
        'context': user_context,
        'new_context': user_context
    }

def chat_with_bot():
    """Interactive chat interface for testing the bot"""
    # Load model and metadata
    model, metadata, preprocessor = load_model_and_metadata()
    
    if model is None or metadata is None or preprocessor is None:
        print("Error: Could not load model and metadata. Make sure the model is trained.")
        return
    
    # Extract metadata
    words = metadata['words']
    classes = metadata['classes']
    intents = metadata['intents']
    
    # Initialize context
    user_context = None
    
    print("=" * 50)
    print("Algerian Real Estate Chatbot")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Thank you for using our service.")
            break
        
        # Predict intent
        intents_list = predict_intent(user_input, model, words, classes, preprocessor)
        
        # Get response
        response_data = get_response(intents_list, intents, user_context)
        
        # Update context
        user_context = response_data['new_context']
        
        # Print response
        print(f"Bot: {response_data['response']}")
        
        # Print additional info if in debug mode
        if args.debug:
            if intents_list:
                print(f"DEBUG: Detected intent: {intents_list[0]['intent']} ({intents_list[0]['probability']:.4f})")
            else:
                print("DEBUG: No intent detected")
            print(f"DEBUG: Current context: {user_context}")
            print("-" * 30)

def test_with_examples():
    """Test the bot with predefined examples"""
    # Load model and metadata
    model, metadata, preprocessor = load_model_and_metadata()
    
    if model is None or metadata is None or preprocessor is None:
        print("Error: Could not load model and metadata. Make sure the model is trained.")
        return
    
    # Extract metadata
    words = metadata['words']
    classes = metadata['classes']
    intents = metadata['intents']
    
    # Test queries
    test_queries = [
        "Hello, how are you?",
        "Bonjour, comment ça va?",
        "السلام عليكم",
        "Sba7 l khir, nhws 3la dar f Oran",
        "I want to buy a villa",
        "Je cherche à louer un appartement",
        "نريد شراء شقة في الجزائر العاصمة",
        "ana nchri dar fi annaba"
    ]
    
    # Set up context tracking
    user_context = None
    
    print("=" * 50)
    print("Testing Chatbot with Example Queries")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Predict intent
        intents_list = predict_intent(query, model, words, classes, preprocessor)
        
        if intents_list:
            print(f"Detected intent: {intents_list[0]['intent']} ({intents_list[0]['probability']:.4f})")
            
            # Get response
            response_data = get_response(intents_list, intents, user_context)
            
            print(f"Response: {response_data['response']}")
            print(f"Current context: {response_data['context']}")
            print(f"New context: {response_data['new_context']}")
            
            # Update context for next query
            user_context = response_data['new_context']
        else:
            print("No intent predicted.")
        
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the trained chatbot model')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information')
    args = parser.parse_args()
    
    if args.interactive:
        chat_with_bot()
    else:
        test_with_examples()