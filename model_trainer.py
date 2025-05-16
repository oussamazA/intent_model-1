import numpy as np
import tensorflow as tf
import random
import pickle
import json
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import re
import argparse

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

class MultilingualPreprocessor:
    def __init__(self):
        # Load stemmers and stopwords for each language
        self.en_stemmer = PorterStemmer()
        self.en_stopwords = set(nltk.corpus.stopwords.words('english'))
        
        try:
            self.fr_stopwords = set(nltk.corpus.stopwords.words('french'))
        except:
            nltk.download('stopwords')
            self.fr_stopwords = set(nltk.corpus.stopwords.words('french'))
        
        # Arabic stopwords (common words)
        self.ar_stopwords = {'من', 'إلى', 'عن', 'على', 'في', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'هناك', 'هنا', 'أنا', 'أنت', 'هو', 'هي'}
        
        # Darija common words to exclude
        self.darija_stopwords = {'ana', 'nta', 'nti', 'hna', 'had', 'hadi', 'rah', 'kayn', 'fi', 'min', 'mea', 'nchri', 'nkri'}
        
        # Common punctuation and special characters to remove
        self.punctuation = string.punctuation + '،؛؟«»''""\\u200f\\u200e'
        
        # Create a table for faster punctuation removal
        self.trans_table = str.maketrans('', '', self.punctuation)
    
    def detect_language(self, text):
        """Detect the language of the text"""
        try:
            # Check for Arabic characters first
            if any(ord(c) > 1000 for c in text):
                return 'ar'
            
            # Check for Darija patterns
            if any(word in text.lower() for word in ['nchri', 'nkri', 'dar', 'wach', '3la', '7aja']):
                return 'darija'
            
            # Try to detect French words
            if any(word in text.lower() for word in ['bonjour', 'salut', 'je', 'tu', 'nous', 'vous', 'ils', 'acheter']):
                return 'fr'
            
            # Default to English
            return 'en'
        except:
            return 'en'  # Default to English if detection fails
    
    def preprocess(self, text):
        """Preprocess text based on detected language"""
        # Convert to lowercase (for non-Arabic)
        if not any(ord(c) > 1000 for c in text):  # Not Arabic
            text = text.lower()
        
        # Remove punctuation
        text = text.translate(self.trans_table)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Detect language
        lang = self.detect_language(text)
        
        # Apply language-specific processing
        if lang == 'en':
            # Remove English stopwords and apply stemming
            tokens = [self.en_stemmer.stem(token) for token in tokens if token not in self.en_stopwords]
        elif lang == 'fr':
            # Remove French stopwords (no stemming for now)
            tokens = [token for token in tokens if token not in self.fr_stopwords]
        elif lang == 'ar':
            # For Arabic, just remove stopwords (no stemming)
            tokens = [token for token in tokens if token not in self.ar_stopwords]
        elif lang == 'darija':
            # For Darija, remove stopwords
            tokens = [token for token in tokens if token not in self.darija_stopwords]
        
        return tokens

def augment_pattern(pattern, count=3):
    """Generate augmented versions of a pattern"""
    augmentations = [pattern]
    
    # 1. Common darija letter replacements
    replacements = {
        'a': '3', 'h': '7', 's': '$', 'q': '9', 't': 't\'', 
        'e': 'i', 'o': 'ou', 'ch': 'ch\'', 'kh': '5', 'gh': 'r', 
        'dh': 'd', 'th': 't', 'ai': 'ay'
    }
    
    # Apply random replacements
    for _ in range(min(2, count)):
        new_pattern = pattern
        for original, replacement in replacements.items():
            if original in new_pattern and random.random() > 0.5:
                new_pattern = new_pattern.replace(original, replacement)
        if new_pattern != pattern and new_pattern not in augmentations:
            augmentations.append(new_pattern)
    
    # 2. Word order variations (for multi-word patterns)
    words = pattern.split()
    if len(words) > 1:
        for _ in range(min(2, count)):
            shuffled = words.copy()
            random.shuffle(shuffled)
            new_pattern = ' '.join(shuffled)
            if new_pattern not in augmentations:
                augmentations.append(new_pattern)
    
    # 3. Shortened forms common in texting
    if len(pattern) > 4:
        shortened = ''.join([c for c in pattern if c.lower() not in 'aeiou'])
        if len(shortened) > 2 and shortened not in augmentations:
            augmentations.append(shortened)
    
    return augmentations

def augment_intents_data(intents):
    """Augment all patterns in the intents data"""
    augmented_intents = []
    
    for intent in tqdm(intents, desc="Augmenting intents"):
        # Create a copy of the intent
        augmented_intent = intent.copy()
        original_patterns = intent['patterns']
        augmented_patterns = []
        
        # For each pattern in the intent
        for pattern in original_patterns:
            # Add the original pattern
            augmented_patterns.append(pattern)
            
            # Determine if this is likely Darija/Arabic/French based on tag or content
            is_french = "_fr" in intent['tag'] or any(c in pattern for c in ['é', 'è', 'ê', 'ç'])
            is_arabic = "_ar" in intent['tag'] or any(ord(c) > 1000 for c in pattern)  # Arabic characters have higher Unicode values
            is_darija = not is_arabic and (
                "nchri" in pattern or "nkri" in pattern or "dar" in pattern or 
                "3" in pattern or "7" in pattern
            )
            
            # Apply different augmentation strategies based on language
            rule_augmentations = augment_pattern(
                pattern, 
                count=5 if is_darija else (3 if is_french else 2)
            )
            augmented_patterns.extend(rule_augmentations[1:])  # Skip the first one which is the original
        
        # Update the intent with augmented patterns
        augmented_intent['patterns'] = list(set(augmented_patterns))  # Remove duplicates
        augmented_intents.append(augmented_intent)
        
    return augmented_intents

def prepare_training_data(intents, preprocessor):
    """Prepare training data from intents"""
    words = []
    classes = []
    documents = []
    
    # Process each intent
    for intent in intents:
        tag = intent['tag']
        # Add class to classes list
        if tag not in classes:
            classes.append(tag)
        
        # Process each pattern in the intent
        for pattern in intent['patterns']:
            # Tokenize and preprocess the pattern
            tokens = preprocessor.preprocess(pattern)
            # Add tokens to words list
            words.extend(tokens)
            # Add the document (tokens + intent)
            documents.append((tokens, tag))
    
    # Remove duplicates and sort
    words = sorted(list(set(words)))
    classes = sorted(classes)
    
    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create training data from words, classes, and documents"""
    # Initialize training data
    training = []
    # Create an empty array for output
    output_empty = [0] * len(classes)
    
    # Training set: bag of words for each document
    for doc in documents:
        # Initialize bag of words
        bag = []
        # List of tokenized words for the pattern
        pattern_words = doc[0]
        
        # Create bag of words array
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)
        
        # Output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        # Add the bag of words and output row to training data
        training.append([bag, output_row])
    
    # Shuffle the training data
    random.shuffle(training)
    
    # Convert to numpy arrays
    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])
    
    return train_x, train_y

def build_model(input_shape, output_shape):
    """Build the neural network model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Dense(128, input_shape=(input_shape,), activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add dropout to prevent overfitting
        
        # Hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main(intents_file):
    # Create output directory
    os.makedirs('model', exist_ok=True)
    
    # Load intents data
    with open(intents_file, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    
    intents = intents_data.get('intents', [])
    print(f"Loaded {len(intents)} intents")
    
    # Create the preprocessor
    preprocessor = MultilingualPreprocessor()
    
    # Augment the intents data
    augmented_intents = augment_intents_data(intents)
    
    # Calculate pattern stats
    total_original_patterns = sum(len(intent['patterns']) for intent in intents)
    total_augmented_patterns = sum(len(intent['patterns']) for intent in augmented_intents)
    
    print(f"Original patterns: {total_original_patterns}")
    print(f"Augmented patterns: {total_augmented_patterns}")
    print(f"Pattern increase: {total_augmented_patterns - total_original_patterns} (+{((total_augmented_patterns/total_original_patterns) - 1)*100:.2f}%)")
    
    # Prepare the training data
    words, classes, documents = prepare_training_data(augmented_intents, preprocessor)
    print(f"Total unique words: {len(words)}")
    print(f"Total classes: {len(classes)}")
    print(f"Total documents: {len(documents)}")
    
    # Create the training data
    train_x, train_y = create_training_data(words, classes, documents)
    print(f"Training data shape: X = {train_x.shape}, Y = {train_y.shape}")
    
    # Split data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(len(words), len(classes))
    
    # Create callback for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        train_x, train_y,
        epochs=200,
        batch_size=16,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(val_x, val_y, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save the model
    print("Saving model...")
    model.save('model/chatbot_model')
    
    # Save metadata
    metadata = {
        'words': words,
        'classes': classes,
        'intents': augmented_intents
    }
    
    with open('model/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save preprocessor
    with open('model/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("Model training complete!")
    print("Model and metadata saved to 'model/' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a chatbot model for Algerian real estate')
    parser.add_argument('--intents', type=str, default='.bolt/config.json', 
                        help='Path to the intents JSON file')
    args = parser.parse_args()
    
    main(args.intents)