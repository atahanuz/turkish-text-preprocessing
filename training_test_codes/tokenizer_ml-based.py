import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import joblib
import ast

def load_tokens_from_file(input_path):
    
    text_list = []
    tokens_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                bracket_pos = line.find(', [')
                if bracket_pos != -1:
                    text = line[:bracket_pos]
                    token_list_str = line[bracket_pos + 2:] 
                    tokens = ast.literal_eval(token_list_str) #Parse the token list using ast
                    text_list.append(text)
                    tokens_list.append(tokens)

    return text_list, tokens_list

#Save the trained model and vectorizer.
def save_model(model, vectorizer, model_path='tokenizer_model.pkl', vectorizer_path='tokenizer_vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

#Load the trained model and vectorizer from disk using joblib.
def load_model(model_path='tokenizer_model.pkl', vectorizer_path='tokenizer_vectorizer.pkl'):

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"\nModel loaded from: {model_path}")
    print(f"Vectorizer loaded from: {vectorizer_path}")
    return model, vectorizer

#Extract features for a character at the given position.
def extract_character_features(text, position):

    features = {}
    text_length = len(text)
    current_char = text[position] if position < text_length else ''
    features['char'] = current_char 
    features['char_lower'] = current_char.lower()  
    features['is_whitespace'] = current_char.isspace()  # Whitespace is a natural token boundary
    features['is_punctuation'] = current_char in '.,!?;:()[]{}"\'-/'  # Punctuation often marks token boundaries or standalone tokens
    features['is_digit'] = current_char.isdigit()  # Digits form separate tokens like dates, numbers, prices etc.
    features['is_letter'] = current_char.isalpha()  # Letters typically continue within a token
    features['is_upper'] = current_char.isupper()  # Uppercase may indicate proper nouns or acronyms with different tokenization rules
    char_prev_1 = text[position - 1] if position >= 1 else '<START>'
    char_prev_2 = text[position - 2] if position >= 2 else '<START>'
    char_next_1 = text[position + 1] if position + 1 < text_length else '<END>'
    char_next_2 = text[position + 2] if position + 2 < text_length else '<END>'
    features['char_prev_1'] = char_prev_1  # Previous character provides context for boundary decisions
    features['char_prev_2'] = char_prev_2  # Two characters back helps capture longer patterns
    features['char_next_1'] = char_next_1  # Next character helps predict if current position ends a token
    features['char_next_2'] = char_next_2  # Two characters ahead provides lookahead context for complex boundaries
    features['bigram_left'] = char_prev_1 + current_char  # Character pairs capture common endings or prefixes
    features['bigram_right'] = current_char + char_next_1  # Character pairs help identify continuation or boundary patterns
    features['trigram'] = char_prev_1 + current_char + char_next_1  # Three-character sequences capture richer contextual patterns
    features['prev_is_whitespace'] = char_prev_1.isspace() if char_prev_1 != '<START>' else False  # Whitespace before often means token start nearby
    features['next_is_whitespace'] = char_next_1.isspace() if char_next_1 != '<END>' else False  # Whitespace after strongly indicates current position is token boundary
    features['prev_is_punctuation'] = char_prev_1 in '.,!?;:()[]{}"\'-/' if char_prev_1 != '<START>' else False  # Punctuation before may indicate token boundary was just passed
    features['next_is_punctuation'] = char_next_1 in '.,!?;:()[]{}"\'-/' if char_next_1 != '<END>' else False  # Punctuation after suggests current token ends here
    features['next_is_upper'] = char_next_1.isupper() if char_next_1 != '<END>' else False  # Uppercase next may indicate new token proper noun or sentence start
    features['next_is_letter'] = char_next_1.isalpha() if char_next_1 != '<END>' else False  # Letter next suggests token continues
    context_window_start = max(0, position - 5)
    context_window_end = min(text_length, position + 6)
    context_window = text[context_window_start:context_window_end]
    features['in_url_pattern'] = bool(re.search(r'https?://', context_window))  # URLs should remain as single tokens despite containing special characters
    features['in_email_pattern'] = bool(re.search(r'@.*\.', context_window))  # Email addresses should be kept together as one token
    features['in_hashtag_pattern'] = context_window.startswith('#') or (position > 0 and text[position-1] == '#')  # Hashtags are single tokens in social media text
    features['apostrophe_pattern'] = current_char == "'" and char_next_1.isalpha()  # Apostrophe within word (like "Ali'nin") should not split
    features['surrounded_by_digits'] = (char_prev_1.isdigit() if char_prev_1 != '<START>' else False) and (char_next_1.isdigit() if char_next_1 != '<END>' else False)  # Characters between digits often stay within number tokens (100.50, 14:30)
    
    return features

#Convert annotated texts and token lists into training examples.
def prepare_training_data(annotated_texts, annotated_tokens_list):
    all_features = []
    all_labels = []
    for text_idx, raw_text in enumerate(annotated_texts):
        gold_tokens = annotated_tokens_list[text_idx]
        reconstructed_text = ' '.join(gold_tokens)
        token_boundaries = set()
        current_position = 0
        for token in gold_tokens:
            token_start = reconstructed_text.find(token, current_position)
            if token_start != -1:
                token_end = token_start + len(token) - 1
                if token_end < len(reconstructed_text):
                    token_boundaries.add(token_end)
                current_position = token_start + len(token)
        
        for char_position in range(len(raw_text)):
            features = extract_character_features(raw_text, char_position)
            label = 1 if char_position in token_boundaries else 0
            all_features.append(features)
            all_labels.append(label)
    
    return all_features, all_labels
#Convert annotated texts and token lists into training data
def prepare_training_data(annotated_texts, annotated_tokens_list):
    all_features = []
    all_labels = []
    for text_idx, raw_text in enumerate(annotated_texts):
        gold_tokens = annotated_tokens_list[text_idx]
        # Find actual positions of tokens in the raw text (not reconstructed)
        token_boundaries = set()
        current_position = 0
        for token in gold_tokens:
            # Locate this token in the raw text starting from last found position
            token_position = raw_text.find(token, current_position) 
            if token_position != -1:
                # Mark the last character of this token as a boundary
                token_end_position = token_position + len(token) - 1
                token_boundaries.add(token_end_position)
                # Continue searching after this token
                current_position = token_position + len(token)
            else:
                # if token not found skip
                print(f"Token '{token}' not found in text '{raw_text}'")
        # Extract features for each character position in the raw text
        for char_position in range(len(raw_text)):
            features = extract_character_features(raw_text, char_position)
            label = 1 if char_position in token_boundaries else 0
            all_features.append(features)
            all_labels.append(label)

    return all_features, all_labels

#Train a logistic regression model for tokenization.
def train_tokenizer_model(training_texts, training_tokens_list):
    print("Extracting features from training data")
    feature_dicts, labels = prepare_training_data(training_texts, training_tokens_list)
    print(f"Total training examples: {len(feature_dicts)}")
    print(f"Positive examples (boundaries): {sum(labels)}")
    print(f"Negative examples (non-boundaries): {len(labels) - sum(labels)}")
    
    vectorizer = DictVectorizer(sparse=True)
    feature_matrix = vectorizer.fit_transform(feature_dicts)
    print("Training logistic regression model...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    model.fit(feature_matrix, labels)
    train_accuracy = model.score(feature_matrix, labels)
    print(f"Training accuracy: {train_accuracy:.4f}")
    return model, vectorizer

#Tokenize input text using the trained model.
def tokenize_text(text, model, vectorizer):

    if not text or len(text.strip()) == 0:
        return []
    
    feature_dicts = []
    for position in range(len(text)):
        features = extract_character_features(text, position)
        feature_dicts.append(features)
    
    feature_matrix = vectorizer.transform(feature_dicts)
    predictions = model.predict(feature_matrix)
    
    tokens = []
    current_token = ""  
    for char_idx, char in enumerate(text):
        current_token += char  
        if predictions[char_idx] == 1 or char_idx == len(text) - 1:
            if current_token.strip():
                tokens.append(current_token)
            current_token = ""
    
    if current_token.strip():
        tokens.append(current_token)
    
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if token:
            cleaned_tokens.append(token)
    
    return cleaned_tokens

#Calculate precision, recall, and F1 score for tokenization.
def calculate_evaluation_metrics(predicted_tokens, gold_tokens):
    predicted_set = set(predicted_tokens)
    gold_set = set(gold_tokens)
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def main():
    print("\n" + "=" * 60 + "ML-Based Logistic Regression Tokenizer" + "=" * 60 + "\n")
    print()

    # Load training data from file
    train_file = "../data/tr_boun-ud/tr_boun-ud-train_tokens.txt"
    test_file = "../data/tr_boun-ud/tr_boun-ud-test_tokens.txt"

    print(f"Loading training data from: {train_file}")
    training_texts, training_tokens = load_tokens_from_file(train_file)
    print(f"Loaded {len(training_texts)} training samples")

    print(f"\nLoading test data from: {test_file}")
    test_texts, test_tokens = load_tokens_from_file(test_file)
    print(f"Loaded {len(test_texts)} test samples\n")

    print(f"Training with {len(training_texts)} sentences\n")

    trained_model, feature_vectorizer = train_tokenizer_model(training_texts, training_tokens)

    # Save the trained model
    save_model(trained_model, feature_vectorizer)

    print("\n" + "=" * 60 + "Testing Tokenizer on Sample Texts" + "=" * 60 + "\n")
    print("\n" + "=" * 60 + "Evaluating on Test Data" + "=" * 60 + "\n")

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # Evaluate on all test samples
    for idx in range(len(test_texts)):
        predicted = tokenize_text(test_texts[idx], trained_model, feature_vectorizer)
        gold = test_tokens[idx]

        precision, recall, f1 = calculate_evaluation_metrics(predicted, gold)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    print(f"Overall Test Metrics (on {len(test_texts)} samples):")
    print(f"  Precision: {total_precision / len(test_texts):.4f}")
    print(f"  Recall: {total_recall / len(test_texts):.4f}")
    print(f"  F1 Score: {total_f1 / len(test_texts):.4f}")
    #user test 
    """
    print("\n" + "=" * 60)
    print("Interactive Tokenization")
    print("=" * 60 + "\n")
    print("Enter Turkish text to tokenize (or 'quit' to exit):\n")
    
    while True:
        user_input = input(">>> ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting tokenizer.")
            break
        
        if user_input.strip():
            tokens = tokenize_text(user_input, trained_model, feature_vectorizer)
            print(f"Tokens: {tokens}\n")
    """

if __name__ == "__main__":
    main()