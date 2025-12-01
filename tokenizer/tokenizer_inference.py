import re
import joblib

def load_trained_tokenizer(model_path='tokenizer_model.pkl', vectorizer_path='tokenizer_vectorizer.pkl'):
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
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


def tokenize(text, model, vectorizer):
    if not text or len(text.strip()) == 0:
        return []
    # Extract features for each character
    feature_dicts = []
    for position in range(len(text)):
        features = extract_character_features(text, position)
        feature_dicts.append(features)
    # Transform features and predict boundaries
    feature_matrix = vectorizer.transform(feature_dicts)
    predictions = model.predict(feature_matrix)
    # Build tokens based on predictions
    tokens = []
    current_token = ""
    for char_idx, char in enumerate(text):
        current_token += char

        # If this is a boundary or the last character, end the token
        if predictions[char_idx] == 1 or char_idx == len(text) - 1:
            if current_token.strip():
                tokens.append(current_token)
            current_token = ""
    # Add any remaining token
    if current_token.strip():
        tokens.append(current_token)
    # Clean tokens
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if token:
            cleaned_tokens.append(token)

    return cleaned_tokens


def tokenize_file(input_file, output_file, model, vectorizer):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if line:
                tokens = tokenize(line, model, vectorizer)
                fout.write(' '.join(tokens) + '\n')

                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...")

    print(f"\nTokenization completed. Output written to: {output_file}")


def main():
    print("=" * 60 + "ML-Based Turkish Tokenizer - Inference" + "=" * 60)
    print()

    # Load the trained model
    model, vectorizer = load_trained_tokenizer()

    print("=" * 60 + "Interactive Tokenization" + "=" * 60)
    print("\nCommands:")
    print("  - Type text to tokenize it")
    print("  - Type 'file <input> <output>' to tokenize a file")
    print("  - Type 'quit' or 'exit' to quit\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting tokenizer.")
                break
            # Check if it's a file command
            if user_input.lower().startswith('file '):
                parts = user_input.split(maxsplit=2)
                if len(parts) == 3:
                    _, input_file, output_file = parts
                    print(f"\nTokenizing file: {input_file}")
                    tokenize_file(input_file, output_file, model, vectorizer)
                else:
                    print("Usage: file <input_file> <output_file>")
            else:
                # Tokenize the input text
                tokens = tokenize(user_input, model, vectorizer)
                print(f"Tokens: {tokens}")
                print(f"Token count: {len(tokens)}\n")

        except KeyboardInterrupt:
            print("\n\nExiting tokenizer.")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
