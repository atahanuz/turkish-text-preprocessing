"""
Turkish Sentence Splitter using Naive Bayes
Features are designed based on punctuation and context around potential sentence boundaries.
"""

import re
from collections import defaultdict
import math
import pickle
import os


class TurkishSentenceSplitter:
    """
    A Naive Bayes-based sentence splitter for Turkish text.

    Features used for classification:
    1. Current character (punctuation mark)
    2. Next character (whitespace, letter, digit, punctuation)
    3. Previous character type
    4. Next word starts with capital letter
    5. Previous word ends with specific patterns
    6. Character after whitespace (if next is whitespace)
    """

    def __init__(self):
        # P(feature | class)
        self.feature_counts = {
            'boundary': defaultdict(int),
            'non_boundary': defaultdict(int)
        }

        # Class counts
        self.class_counts = {'boundary': 0, 'non_boundary': 0}

        # Total feature counts per class
        self.total_features = {'boundary': 0, 'non_boundary': 0}

        # Vocabulary size for Laplace smoothing
        self.vocab = set()

    def extract_features(self, text, pos):
        """
        Extract features from text at position pos.

        Args:
            text: The full text
            pos: Position of the potential boundary character

        Returns:
            List of feature strings
        """
        features = []

        # Feature 1: Current character
        curr_char = text[pos]
        features.append(f"curr_char={curr_char}")

        # Feature 2: Next character (if exists)
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            if next_char.isspace():
                features.append("next_char=SPACE")

                # Feature 6: Character after whitespace
                if pos + 2 < len(text):
                    char_after_space = text[pos + 2]
                    if char_after_space.isupper():
                        features.append("after_space=UPPER")
                    elif char_after_space.islower():
                        features.append("after_space=LOWER")
                    elif char_after_space.isdigit():
                        features.append("after_space=DIGIT")
                    elif char_after_space in '.!?':
                        features.append("after_space=PUNCT")
                    else:
                        features.append("after_space=OTHER")
            elif next_char.isupper():
                features.append("next_char=UPPER")
            elif next_char.islower():
                features.append("next_char=LOWER")
            elif next_char.isdigit():
                features.append("next_char=DIGIT")
            elif next_char in '"\'':
                features.append("next_char=QUOTE")
            elif next_char in '.!?':
                features.append("next_char=PUNCT")
            else:
                features.append("next_char=OTHER")
        else:
            features.append("next_char=END")

        # Feature 3: Previous character type
        if pos > 0:
            prev_char = text[pos - 1]
            if prev_char.isspace():
                features.append("prev_char=SPACE")
            elif prev_char.isupper():
                features.append("prev_char=UPPER")
            elif prev_char.islower():
                features.append("prev_char=LOWER")
            elif prev_char.isdigit():
                features.append("prev_char=DIGIT")
            elif prev_char in '"\'':
                features.append("prev_char=QUOTE")
            else:
                features.append("prev_char=OTHER")
        else:
            features.append("prev_char=START")

        # Feature 4: Check if next word starts with capital (looking ahead)
        # Extract next word after this position
        remaining_text = text[pos + 1:].lstrip()
        if remaining_text and remaining_text[0].isupper():
            features.append("next_word_capital=True")
        else:
            features.append("next_word_capital=False")

        # Feature 5: Previous word pattern (look for abbreviations, numbers, etc.)
        # Get text before current position
        before_text = text[:pos]
        # Find the last word
        words_before = before_text.split()
        if words_before:
            last_word = words_before[-1]
            # Check if it's a number
            if last_word.replace(',', '').replace('.', '').isdigit():
                features.append("prev_word=NUMBER")
            # Check if it's likely an abbreviation (single letter or short uppercase)
            elif len(last_word) <= 3 and last_word.isupper():
                features.append("prev_word=ABBREV")
            # Check if it's all lowercase
            elif last_word.islower():
                features.append("prev_word=LOWER")
            # Check if it starts with capital
            elif last_word[0].isupper():
                features.append("prev_word=CAPITAL")
            else:
                features.append("prev_word=OTHER")

        return features

    def train(self, sentences):
        """
        Train the Naive Bayes classifier on a list of sentences.

        Args:
            sentences: List of sentences (already split correctly)
        """
        # Process each sentence to find boundaries and non-boundaries
        for sentence in sentences:
            # Within each sentence, all punctuation marks are non-boundaries
            # except the last one (if it's a punctuation mark)
            for i, char in enumerate(sentence):
                if char in '.!?':
                    features = self.extract_features(sentence, i)

                    # Check if this is the last punctuation in the sentence
                    is_last = True
                    for j in range(i + 1, len(sentence)):
                        if sentence[j] in '.!?':
                            is_last = False
                            break

                    # The last punctuation mark in a sentence is a boundary
                    # All others are non-boundaries
                    class_label = 'boundary' if is_last else 'non_boundary'
                    self.class_counts[class_label] += 1

                    for feature in features:
                        self.feature_counts[class_label][feature] += 1
                        self.total_features[class_label] += 1
                        self.vocab.add(feature)

        # Also train on concatenated sentences to learn boundary patterns
        # Train with BOTH space and no-space scenarios
        for i in range(len(sentences) - 1):
            # Scenario 1: Create text without space between sentences
            combined_no_space = sentences[i] + sentences[i + 1]
            boundary_pos_no_space = len(sentences[i]) - 1

            # Check if it ends with punctuation
            if boundary_pos_no_space >= 0 and combined_no_space[boundary_pos_no_space] in '.!?':
                features = self.extract_features(combined_no_space, boundary_pos_no_space)

                # This is a boundary (even without space after)
                class_label = 'boundary'
                self.class_counts[class_label] += 1

                for feature in features:
                    self.feature_counts[class_label][feature] += 1
                    self.total_features[class_label] += 1
                    self.vocab.add(feature)

            # Scenario 2: Create text WITH space between sentences
            combined_with_space = sentences[i] + ' ' + sentences[i + 1]
            boundary_pos_with_space = len(sentences[i]) - 1

            # Check if it ends with punctuation
            if boundary_pos_with_space >= 0 and combined_with_space[boundary_pos_with_space] in '.!?':
                features = self.extract_features(combined_with_space, boundary_pos_with_space)

                # This is a boundary (with space after)
                class_label = 'boundary'
                self.class_counts[class_label] += 1

                for feature in features:
                    self.feature_counts[class_label][feature] += 1
                    self.total_features[class_label] += 1
                    self.vocab.add(feature)

    def predict(self, text, pos):
        """
        Predict if position pos is a sentence boundary.

        Args:
            text: The full text
            pos: Position of the punctuation mark

        Returns:
            'boundary' or 'non_boundary'
        """
        features = self.extract_features(text, pos)

        # Calculate log probabilities to avoid underflow
        scores = {}

        for class_label in ['boundary', 'non_boundary']:
            # P(class)
            total_examples = sum(self.class_counts.values())
            if total_examples == 0:
                class_prob = 0.5
            else:
                class_prob = self.class_counts[class_label] / total_examples

            score = math.log(class_prob) if class_prob > 0 else -1000

            # P(features | class) with Laplace smoothing
            vocab_size = len(self.vocab)

            for feature in features:
                # Laplace smoothing: (count + 1) / (total + vocab_size)
                feature_count = self.feature_counts[class_label][feature]
                total = self.total_features[class_label]

                feature_prob = (feature_count + 1) / (total + vocab_size)
                score += math.log(feature_prob)

            scores[class_label] = score

        # Return class with highest score
        return 'boundary' if scores['boundary'] > scores['non_boundary'] else 'non_boundary'

    def split_sentences(self, text):
        """
        Split text into sentences using the trained model.

        Args:
            text: The text to split

        Returns:
            List of sentences
        """
        sentences = []
        current_sentence = []

        for i, char in enumerate(text):
            current_sentence.append(char)

            # Check if this is a potential boundary
            if char in '.!?':
                prediction = self.predict(text, i)

                if prediction == 'boundary':
                    # This is a sentence boundary
                    sentence = ''.join(current_sentence).strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = []

        # Add any remaining text as the last sentence
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def save_model(self, filepath):
        """
        Save the trained model to a file using pickle.

        Args:
            filepath: Path where the model should be saved
        """
        model_data = {
            'feature_counts': self.feature_counts,
            'class_counts': self.class_counts,
            'total_features': self.total_features,
            'vocab': self.vocab
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.feature_counts = model_data['feature_counts']
        self.class_counts = model_data['class_counts']
        self.total_features = model_data['total_features']
        self.vocab = model_data['vocab']


def load_training_data(filename):
    """
    Load training data from file.
    Each line is a sentence.

    Args:
        filename: Path to the training data file

    Returns:
        List of sentences
    """
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def main():
    # Example usage
    print("Training Turkish Sentence Splitter...")

    # Load training data
    train_sentences = load_training_data('../data/tr_boun-ud-train_text.txt')
    print(f"Loaded {len(train_sentences)} training sentences")

    # Train the model
    splitter = TurkishSentenceSplitter()
    splitter.train(train_sentences)
    print("Training complete!")

    # Save the model
    model_path = '../data/sentence_splitter_nb_model.pkl'
    splitter.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Test on example
    test_text = "Ali okula gitti.Ali eve gitti."
    print(f"\nTest text: {test_text}")

    result = splitter.split_sentences(test_text)
    print("\nResult:")
    for i, sentence in enumerate(result, 1):
        print(f"{i}- {sentence}")

    # Test on more examples
    print("\n" + "="*50)
    test_text2 = "1936 yılındayız.Adeta kendimden geçmiş bir haldeyim.Rüzgâr yine güçlü esiyordu."
    print(f"Test text: {test_text2}")

    result2 = splitter.split_sentences(test_text2)
    print("\nResult:")
    for i, sentence in enumerate(result2, 1):
        print(f"{i}- {sentence}")


if __name__ == "__main__":
    main()
