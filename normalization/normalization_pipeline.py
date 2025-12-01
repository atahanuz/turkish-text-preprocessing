#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turkish Text Normalization Pipeline
Processes tokens through a 6-step pipeline:
1. Abbreviation Extension
2. Letter Case Transformation
3. Diacritic Restoration
4. Vowel Restoration
5. Spelling Correction
6. Accent Normalization
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalization.abbreviation_extension import expand_abbreviation
from normalization.letter_case_transformation import letter_case_transformation, turkish_capitalize
from normalization.diacritic_restoration import correct_word as restore_diacritics, load_vocabulary as load_diacritic_vocab
from normalization.vowel_restoration import restore_vowels
from normalization.spelling_correction import spelling_correction
from normalization.accent_normalization import normalize_accent
from normalization.morho_parser import analyze_text


def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on project root."""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(current_dir)
    # Join with the relative path
    return os.path.join(project_root, relative_path)


class NormalizationPipeline:
    """
    Turkish text normalization pipeline that processes each token independently
    through 6 sequential steps.
    """

    def __init__(self, lexicon_path='data/Turkish_Corpus_3M.txt', use_morpho_parser=False):
        """
        Initialize the normalization pipeline.

        Args:
            lexicon_path: Path to Turkish lexicon file (relative to project root)
            use_morpho_parser: Whether to use morphological parser for proper noun detection
        """
        # Convert to absolute path if relative
        if not os.path.isabs(lexicon_path):
            self.lexicon_path = get_absolute_path(lexicon_path)
        else:
            self.lexicon_path = lexicon_path

        # Pre-load vocabulary for diacritic restoration
        self.diacritic_vocab = load_diacritic_vocab(self.lexicon_path)

        # Flag for morphological parser usage
        self.use_morpho_parser = use_morpho_parser

    def normalize_token(self, token):
        """
        Process a single token through the 6-step normalization pipeline.

        Pipeline order:
        0. Abbreviation Extension: Expand abbreviations to their full forms
        1. Letter Case Transformation: Normalize Turkish letter casing
        2. Diacritic Restoration: Restore Turkish characters (ş, ç, ğ, ü, ö, ı)
        3. Vowel Restoration: Restore missing vowels
        4. Spelling Correction: Fix spelling errors
        5. Accent Normalization: Normalize colloquial forms to standard Turkish

        Args:
            token: Input token to normalize

        Returns:
            Normalized token
        """
        # Skip empty tokens
        if not token:
            return token

        # Skip tokens with no alphabetic characters (pure punctuation)
        if not any(c.isalpha() for c in token):
            return token

        result = token

        # Step 0: Abbreviation Extension
        result = expand_abbreviation(result)

        # Step 1: Letter Case Transformation
        result = letter_case_transformation(result)

        # Step 2: Diacritic Restoration
        result = restore_diacritics(result, self.diacritic_vocab)

        # Step 3: Vowel Restoration
        result = restore_vowels(result, self.lexicon_path)

        # Step 4: Spelling Correction
        result = spelling_correction(result, self.lexicon_path, max_edit_distance=2)

        # Step 5: Accent Normalization
        result = normalize_accent(result)

        return result

    def _is_proper_noun(self, word, morpho_data):
        """
        Check if a word is a proper noun based on morphological analysis.

        Args:
            word: The word to check (original form before normalization)
            morpho_data: Dictionary of morphological analysis results

        Returns:
            True if the word is a proper noun, False otherwise
        """
        if word not in morpho_data:
            return False

        analyses = morpho_data[word]
        # Check if any analysis contains [Prop] tag
        for analysis in analyses:
            if '[Prop]' in analysis:
                return True
        return False

    def normalize_sequence(self, tokens):
        """
        Process a sequence of tokens through the normalization pipeline.
        Each token is processed independently.
        If morpho_parser is enabled, proper nouns are detected and capitalized.

        Args:
            tokens: List of tokens or space-separated string

        Returns:
            List of normalized tokens
        """
        # Handle both list and string input
        if isinstance(tokens, str):
            tokens = tokens.split()

        # Get morphological analysis if enabled
        morpho_data = {}
        if self.use_morpho_parser and tokens:
            try:
                text = ' '.join(tokens)
                morpho_data = analyze_text(text)
            except Exception as e:
                print(f"Warning: Morphological parser failed: {e}")
                print("Continuing without proper noun detection...")

        normalized_tokens = []
        for token in tokens:
            normalized = self.normalize_token(token)

            # Check if original token is a proper noun and capitalize it
            if self.use_morpho_parser and self._is_proper_noun(token, morpho_data):
                normalized = turkish_capitalize(normalized)

            normalized_tokens.append(normalized)

        return normalized_tokens

    def normalize_text(self, text):
        """
        Process entire text through normalization pipeline.

        Args:
            text: Input text string

        Returns:
            Normalized text string
        """
        tokens = text.split()
        normalized_tokens = self.normalize_sequence(tokens)
        return ' '.join(normalized_tokens)


def normalize_pipeline(tokens, lexicon_path='data/Turkish_Corpus_3M.txt'):
    """
    Convenience function to normalize a sequence of tokens.

    Args:
        tokens: List of tokens or space-separated string
        lexicon_path: Path to Turkish lexicon file

    Returns:
        List of normalized tokens
    """
    pipeline = NormalizationPipeline(lexicon_path)
    return pipeline.normalize_sequence(tokens)


if __name__ == '__main__':
    # Test the pipeline with the example from requirements
    print("Turkish Text Normalization Pipeline Test")
    print("=" * 60)

    # Example: "Bugun hava cok gzl"
    # Expected: Each token should be processed independently
    test_sequence = "Bugun hava cok gzl"
    tokens = test_sequence.split()

    print(f"\nInput sequence: {test_sequence}")
    print(f"Tokens: {tokens}")
    print("\nProcessing each token through 6-step pipeline:")
    print("-" * 60)

    pipeline = NormalizationPipeline(lexicon_path='data/Turkish_Corpus_3M.txt')

    for token in tokens:
        print(f"\nToken: {token}")

        # Step by step processing
        step0 = expand_abbreviation(token)
        print(f"  0. Abbreviation Extension: {step0}")

        step1 = letter_case_transformation(step0)
        print(f"  1. Letter Case Transformation: {step1}")

        step2 = restore_diacritics(step1, pipeline.diacritic_vocab)
        print(f"  2. Diacritic Restoration: {step2}")

        step3 = restore_vowels(step2, pipeline.lexicon_path)
        print(f"  3. Vowel Restoration: {step3}")

        step4 = spelling_correction(step3, pipeline.lexicon_path, max_edit_distance=2)
        print(f"  4. Spelling Correction: {step4}")

        step5 = normalize_accent(step4)
        print(f"  5. Accent Normalization: {step5}")

        print(f"  Final: {token} → {step5}")

    print("\n" + "=" * 60)
    print("\nFull pipeline result:")
    normalized_tokens = pipeline.normalize_sequence(tokens)
    print(f"Input:  {test_sequence}")
    print(f"Output: {' '.join(normalized_tokens)}")

    print("\n" + "=" * 60)
    print("\nAdditional test cases:")

    test_cases = [
        "geliyom sana",
        "yarin gidecegim",
        "cok guzel bir gun",
        "nslsn bugun",
    ]

    for test in test_cases:
        normalized = pipeline.normalize_text(test)
        print(f"  {test:30} → {normalized}")
