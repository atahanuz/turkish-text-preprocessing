#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abbreviation Extension Module
Expands Turkish abbreviations to their full forms based on data/abbreviations.txt
"""

import os
import re
from typing import Dict, Optional


def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on project root."""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(current_dir)
    # Join with the relative path
    return os.path.join(project_root, relative_path)


def load_abbreviations(abbreviations_path='data/abbreviations.txt') -> Dict[str, str]:
    """
    Load abbreviations from file into a dictionary.

    Args:
        abbreviations_path: Path to abbreviations file (relative to project root)

    Returns:
        Dictionary mapping abbreviations to their full forms
    """
    # Convert to absolute path if relative
    if not os.path.isabs(abbreviations_path):
        abbreviations_path = get_absolute_path(abbreviations_path)

    abbreviations = {}

    try:
        with open(abbreviations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                # Split on first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    abbrev = parts[0].strip()
                    full_form = parts[1].strip()
                    if abbrev and full_form:
                        # Store both lowercase and original case versions
                        abbreviations[abbrev.lower()] = full_form

    except FileNotFoundError:
        print(f"Warning: Abbreviations file not found at {abbreviations_path}")
        return {}
    except Exception as e:
        print(f"Error loading abbreviations: {e}")
        return {}

    return abbreviations


# Global abbreviations dictionary (loaded once)
_ABBREVIATIONS_CACHE: Optional[Dict[str, str]] = None


def get_abbreviations(abbreviations_path='data/abbreviations.txt') -> Dict[str, str]:
    """
    Get abbreviations dictionary, using cache if available.

    Args:
        abbreviations_path: Path to abbreviations file

    Returns:
        Dictionary mapping abbreviations to their full forms
    """
    global _ABBREVIATIONS_CACHE

    if _ABBREVIATIONS_CACHE is None:
        _ABBREVIATIONS_CACHE = load_abbreviations(abbreviations_path)

    return _ABBREVIATIONS_CACHE


def expand_abbreviation(word: str, abbreviations_path='data/abbreviations.txt') -> str:
    """
    Expand a single word if it's an abbreviation.

    Args:
        word: Input word (potentially an abbreviation)
        abbreviations_path: Path to abbreviations file

    Returns:
        Expanded form if word is an abbreviation, otherwise original word
    """
    if not word or not word.strip():
        return word

    # Get abbreviations dictionary
    abbreviations = get_abbreviations(abbreviations_path)

    # Check if word (lowercase) is an abbreviation
    word_lower = word.lower()
    if word_lower in abbreviations:
        return abbreviations[word_lower]

    return word


def expand_abbreviations_in_text(text: str, abbreviations_path='data/abbreviations.txt') -> str:
    """
    Expand all abbreviations in a text string.

    Args:
        text: Input text containing potential abbreviations
        abbreviations_path: Path to abbreviations file

    Returns:
        Text with abbreviations expanded
    """
    if not text:
        return text

    # Get abbreviations dictionary
    abbreviations = get_abbreviations(abbreviations_path)

    # Split text into words while preserving punctuation and whitespace
    # This regex splits on word boundaries but preserves the separators
    tokens = re.split(r'(\W+)', text)

    result = []
    for token in tokens:
        if token and token.strip() and token.isalpha():
            # It's a word, check if it's an abbreviation
            expanded = expand_abbreviation(token, abbreviations_path)
            result.append(expanded)
        else:
            # It's punctuation or whitespace, keep as is
            result.append(token)

    return ''.join(result)


# Example usage
if __name__ == '__main__':
    # Test abbreviation expansion
    print("Abbreviation Extension Test")
    print("=" * 60)

    test_words = [
        "abd",
        "aa",
        "ab",
        "tl",
        "tic",
        "hello",  # Not an abbreviation
        "tobb",
        "nato"
    ]

    print("\nSingle word expansion:")
    print("-" * 60)
    for word in test_words:
        expanded = expand_abbreviation(word)
        if expanded != word:
            print(f"{word} → {expanded}")
        else:
            print(f"{word} → (no expansion)")

    print("\n" + "=" * 60)
    print("\nText expansion:")
    print("-" * 60)

    test_texts = [
        "abd ve ab arasında",
        "tobb etü bir üniversitedir",
        "nato üyesi olan ülke",
        "bugün hava çok güzel"
    ]

    for text in test_texts:
        expanded = expand_abbreviations_in_text(text)
        print(f"Input:  {text}")
        print(f"Output: {expanded}")
        print()
