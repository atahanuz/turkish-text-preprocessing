#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format morphological parser output into a structured JSON format
"""

import requests
import json


def parse_morphological_output(text):
    """
    Parse the morphological analysis text into a structured format
    """
    lines = text.strip().split('\n')
    result = {}
    current_word = None

    for line in lines:
        # Skip empty lines and separator lines
        if not line or line.strip() == '-' * 50:
            continue

        # If line starts with tab, it's an analysis
        if line.startswith('\t'):
            analysis = line.strip()
            if current_word is not None:
                result[current_word].append(analysis)
        else:
            # It's a new word
            current_word = line.strip()
            result[current_word] = []

    return result


def print_formatted_output(parsed_data):
    """
    Print the parsed data in a nicely formatted way
    """
    print(json.dumps({"root": parsed_data}, indent=2, ensure_ascii=False))


def analyze_text(text):
    """
    Send text to morphological parser and return formatted results
    """
    url = 'http://localhost:4444/evaluateMD'
    response = requests.post(url, json={'textarea': text})

    if response.status_code == 200:
        result = response.json()
        parsed = parse_morphological_output(result['text'])
        return parsed
    else:
        raise Exception(f"API request failed with status {response.status_code}")


if __name__ == "__main__":
    # Example text
    text = "ahmet 10 kere türkiye'ye gitti"

    # Analyze and format
    parsed_output = analyze_text(text)
    print_formatted_output(parsed_output)

## Example output:

"""
  {
  "root": {
    "ahmet": [
      "Ahmet[Noun] [Prop] [A3sg] [Pnon] [Nom]"
    ],
    "10": [
      "10[Num]"
    ],
    "kere": [
      "kere[Noun]  [A3sg] [Pnon] [Nom]"
    ],
    "türkiye'ye": [
      "Türkiye[Noun] [Prop] [A3sg] [Pnon] '[Apos] YA[Dat]"
    ],
    "gitti": [
      "git[Verb] [Pos] DH[Past] [A3sg]"
    ]
  }
}  
    
    
    
"""