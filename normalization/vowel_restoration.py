import re

TURKISH_VOWELS = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']
lexicon_path = "../data/Turkish_Corpus_3M.txt"

import re

TURKISH_VOWELS = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']

def is_vowel(char):
    return char.lower() in TURKISH_VOWELS

def load_lexicon(lexicon_file):
    """Load Turkish lexicon into set"""
    lexicon = set()
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                lexicon.add(word)
    return lexicon

def extract_consonants(word):
    """Extract consonants preserving order"""
    return [c for c in word.lower() if not is_vowel(c)]

def get_segments(word):
    """Get segments between consonants"""
    segments = []
    current_consonants = []
    current_vowels = []
    
    for char in word.lower():
        if is_vowel(char):
            current_vowels.append(char)
        else:
            if current_consonants or current_vowels:
                segments.append((current_consonants.copy(), current_vowels.copy()))
            current_consonants = [char]
            current_vowels = []
    
    if current_consonants or current_vowels:
        segments.append((current_consonants, current_vowels))
    
    return segments

def match_pattern(word, lex_word):
    """Check if lex_word matches word pattern with constraints"""
    word_consonants = extract_consonants(word)
    lex_consonants = extract_consonants(lex_word)
    
    if word_consonants != lex_consonants:
        return False
    
    word_idx = 0
    lex_idx = 0
    consonant_count = 0
    
    while word_idx < len(word) and lex_idx < len(lex_word):
        word_char = word[word_idx]
        lex_char = lex_word[lex_idx]
        
        if not is_vowel(word_char) and not is_vowel(lex_char):
            if word_char != lex_char:
                return False
            consonant_count += 1
            
            # No vowels before first consonant
            if consonant_count == 1 and lex_idx > 0:
                return False
            
            word_idx += 1
            lex_idx += 1
        elif is_vowel(word_char) and is_vowel(lex_char):
            # Existing vowel must match
            if word_char != lex_char:
                return False
            word_idx += 1
            lex_idx += 1
        elif is_vowel(word_char):
            # Word has vowel, lex doesn't at this position - invalid
            return False
        else:
            # Lex has vowel, word doesn't - count vowels between consonants
            vowel_count = 0
            while lex_idx < len(lex_word) and is_vowel(lex_word[lex_idx]):
                vowel_count += 1
                lex_idx += 1
            
            # Only one vowel allowed between consonants
            if vowel_count != 1:
                return False
    
    # No vowels after last consonant
    if lex_idx < len(lex_word):
        if any(is_vowel(c) for c in lex_word[lex_idx:]):
            return False
    
    return word_idx == len(word) and lex_idx == len(lex_word)

def count_vowel_additions(word, lex_word):
    """Count how many vowels need to be added"""
    word_vowels = sum(1 for c in word if is_vowel(c))
    lex_vowels = sum(1 for c in lex_word if is_vowel(c))
    return lex_vowels - word_vowels

def generate_candidates(word, lexicon):
    """Generate vowel restoration candidates with minimum changes"""
    candidates = []
    
    word_lower = word.lower()
    
    for lex_word in lexicon:
        if match_pattern(word_lower, lex_word):
            additions = count_vowel_additions(word_lower, lex_word)
            candidates.append((lex_word, additions))
    
    if not candidates:
        return []
    
    # Return candidate with minimum vowel additions
    candidates.sort(key=lambda x: x[1])
    return [candidates[0][0]]

def restore_vowels(word, lexicon_file):
    """Restore missing vowels using lexicon lookup"""
    lexicon = load_lexicon(lexicon_file)
    
    word_lower = word.lower()
    
    if word_lower in lexicon:
        return word
    
    candidates = generate_candidates(word_lower, lexicon)
    
    if candidates:
        return candidates[0]
    
    return word


# Test case
if __name__ == "__main__":


    # Test cases from paper
    test_cases = [
        # Complete vowel removal
        ('kldn', 'okuldan'),
        ('svyrm', 'seviyorum'),
        ('Twttr', 'twitter'),
        ('bgn', 'bugün'),
        ('glncl', 'eğlenceli'),
        
        # Partial vowel restoration (some vowels kept)
        ('okldn', 'okuldan'),
        ('sevyrm', 'seviyorum'),
        ('gelyrm', 'geliyorum'),
        ('gitmiycm', 'gitmeyeceğim'),
        ('yapyrm', 'yapiyorum'),
        
        # Words with existing vowels that constrain output
        ('savyrm', 'seviyorum (if constraint works)'),
        ('okuldn', 'okuldan'),
        ('araba', 'araba (already complete)'),
        
        # Proper nouns
        ('ankr', 'ankara'),
        ('ankardn', 'ankaradan'),
        ('stnbl', 'istanbul'),
        
        # Common Turkish words
        ('ktp', 'kitap'),
        ('ktptn', 'kitaptan'),
        ('tlf', 'telefon'),
        ('tlfndn', 'telefondan'),
        ('blgsyr', 'bilgisayar'),
        
        # Verbs with negation
        ('gelmyr', 'gelmiyor'),
        ('yapmyr', 'yapmıyor'),
        ('svmyrm', 'sevmiyorum'),
        
        # Future tense
        ('glcm', 'geleceğim'),
        ('gelmycm', 'gelmeyeceğim'),
        ('ypcm', 'yapacağım'),
        
        # Words with ambiguous patterns
        ('kldn', 'koldan or kuldan or kıldan'),
        
        # Social media style
        ('ltfn', 'lütfen'),
        ('ck', 'çok'),
        
        # Common social media abbreviations
        ('nbr', 'ne haber (needs lexicon entry)'),
        ('slm', 'selam (needs lexicon entry)'),
    ]
    
    print("Vowel Restoration Test Cases:")
    print("-" * 50)
    for word, expected in test_cases:
        result = restore_vowels(word, lexicon_path)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {word:15} -> Output: {result:15} (Expected: {expected})")

    