#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turkish Diacritic Correction Module
Restores Turkish characters (ş, ç, ğ, ü, ö, ı) from ASCII equivalents
"""

import re
from typing import Dict, List, Set, Tuple

lexicon_path = "../data/Turkish_Corpus_3M.txt"

# Turkish character mappings
DIACRITIC_MAP = {
    's': ['s', 'ş'],
    'c': ['c', 'ç'],
    'g': ['g', 'ğ'],
    'u': ['u', 'ü'],
    'o': ['o', 'ö'],
    'i': ['i', 'ı', 'î'],
    'S': ['S', 'Ş'],
    'C': ['C', 'Ç'],
    'G': ['G', 'Ğ'],
    'U': ['U', 'Ü'],
    'O': ['O', 'Ö'],
    'I': ['I', 'İ']
}

# Common word corrections (high-frequency words)
WORD_CORRECTIONS = {
    'ben': 'ben', 'sen': 'sen', 'o': 'o', 'biz': 'biz', 'siz': 'siz', 'onlar': 'onlar',
    'bu': 'bu', 'su': 'şu', 'o': 'o', 'bunlar': 'bunlar', 'sunlar': 'şunlar', 'onlar': 'onlar',
    'bir': 'bir', 'iki': 'iki', 'uc': 'üç', 'dort': 'dört', 'bes': 'beş', 'alti': 'altı',
    'yedi': 'yedi', 'sekiz': 'sekiz', 'dokuz': 'dokuz', 'on': 'on',
    'cok': 'çok', 'cox': 'çok', 'az': 'az', 'hic': 'hiç',
    'var': 'var', 'yok': 'yok', 've': 've', 'veya': 'veya', 'ile': 'ile',
    'icin': 'için', 'gibi': 'gibi', 'kadar': 'kadar', 'gore': 'göre',
    'sonra': 'sonra', 'once': 'önce', 'simdi': 'şimdi', 'suanda': 'şuanda',
    'ne': 'ne', 'nasil': 'nasıl', 'neden': 'neden', 'nerede': 'nerede', 'kim': 'kim',
    'hangi': 'hangi', 'kac': 'kaç', 'ne zaman': 'ne zaman',
    'gundem': 'gündem', 'gun': 'gün', 'gece': 'gece', 'sabah': 'sabah', 'aksam': 'akşam',
    'iyi': 'iyi', 'guzel': 'güzel', 'kotu': 'kötü', 'fena': 'fena',
    'buyuk': 'büyük', 'kucuk': 'küçük', 'uzun': 'uzun', 'kisa': 'kısa',
    'gitmek': 'gitmek', 'gelmek': 'gelmek', 'yapmak': 'yapmak', 'olmak': 'olmak',
    'almak': 'almak', 'vermek': 'vermek', 'gormek': 'görmek', 'bilmek': 'bilmek',
    'istiyorum': 'istiyorum', 'geliyorum': 'geliyorum', 'gidiyorum': 'gidiyorum',
    'yapiyorum': 'yapıyorum', 'biliyorum': 'biliyorum',
    'istanbul': 'İstanbul', 'ankara': 'Ankara', 'izmir': 'İzmir', 'bursa': 'Bursa',
    'turkiye': 'Türkiye', 'turk': 'Türk',
    'lutfen': 'lütfen', 'ltfn': 'lütfen', 'tsk': 'teşekkür', 'tskler': 'teşekkürler',
    'msj': 'mesaj', 'tmm': 'tamam', 'ok': 'tamam',
    'nbr': 'ne haber', 'naber': 'ne haber', 'nslsn': 'nasılsın',
    'sey': 'şey', 'seyler': 'şeyler', 'seyi': 'şeyi',
    'su': 'şu', 'sunu': 'şunu', 'sunlar': 'şunlar',
    'sehir': 'şehir', 'seker': 'şeker', 'sarki': 'şarkı',
    'calisma': 'çalışma', 'calisiyorum': 'çalışıyorum', 'calismak': 'çalışmak',
    'cocuk': 'çocuk', 'cicek': 'çiçek',
    'aglamak': 'ağlamak', 'diger': 'diğer', 'dogru': 'doğru',
    'ogrenci': 'öğrenci', 'ogrenmek': 'öğrenmek', 'ogretmen': 'öğretmen',
    'ulke': 'ülke', 'universite': 'üniversite', 'urun': 'ürün',
    'ornek': 'örnek', 'onemli': 'önemli', 'ozel': 'özel',
    'isik': 'ışık', 'kisin': 'kışın', 'kis': 'kış',
}

# Suffix patterns (common Turkish suffixes)
SUFFIX_PATTERNS = {
    'yor': 'yor', 'yor': 'yor',  # present continuous
    'mis': 'miş', 'mus': 'muş', 'mıs': 'mış', 'mış': 'mış',  # past narrative
    'di': 'dı', 'dı': 'dı', 'du': 'du', 'dü': 'dü',  # past definite
    'acak': 'acak', 'ecek': 'ecek',  # future
    'mak': 'mak', 'mek': 'mek',  # infinitive
    'lik': 'lık', 'luk': 'luk', 'lık': 'lık', 'lük': 'lük',  # noun suffix
    'ci': 'cı', 'cı': 'cı', 'cu': 'cu', 'cü': 'cü',  # agent suffix
    'siz': 'siz', 'sız': 'sız', 'suz': 'suz', 'süz': 'süz',  # without
    'li': 'li', 'lı': 'lı', 'lu': 'lu', 'lü': 'lü',  # with
}


def load_vocabulary(lexicon_path: str = None) -> Set[str]:
    """Load Turkish vocabulary from file or use built-in corrections"""
    if lexicon_path:
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            pass
    return set(WORD_CORRECTIONS.values())


def correct_word_with_dict(word: str, vocabulary: Set[str]) -> str:
    """Correct single word using vocabulary lookup"""
    word_lower = word.lower()
    
    # Direct lookup in corrections
    if word_lower in WORD_CORRECTIONS:
        corrected = WORD_CORRECTIONS[word_lower]
        return match_case(corrected, word)
    
    # Check if already in vocabulary
    if word_lower in vocabulary:
        return word
    
    return word


def generate_candidates(word: str) -> List[str]:
    """Generate possible diacritic variants using iterative approach"""
    if not word:
        return [word]
    
    # Find positions with ambiguous characters
    ambiguous_positions = []
    for i, char in enumerate(word):
        char_lower = char.lower()
        if char_lower in DIACRITIC_MAP and len(DIACRITIC_MAP[char_lower]) > 1:
            ambiguous_positions.append((i, char, DIACRITIC_MAP[char_lower]))
    
    if not ambiguous_positions:
        return [word]
    
    # Generate candidates iteratively
    candidates = [list(word)]
    
    for pos, orig_char, replacements in ambiguous_positions:
        new_candidates = []
        for candidate in candidates:
            for replacement in replacements:
                new_candidate = candidate.copy()
                # Match case
                if orig_char.isupper():
                    replacement = replacement.upper()
                new_candidate[pos] = replacement
                new_candidates.append(new_candidate)
        candidates = new_candidates
        
        # Limit explosion
        if len(candidates) > 200:
            candidates = candidates[:200]
    
    return [''.join(c) for c in candidates]


def score_candidate(word: str, vocabulary: Set[str]) -> float:
    """Score a candidate word based on vocabulary match and patterns"""
    score = 0.0
    word_lower = word.lower()
    
    # Vocabulary match
    if word_lower in vocabulary:
        score += 10.0
    
    # Common patterns
    if any(word_lower.endswith(suffix) for suffix in ['yor', 'miş', 'muş', 'mış']):
        score += 2.0
    
    # Turkish character frequency
    turkish_chars = sum(1 for c in word if c in 'şçğüöıŞÇĞÜÖİ')
    score += turkish_chars * 0.5
    
    return score


def correct_word(word: str, vocabulary: Set[str]) -> str:
    """Correct diacritics in a single word"""
    # Skip if already has Turkish characters
    if any(c in word for c in 'şçğüöıŞÇĞÜÖİ'):
        return word
    
    # Skip very short words
    if len(word) <= 1:
        return word
    
    word_lower = word.lower()
    
    # Try dictionary correction first
    if word_lower in WORD_CORRECTIONS:
        corrected = WORD_CORRECTIONS[word_lower]
        return match_case(corrected, word)
    
    # Check if ASCII version is already in vocabulary (don't change)
    if word_lower in vocabulary:
        return word
    
    # Generate and score candidates
    candidates = generate_candidates(word)
    if not candidates:
        return word
    
    # Find candidates in vocabulary
    vocab_candidates = []
    for candidate in candidates:
        if candidate.lower() in vocabulary:
            vocab_candidates.append(candidate)
    
    # Only correct if we found exact vocabulary match
    if vocab_candidates:
        # Return the candidate with most Turkish chars (most likely correct)
        vocab_candidates.sort(
            key=lambda x: sum(1 for c in x if c in 'şçğüöıŞÇĞÜÖİ'),
            reverse=True
        )
        return vocab_candidates[0]
    
    # No confident correction found
    return word


def match_case(corrected: str, original: str) -> str:
    """Match the case pattern of original word"""
    if original.isupper():
        return corrected.upper()
    elif original[0].isupper() and len(original) > 1:
        return corrected[0].upper() + corrected[1:].lower()
    return corrected.lower()


def correct_text(text: str, vocabulary: Set[str] = None) -> str:
    """Correct diacritics in entire text"""
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    # Tokenize preserving punctuation
    pattern = r'\b\w+\b'
    
    def replace_word(match):
        word = match.group(0)
        return correct_word(word, vocabulary)
    
    return re.sub(pattern, replace_word, text)


def correct_batch(texts: List[str], vocabulary: Set[str] = None) -> List[str]:
    """Correct multiple texts efficiently"""
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    return [correct_text(text, vocabulary) for text in texts]


# Pipeline function
def diacritic_correction_pipeline(text: str, lexicon_path: str = None) -> str:
    """Main pipeline function for diacritic correction"""
    vocabulary = load_vocabulary(lexicon_path)
    return correct_text(text, vocabulary)


if __name__ == '__main__':
    # Test examples
    test_cases = [
        "cok guzel bir gun",
        "istanbul cok buyuk bir sehir",
        "lutfen bana yardim edin",
        "ogrenci universite gidiyor",
        "turkiye cox guzel",
        "simdi ne yapacagim bilmiyorum",
        "geliyorum oraya hemen",
    ]
    
    print("Turkish Diacritic Correction Test")
    print("=" * 60)
    
    for test in test_cases:
        corrected = diacritic_correction_pipeline(test, lexicon_path)
        print(f"Original:  {test}")
        print(f"Corrected: {corrected}")
        print()