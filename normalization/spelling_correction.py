import re
from collections import defaultdict, Counter
import heapq
import tempfile
import os

def spelling_correction(word, lexicon_path, error_model_path=None, max_edit_distance=2, top_k=5):
    """SC#4: Error model + Language model based spelling correction (diacritic-aware)"""
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        lexicon = [line.strip() for line in f if line.strip()]
    
    if word in lexicon:
        return word
    word_freq = Counter(lexicon)
    total = sum(word_freq.values())
    word_prob = {w: freq/total for w, freq in word_freq.items()}
    lexicon_set = set(lexicon)
    
    error_costs = {}
    if error_model_path:
        with open(error_model_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    pattern, cost = line.strip().split(':', 1)
                    error_costs[pattern] = float(cost)
    
    # Diacritic mappings (ASCII -> Turkish)
    diacritic_map = {
        'c': ['c', 'ç'],
        's': ['s', 'ş'],
        'g': ['g', 'ğ'],
        'o': ['o', 'ö'],
        'u': ['u', 'ü'],
        'i': ['i', 'ı', 'İ'],
        'I': ['I', 'ı', 'İ']
    }
    
    default_insert_cost = 1.0
    default_delete_cost = 1.0
    default_substitute_cost = 1.0
    default_transpose_cost = 1.0
    diacritic_substitute_cost = 0.3  # Lower cost for diacritic changes
    
    def edits_with_cost(w, max_dist=2):
        candidates = []
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        
        # Deletes
        deletes = [(L + R[1:], default_delete_cost) for L, R in splits if R]
        
        # Inserts
        alphabet = 'abcçdefgğhıijklmnoöprsştuüvyz'
        inserts = [(L + c + R, default_insert_cost) for L, R in splits for c in alphabet]
        
        # Substitutes (including diacritic-aware)
        substitutes = []
        for L, R in splits:
            if R:
                # Regular substitutes
                for c in alphabet:
                    if c != R[0]:
                        # Check if it's a diacritic variant
                        is_diacritic_variant = False
                        base_char = R[0].lower()
                        if base_char in diacritic_map and c in diacritic_map[base_char]:
                            cost = diacritic_substitute_cost
                            is_diacritic_variant = True
                        else:
                            cost = default_substitute_cost
                        
                        substitutes.append((L + c + R[1:], cost))
        
        # Transposes
        transposes = [(L + R[1] + R[0] + R[2:], default_transpose_cost) for L, R in splits if len(R) > 1]
        
        edits1 = deletes + inserts + substitutes + transposes
        
        if max_dist == 1:
            return edits1
        
        # Edits at distance 2
        edits2 = []
        seen_e2 = set()
        for e1, cost1 in edits1:
            for e2, cost2 in edits_with_cost(e1, max_dist=1):
                if e2 not in seen_e2:
                    seen_e2.add(e2)
                    edits2.append((e2, cost1 + cost2))
        
        return edits1 + edits2
    
    # Generate candidates
    candidates_with_scores = []
    seen = set()
    
    for candidate, error_cost in edits_with_cost(word, max_dist=max_edit_distance):
        if candidate in lexicon_set and candidate not in seen:
            seen.add(candidate)
            lm_score = word_prob.get(candidate, 1e-10)
            score = lm_score / (1 + error_cost)
            candidates_with_scores.append((score, candidate, error_cost))
    
    top_candidates = heapq.nlargest(top_k, candidates_with_scores, key=lambda x: x[0])
    
    if top_candidates:
        return top_candidates[0][1]
    
    return word

def learn_error_model(parallel_corpus_path, output_path):
    error_patterns = defaultdict(int)
    
    with open(parallel_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                wrong, correct = line.strip().split('\t')
                
                m, n = len(wrong), len(correct)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if wrong[i-1] == correct[j-1]:
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                
                i, j = m, n
                while i > 0 or j > 0:
                    if i > 0 and j > 0 and wrong[i-1] == correct[j-1]:
                        i -= 1
                        j -= 1
                    elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                        error_patterns[f'sub:{wrong[i-1]}:{correct[j-1]}'] += 1
                        i -= 1
                        j -= 1
                    elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                        error_patterns[f'del:{wrong[i-1]}'] += 1
                        i -= 1
                    elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                        error_patterns[f'ins:{correct[j-1]}'] += 1
                        j -= 1
    
    total = sum(error_patterns.values())
    with open(output_path, 'w', encoding='utf-8') as f:
        for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
            cost = 1.0 - (count / total)
            f.write(f'{pattern}:{cost:.4f}\n')

def test_spelling_correction():
    
    lexicon_path = "../data/Turkish_Corpus_3M.txt"
    try:
        test_cases = [
    # Diacritic errors - ç/c
    ("cok", "çok"),
    ("cocuk", "çocuk"),
    ("agac", "ağaç"),
    ("cicek", "çiçek"),
    ("kac", "kaç"),
    
    # Diacritic errors - ş/s
    ("seker", "şeker"),
    ("masa", "masa"),
    ("basla", "başla"),
    ("isik", "ışık"),
    ("sarki", "şarkı"),
    
    # Diacritic errors - ğ/g
    ("agir", "ağır"),
    ("yagmur", "yağmur"),
    ("dogru", "doğru"),
    ("bagir", "bağır"),
    ("soguk", "soğuk"),
    
    # Diacritic errors - ö/o
    ("guzel", "güzel"),
    ("goz", "göz"),
    ("sonra", "sonra"),
    ("gosterge", "gösterge"),
    ("koy", "köy"),
    
    # Diacritic errors - ü/u
    ("uzgun", "üzgün"),
    ("urun", "ürün"),
    ("yuksel", "yüksel"),
    ("gul", "gül"),
    ("mutlu", "mutlu"),
    
    # Diacritic errors - ı/i
    ("sinif", "sınıf"),
    ("isim", "isim"),
    ("kitap", "kitap"),
    ("kirmizi", "kırmızı"),
    ("yikama", "yıkama"),
    
    # Multiple diacritics
    ("tesekkur", "teşekkür"),
    ("ogretmen", "öğretmen"),
    ("ogrenci", "öğrenci"),
    ("gonul", "gönül"),
    ("guneş", "güneş"),
    
    # Deletion errors
    ("okl", "okul"),
    ("ktap", "kitap"),
    ("dfter", "defter"),
    ("klm", "kalem"),
    ("mrhaba", "merhaba"),
    
    # Insertion errors
    ("okull", "okul"),
    ("gitt", "git"),
    ("kaleem", "kalem"),
    ("evvde", "evde"),
    ("yazzma", "yazma"),
    
    # Transposition errors
    ("gti", "git"),
    ("evd", "evde"),
    ("klitap", "kitap"),
    ("kaelm", "kalem"),
    ("yerş", "yer"),
    
    # Substitution errors
    ("peder", "defter"),
    ("kelem", "kalem"),
    ("yaz", "yaz"),
    ("akul", "okul"),
    ("gat", "git"),
    
    # Vowel omissions
    ("ktp", "kitap"),
    ("klm", "kalem"),
    ("dftr", "defter"),
    ("mrh", "merhaba"),
    ("nslsn", "nasılsın"),
    
    # Multiple errors
    ("okldan", "okuldan"),
    ("merhba", "merhaba"),
    ("nasilsin", "nasılsın"),
    ("bugun", "bugün"),
    ("eglencli", "eğlenceli"),
    
    # Common typos
    ("sevyorum", "seviyorum"),
    ("gidyorum", "gidiyorum"),
    ("geliyrom", "geliyorum"),
    ("yapiorm", "yapıyorum"),
    ("biliyrom", "biliyorum"),
    
    # Social media style
    ("iyii", "iyi"),
    ("kotuu", "kötü"),
    ("twtter", "twitter"),
    ("aksam", "akşam"),
    ("sabah", "sabah"),
    
    # Complex combinations
    ("tesekkurr", "teşekkür"),
    ("okulda", "okulda"),
    ("sinifta", "sınıfta"),
    ("deftre", "defter"),
    ("ogrencii", "öğrenci"),
    ("ogretmen", "öğretmen"),
    ]
        
        print("=" * 60)
        print("TURKISH SPELLING CORRECTION TEST (DIACRITIC-AWARE)")
        print("=" * 60)
        
        correct = 0
        total = len(test_cases)
        
        for misspelled, expected in test_cases:
            
            corrected = spelling_correction(misspelled, lexicon_path, max_edit_distance=2)
            is_correct = corrected == expected
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status} Input: '{misspelled}' → Output: '{corrected}' (Expected: '{expected}')")
        
        print("=" * 60)
        print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
        print("=" * 60)
        
        print("\nTesting with out-of-vocabulary words:")
        oov_words = ["zxqwerty", "abcdefgh", "qqqqq"]
        for word in oov_words:
            result = spelling_correction(word, lexicon_path, max_edit_distance=2)
            print(f"  Input: '{word}' → Output: '{result}'")
        
    finally:
        pass

if __name__ == "__main__":
    test_spelling_correction()
    # Create temporary lexicon file with Turkish words
    
    