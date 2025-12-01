import re
from typing import List, Tuple


# Normalization patterns: colloquial → standard
FUTURE_PATTERNS = [
    # Vowel-omitted forms: (stem ending in vowel)(consonant)cak → (stem)(consonant)acak
    # This ensures we only match when a consonant DIRECTLY precedes cak/cek (vowel omitted)
    # and NOT when there's already a vowel there (standard form like "yapacak")
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cak\b', r'\1\2acak'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cam\b', r'\1\2acağım'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])can\b', r'\1\2acaksın'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])caz\b', r'\1\2acağız'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])caklar\b', r'\1\2acaklar'),
    
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cek\b', r'\1\2ecek'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cem\b', r'\1\2eceğim'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cen\b', r'\1\2eceksin'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cez\b', r'\1\2eceğiz'),
    (r'\b(\w*[aeiouöüı])([bçdfghjklmnprsştvyz])cekler\b', r'\1\2ecekler'),

    # Sesli harfle biten fiil kökleri: okucam → okuyacağım, başlıcaz → başlayacağız
    (r'\b(\w+[aıou])cam\b', r'\1yacağım'),
    (r'\b(\w+[aıou])can\b', r'\1yacaksın'),
    (r'\b(\w+[aıou])cak\b', r'\1yacak'),
    (r'\b(\w+[aıou])caz\b', r'\1yacağız'),

    (r'\b(\w+[eöiü])cem\b', r'\1yeceğim'),
    (r'\b(\w+[eöiü])cen\b', r'\1yeceksin'),
    (r'\b(\w+[eöiü])cek\b', r'\1yecek'),
    (r'\b(\w+[eöiü])cez\b', r'\1yeceğiz'),
    
    # Standard accent normalizations
    # -acam/-ıcam → -acağım
    (r'(\w+)acam\b', r'\1acağım'),
    (r'(\w+)ıcam\b', r'\1acağım'),
    (r'(\w+)icam\b', r'\1eceğim'),
    
    # -ecem/-icem → -eceğim
    (r'(\w+)ecem\b', r'\1eceğim'),
    (r'(\w+)icem\b', r'\1eceğim'),
    
    # -acan/-ıcan → -acaksın
    (r'(\w+)acan\b', r'\1acaksın'),
    (r'(\w+)ıcan\b', r'\1acaksın'),
    
    # -ecen/-icen → -eceksin
    (r'(\w+)ecen\b', r'\1eceksin'),
    (r'(\w+)icen\b', r'\1eceksin'),
    
    # -acaz/-ıcaz → -acağız
    (r'(\w+)acaz\b', r'\1acağız'),
    (r'(\w+)ıcaz\b', r'\1acağız'),
    
    # -ecez/-icez → -eceğiz
    (r'(\w+)ecez\b', r'\1eceğiz'),
    (r'(\w+)icez\b', r'\1eceğiz'),
    
    # -ıcak/-icek → -acak/-ecek
    (r'(\w+)ıcak\b', r'\1acak'),
    (r'(\w+)icek\b', r'\1ecek'),
]

PRESENT_CONTINUOUS_PATTERNS = [
    # -iyom/-iom → -iyorum
    (r'(\w+)iyom\b', r'\1iyorum'),
    (r'(\w+)iom\b', r'\1iyorum'),
    
    # -ıyom/-ıom → -ıyorum
    (r'(\w+)ıyom\b', r'\1ıyorum'),
    (r'(\w+)ıom\b', r'\1ıyorum'),
    
    # -uyom/-uom → -uyorum
    (r'(\w+)uyom\b', r'\1uyorum'),
    (r'(\w+)uom\b', r'\1uyorum'),
    
    # -üyom/-üom → -üyorum
    (r'(\w+)üyom\b', r'\1üyorum'),
    (r'(\w+)üom\b', r'\1üyorum'),
    
    # -iyon/-iyosun → -iyorsun
    (r'(\w+)iyon\b', r'\1iyorsun'),
    (r'(\w+)iyosun\b', r'\1iyorsun'),
    
    # -ıyon/-ıyosun → -ıyorsun
    (r'(\w+)ıyon\b', r'\1ıyorsun'),
    (r'(\w+)ıyosun\b', r'\1ıyorsun'),
    
    # -uyon/-uyosun → -uyorsun
    (r'(\w+)uyon\b', r'\1uyorsun'),
    (r'(\w+)uyosun\b', r'\1uyorsun'),
    
    # -üyon/-üyosun → -üyorsun
    (r'(\w+)üyon\b', r'\1üyorsun'),
    (r'(\w+)üyosun\b', r'\1üyorsun'),
    
    # -iyo → -iyor
    (r'(\w+)iyo\b', r'\1iyor'),
    (r'(\w+)ıyo\b', r'\1ıyor'),
    (r'(\w+)uyo\b', r'\1uyor'),
    (r'(\w+)üyo\b', r'\1üyor'),
    
    # -iyoz → -iyoruz
    (r'(\w+)iyoz\b', r'\1iyoruz'),
    (r'(\w+)ıyoz\b', r'\1ıyoruz'),
]

NEGATIVE_FUTURE_PATTERNS = [
    # -mıycam/-mayacam → -mayacağım
    (r'(\w+)mıycam\b', r'\1mayacağım'),
    (r'(\w+)mayacam\b', r'\1mayacağım'),
    
    # -miycem/-meyecem → -meyeceğim
    (r'(\w+)miycem\b', r'\1meyeceğim'),
    (r'(\w+)meyecem\b', r'\1meyeceğim'),
    
    # -mıycak → -mayacak
    (r'(\w+)mıycak\b', r'\1mayacak'),
    (r'(\w+)miycek\b', r'\1meyecek'),
    
    # -mıycan → -mayacaksın
    (r'(\w+)mıycan\b', r'\1mayacaksın'),
    (r'(\w+)miyecen\b', r'\1meyeceksin'),
]

QUESTION_PATTERNS = [
    # Attached questions: geliyonmu → geliyor musun
    (r'(\w+)iyonmu\b', r'\1iyor musun'),
    (r'(\w+)ıyonmu\b', r'\1ıyor musun'),
    (r'(\w+)uyonmu\b', r'\1uyor musun'),
    (r'(\w+)üyonmu\b', r'\1üyor musun'),
    
    # -yomu variants
    (r'(\w+)iyomu\b', r'\1iyor mu'),
    (r'(\w+)ıyomu\b', r'\1ıyor mu'),
]

OTHER_PATTERNS = [
    # Copula restoration: -dı/-di → -dır/-dir (context-dependent, simplified)
    (r'(\w+[aıou])dı\b', r'\1dır'),
    (r'(\w+[eiöü])di\b', r'\1dir'),
    (r'(\w+[aıou])du\b', r'\1dur'),
    (r'(\w+[eiöü])dü\b', r'\1dür'),
]


def apply_patterns(text: str, patterns: List[Tuple[str, str]]) -> str:
    """
    Apply regex patterns to normalize text.
    
    Args:
        text: Input text (colloquial)
        patterns: List of (pattern, replacement) tuples
    
    Returns:
        Normalized text (standard)
    """
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text


def normalize_future_tense(text: str) -> str:
    """Normalize colloquial future tense to standard"""
    return apply_patterns(text, FUTURE_PATTERNS)


def normalize_present_continuous(text: str) -> str:
    """Normalize colloquial present continuous to standard"""
    return apply_patterns(text, PRESENT_CONTINUOUS_PATTERNS)


def normalize_negative_future(text: str) -> str:
    """Normalize colloquial negative future to standard"""
    return apply_patterns(text, NEGATIVE_FUTURE_PATTERNS)


def normalize_questions(text: str) -> str:
    """Normalize colloquial question forms to standard"""
    return apply_patterns(text, QUESTION_PATTERNS)


def normalize_other(text: str) -> str:
    """Normalize other colloquial patterns to standard"""
    return apply_patterns(text, OTHER_PATTERNS)


def normalize_accent(text: str, patterns: List[str] = None) -> str:
    """
    Main normalization function: colloquial → standard Turkish.
    
    Args:
        text: Input text (colloquial/informal)
        patterns: List of pattern types ['future', 'present', 'negative', 'question', 'other']
                 If None, applies all patterns
    
    Returns:
        Normalized text (standard/formal)
    """
    if patterns is None:
        # Apply negative BEFORE future to handle negative future forms correctly
        patterns = ['negative', 'future', 'present', 'question', 'other']
    
    normalizers = {
        'future': normalize_future_tense,
        'present': normalize_present_continuous,
        'negative': normalize_negative_future,
        'question': normalize_questions,
        'other': normalize_other
    }
    
    for pattern_type in patterns:
        if pattern_type in normalizers:
            text = normalizers[pattern_type](text)
    
    return text


def batch_normalize(texts: List[str], **kwargs) -> List[str]:
    """
    Normalize multiple texts.
    
    Args:
        texts: List of input texts
        **kwargs: Arguments to pass to normalize_accent
    
    Returns:
        List of normalized texts
    """
    return [normalize_accent(text, **kwargs) for text in texts]


# Example usage
if __name__ == '__main__':
    test_cases = [
        "yarin istanbula gidicem",
        "ne yapıyon su anda?",
        "bugün sinemaya gitmiycem",
        "gelecek hafta baslıcaz",
        "sen de gelecek misin?",
        "universitede calisiyom",
        "yarin saat kacta geliyon?",
        "bu isi yapıcan degil mi?",
        "okula gidiyom ama yarin gitmiycem",
        "ne yapiyonmu sen?",
    ]
    
    print("Accent Normalization: Colloquial → Standard")
    print("=" * 80)
    for text in test_cases:
        normalized = normalize_accent(text)
        if normalized != text:
            print(f"Colloquial: {text}")
            print(f"Standard:   {normalized}")
            print()