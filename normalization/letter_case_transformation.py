tr_upper = {'ı': 'I', 'i': 'İ', 'ş': 'Ş', 'ç': 'Ç', 'ö': 'Ö', 'ü': 'Ü', 'ğ': 'Ğ'}

def turkish_lower(text):
    tr_upper_to_lower = {
        'I': 'ı',
        'İ': 'i',
        'Ş': 'ş',
        'Ç': 'ç',
        'Ö': 'ö',
        'Ü': 'ü',
        'Ğ': 'ğ',
    }
    result = []
    for c in text:
        if c in tr_upper_to_lower:
            result.append(tr_upper_to_lower[c])
        else:
            result.append(c.lower())
    return ''.join(result)


def turkish_upper(text):
    tr_lower_to_upper = {
        'ı': 'I',
        'i': 'İ',
        'ş': 'Ş',
        'ç': 'Ç',
        'ö': 'Ö',
        'ü': 'Ü',
        'ğ': 'Ğ',
    }
    result = []
    for c in text:
        if c in tr_lower_to_upper:
            result.append(tr_lower_to_upper[c])
        else:
            result.append(c.upper())
    return ''.join(result)


def turkish_capitalize(text):
    if not text:
        return text
    first_char = text[0]
    rest = text[1:] if len(text) > 1 else ""
    
    # Capitalize first character using Turkish rules
    if first_char in tr_upper:
        first_upper = tr_upper[first_char]
    else:
        first_upper = first_char.upper()
    
    return first_upper + turkish_lower(rest)


def is_turkish_lowercase(text):
    return text == turkish_lower(text)


def is_turkish_uppercase(text):
    return text == turkish_upper(text)


def is_turkish_proper(text):
    if len(text) == 0:
        return False
    if len(text) == 1:
        return text == turkish_upper(text)
    return text[0] == turkish_upper(text[0]) and text[1:] == turkish_lower(text[1:])


def letter_case_transformation(token):
    has_apostrophe = "'" in token
    has_period = "." in token
    
    is_lowercase = is_turkish_lowercase(token)
    is_uppercase = is_turkish_uppercase(token)
    is_proper = is_turkish_proper(token)
    is_mixed = not is_lowercase and not is_uppercase and not is_proper
    
    # Already proper noun case with apostrophe - leave untouched
    if is_proper and has_apostrophe:
        return token
    
    # Lowercase case
    if is_lowercase:
        if has_apostrophe or has_period:
            return turkish_capitalize(token)
        return token
    
    # UPPERCASE or miXEd CaSe
    if is_uppercase or is_mixed:
        if has_apostrophe or has_period:
            return turkish_capitalize(token)
        return turkish_lower(token)
    
    # Proper noun case without apostrophe - convert to lowercase for next stages
    return turkish_lower(token)


# Test cases - All Turkish
if __name__ == "__main__":
    test_cases = [
        # Lowercase without apostrophe/period - unchanged
        ("umuttan", "umuttan"),
        ("çiçek", "çiçek"),
        ("şeker", "şeker"),
        ("ağaç", "ağaç"),
        ("gitmeyeceğim", "gitmeyeceğim"),
        ("ışık", "ışık"),
        ("istanbul", "istanbul"),
        
        # Lowercase with apostrophe - convert to proper case (Turkish names/places)
        ("ahmet'ten", "Ahmet'ten"),
        ("istanbul'dan", "İstanbul'dan"),
        ("ankara'ya", "Ankara'ya"),
        ("ayşe'nin", "Ayşe'nin"),
        ("çağla'dan", "Çağla'dan"),
        ("ığdır'dan", "Iğdır'dan"),
        
        # Lowercase with period (Turkish abbreviations)
        ("dr.", "Dr."),
        ("prof.", "Prof."),
        ("yrd.", "Yrd."),
        
        # Proper case with apostrophe - leave untouched (valid Turkish proper nouns)
        ("Ahmet'ten", "Ahmet'ten"),
        ("Umut'tan", "Umut'tan"),
        ("Ankara'dakiler", "Ankara'dakiler"),
        ("İstanbul'a", "İstanbul'a"),
        ("Çağla'dan", "Çağla'dan"),
        ("Ayşe'nin", "Ayşe'nin"),
        
        # UPPERCASE with apostrophe/period - convert to proper case
        ("AHMET'TEN", "Ahmet'ten"),
        ("ANKARA'YA", "Ankara'ya"),
        ("ÇAĞLA'DAN", "Çağla'dan"),
        ("DR.", "Dr."),
        ("PROF.", "Prof."),
        ("IŞIK'TAN", "Işık'tan"),
        
        # UPPERCASE without apostrophe/period - convert to lowercase
        ("TÜRKÇE", "türkçe"),
        ("İSTANBUL", "istanbul"),
        ("ÇALIŞMAK", "çalışmak"),
        ("ÖĞRENCİ", "öğrenci"),
        ("IŞIK", "ışık"),
        
        # miXEd CaSe with apostrophe - convert to proper case
        ("aHmEt'TeN", "Ahmet'ten"),
        ("aNkArA'yA", "Ankara'ya"),
        ("çAğLa'DaN", "Çağla'dan"),
        ("IşIk'TaN", "Işık'tan"),
        
        # miXEd CaSe without apostrophe/period - convert to lowercase
        ("TüRkÇe", "türkçe"),
        ("İsTaNbUl", "istanbul"),
        ("ÇaLıŞmAk", "çalışmak"),
        ("IşIk", "ışık"),
        
        # Proper case without apostrophe - convert to lowercase (for next stages)
        ("Ahmet", "ahmet"),
        ("Umut", "umut"),
        ("Çiçek", "çiçek"),
        ("Şeker", "şeker"),
        ("İpek", "ipek"),
        ("Işık", "ışık"),
        
        # Turkish edge cases with I/İ and ı/i
        ("I", "ı"),
        ("İ", "i"),
        ("IŞIK", "ışık"),
        ("İĞDIR'DAN", "Iğdır'dan"),
        
        # Multiple apostrophes (rare but possible)
        ("ali'nin'ki", "Ali'nin'ki"),
        
        # Single character edge cases
        ("i", "i"),
        ("ı", "ı"),
    ]
    
    print("Testing letter_case_transformation - Turkish Test Cases:")
    all_passed = True
    for input_token, expected in test_cases:
        result = letter_case_transformation(input_token)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} Girdi: '{input_token}' → Çıktı: '{result}' (Beklenen: '{expected}')")
    
    print(f"\n{'Tüm testler başarılı!' if all_passed else 'Bazı testler başarısız!'}")