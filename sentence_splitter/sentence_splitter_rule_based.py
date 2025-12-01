import re
from typing import List, Tuple


class TurkishSentenceSplitter:
 

    def __init__(self):
        """Initialize the Turkish sentence splitter with common abbreviations."""
        # Common Turkish abbreviations that should NOT trigger sentence breaks
        self.abbreviations = {
            'dr', 'prof', 'doç', 'yrd', 'yar',  # Academic titles
            'mr', 'mrs', 'ms',  # English titles sometimes used
            'ltd', 'a.ş', 'a.ş', 'şti', 'inc', 'corp',  # Company suffixes
            'no', 'tel', 'faks', 'gsm',  # Contact info
            'vs', 'vb', 'vd',  # Et cetera equivalents
            'kr', 'blv', 'cad', 'sok', 'mah',  # Address components
            'ör', 'örn',  # Example (örnek/örneğin)
            'dz', 'kd', 'tğm', 'bnb', 'yb', 'alb', 'korg',  # Military ranks
            'bkz', 'sf', 'sy', 's',  # Reference abbreviations
            'apt', 'daire', 'kat',  # Building info
            'st', 'ave', 'rd',  # Street abbreviations (English)
        }

        # Quote pairs - opening and closing quotes
        self.quote_pairs = [
            ('"', '"'),  # Standard double quotes
            ("'", "'"),  # Single quotes
            ('«', '»'),  # French guillemets (sometimes used in Turkish)
            ('"', '"'),  # Smart quotes
            (''', '''),  # Smart single quotes
        ]

        # End of sentence markers
        self.eos_markers = {'.', '!', '?'}

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text to split into sentences

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        # Preprocess: normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        sentences = []
        current_sentence = []
        quote_stack = []  # Track open quotes
        i = 0

        while i < len(text):
            char = text[i]
            current_sentence.append(char)

            # Track quote state
            if self._is_quote_start(char):
                quote_stack.append(char)
            elif self._is_quote_end(char, quote_stack):
                if quote_stack:
                    quote_stack.pop()
                    # Check if we should break after closing quote
                    if self._should_break_after_quote(text, i):
                        sentence = ''.join(current_sentence).strip()
                        if sentence:
                            sentences.append(sentence)
                        current_sentence = []
                        i += 1
                        continue

            # Check for potential sentence boundary
            if char in self.eos_markers:
                # Check if we should break here
                if self._should_break(text, i, quote_stack):
                    # Extract the sentence
                    sentence = ''.join(current_sentence).strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = []

            i += 1

        # Add any remaining text as the last sentence
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def _is_quote_start(self, char: str) -> bool:
        """Check if character is an opening quote."""
        return char in [pair[0] for pair in self.quote_pairs]

    def _is_quote_end(self, char: str, quote_stack: List[str]) -> bool:
        """Check if character is a closing quote matching the most recent opening quote."""
        if not quote_stack:
            return False

        last_open = quote_stack[-1]
        for open_q, close_q in self.quote_pairs:
            if last_open == open_q and char == close_q:
                return True
        return False

    def _should_break_after_quote(self, text: str, pos: int) -> bool:
        """
        Check if we should break after a closing quote.

        Args:
            text: Full text
            pos: Current position (at a closing quote)

        Returns:
            True if we should break after this quote
        """
        # Look for punctuation immediately before the quote
        has_punct_inside = False
        check_pos = pos - 1
        while check_pos >= 0 and text[check_pos] in self.eos_markers:
            has_punct_inside = True
            check_pos -= 1

        if not has_punct_inside:
            return False

        # Look ahead to see what comes next
        next_pos = pos + 1
        while next_pos < len(text) and text[next_pos].isspace():
            next_pos += 1

        if next_pos >= len(text):
            return True  # End of text

        next_char = text[next_pos]

        # Check if there's a period or other sentence marker after the quote
        # e.g., "Hello." dedi.
        if next_char in self.eos_markers:
            return False  # Don't break, wait for the outer punctuation

        # Break if next sentence starts with capital letter
        if next_char.isupper():
            return True

        # Break if another quote starts (new dialogue)
        if self._is_quote_start(next_char):
            return True

        return False

    def _should_break(self, text: str, pos: int, quote_stack: List[str]) -> bool:
        """
        Determine if we should break the sentence at this position.

        Args:
            text: Full text
            pos: Current position (at an EOS marker)
            quote_stack: Current quote stack

        Returns:
            True if we should break, False otherwise
        """
        # Don't break if we're inside quotes
        if quote_stack:
            # Don't break inside quotes
            return False

        # Check for ellipsis (...)
        if self._is_ellipsis(text, pos):
            # We're at the third dot of an ellipsis - treat as sentence boundary
            # Always break after ellipsis to treat it as a sentence separator
            return True

        # Check for abbreviations
        if self._is_abbreviation(text, pos):
            return False

        # Check for decimal numbers (e.g., 3.14)
        if self._is_decimal_point(text, pos):
            return False

        # Check for multiple punctuation (e.g., "!!", "!?")
        if self._is_multiple_punctuation(text, pos):
            # Only break after the last punctuation mark
            return self._is_last_in_punctuation_sequence(text, pos)

        # Check what comes after
        if pos + 1 < len(text):
            next_char = text[pos + 1]

            # Skip whitespace to find the actual next character
            next_pos = pos + 1
            while next_pos < len(text) and text[next_pos].isspace():
                next_pos += 1

            if next_pos < len(text):
                next_nonspace = text[next_pos]

                # Check if next character is a closing quote
                if self._is_quote_end(next_nonspace, quote_stack if quote_stack else ['"']):
                    # Look ahead after the quote
                    after_quote_pos = next_pos + 1
                    while after_quote_pos < len(text) and text[after_quote_pos].isspace():
                        after_quote_pos += 1

                    if after_quote_pos >= len(text):
                        return True  # End of text after quote

                    after_quote_char = text[after_quote_pos]
                    # Break if next sentence starts with capital or is another quote
                    return after_quote_char.isupper() or self._is_quote_start(after_quote_char)

                # Check if next char is lowercase
                if next_nonspace.islower():
                    # Check for coordinating conjunctions that might start sentences
                    remaining_text = text[next_pos:].lower()
                    sentence_starters = ['ve ', 'ama ', 'fakat ', 'ancak ', 'lakin ', 'veya ', 'yahut ']
                    if any(remaining_text.startswith(starter) for starter in sentence_starters):
                        return True

                    # For lowercase letters after period, still break to allow splitting
                    # This handles informal text and all-lowercase input
                    return True

                # Break if next char is uppercase or a digit (new sentence)
                if next_nonspace.isupper() or next_nonspace.isdigit():
                    return True

                # Break if next char is a quote start (new quoted sentence)
                if self._is_quote_start(next_nonspace):
                    return True
        else:
            # End of text
            return True

        # Default: break
        return True

    def _is_ellipsis(self, text: str, pos: int) -> bool:
        """Check if position is part of an ellipsis (...)"""
        if text[pos] != '.':
            return False

        # Check for three consecutive dots starting at current position
        if pos + 2 < len(text) and text[pos:pos+3] == '...':
            # We're at the start of an ellipsis
            # Check if this is the last dot (meaning we're at pos+2)
            return False  # Don't treat as ellipsis yet, wait for the third dot

        # Check if we're in the middle of ellipsis
        if pos >= 1 and text[pos-1] == '.':
            if pos + 1 < len(text) and text[pos+1] == '.':
                # We're in the middle
                return False  # Wait for the third dot
            elif pos >= 2 and text[pos-2] == '.':
                # We're at the end of an ellipsis (third dot)
                return True

        # Check if we just completed an ellipsis pattern
        if pos >= 2 and text[pos-2:pos+1] == '...':
            # We're at the third dot of an ellipsis
            return True

        return False

    def _is_abbreviation(self, text: str, pos: int) -> bool:
        """Check if the period is part of an abbreviation."""
        if text[pos] != '.':
            return False

        # Extract word before the period
        start = pos - 1
        while start >= 0 and (text[start].isalnum() or text[start] == '.'):
            start -= 1
        start += 1

        word = text[start:pos].lower().strip()

        # Remove any dots within the word for comparison
        word_clean = word.replace('.', '')

        # Check against known abbreviations
        if word_clean in self.abbreviations:
            return True

        # Check for single letter abbreviations (e.g., "S. Ahmet")
        if len(word_clean) == 1 and word_clean.isalpha():
            return True

        # Check for pattern like "A.Ş." or "vs."
        if '.' in word and len(word_clean) <= 4:
            return True

        return False

    def _is_decimal_point(self, text: str, pos: int) -> bool:
        """Check if period is part of a decimal number."""
        if text[pos] != '.':
            return False

        # Check for digit before and after
        has_digit_before = pos > 0 and text[pos - 1].isdigit()
        has_digit_after = pos + 1 < len(text) and text[pos + 1].isdigit()

        return has_digit_before and has_digit_after

    def _is_multiple_punctuation(self, text: str, pos: int) -> bool:
        """Check if there are multiple punctuation marks in sequence."""
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            return next_char in self.eos_markers
        return False

    def _is_last_in_punctuation_sequence(self, text: str, pos: int) -> bool:
        """Check if this is the last punctuation mark in a sequence."""
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            return next_char not in self.eos_markers
        return True


# Example usage and testing
if __name__ == "__main__":
    splitter = TurkishSentenceSplitter()

    # Test cases
    test_texts = [
        # Basic sentences
        'Bugün hava çok güzel. Yarın da güzel olacak.',

        # Exclamation and question marks
        'Ne güzel bir gün! Değil mi? Kesinlikle öyle.',

        # Quotations
        'Ali "Bugün hava çok güzel." dedi. Ayşe de "Evet, haklısın." diye cevap verdi.',

        # Nested and mixed quotes
        "Öğretmen 'Ders çalışın!' dedi. Öğrenciler çalıştı.",

        # Abbreviations
        'Dr. Ahmet Yılmaz geldi. Prof. Ayşe Demir de oradaydı.',

        # Numbers and decimals
        'Fiyat 3.14 TL idi. Yeni fiyat 5.99 TL oldu.',

        # Ellipsis
        'Belki... Ama emin değilim. Yarın göreceğiz.',

        # Multiple punctuation
        'Ne yapıyorsun?! Dur bir dakika!! Şimdi geliyorum...',

        # Quotes with punctuation inside
        'Ali "Nereye gidiyorsun?" diye sordu. Ayşe "İşe gidiyorum." dedi.',

        # Complex case
        'Şirket A.Ş. olarak kuruldu. Genel Müdür Dr. Mehmet Yılmaz, "Hedefimiz büyümek." dedi. '
        'Proje 2.5 milyon TL tutarında... Başarılı olacağız!',

        # Edge cases
        'Cümle bir. Cümle iki. Cümle üç.',

        # Lowercase after period (shouldn't break)
        'Örneğin şu durum. ama başka bir şey.',

        # No ending punctuation
        'Bu bir cümle',

        # Coordinating conjunctions
        'Hava soğuktu. Ama dışarı çıktık. Ve geziye gittik.',
    ]

    print("Turkish Sentence Splitter Test Results")
    print("=" * 80)

    for i, text in enumerate(test_texts, 1):
        sentences = splitter.split(text)
        print(f"\nTest {i}:")
        print(f"Input: {text}")
        print(f"Sentences ({len(sentences)}):")
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}. {sentence}")
