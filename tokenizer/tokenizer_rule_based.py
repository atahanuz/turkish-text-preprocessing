import re
from typing import List


class TurkishTokenizer:
    def __init__(self, mwe_file: str = "multiword.txt"):
        """
        Initialize the Turkish tokenizer.

        Args:
            mwe_file: Path to file containing multi-word expressions (one per line)
        """
        self.mwes = []
        if mwe_file:
            self.load_mwes(mwe_file)

        # Regex patterns for special entities
        self.url_pattern = re.compile(
            r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.number_pattern = re.compile(r'\b\d+([.,]\d+)*\b')

    def load_mwes(self, mwe_file: str):
        """Load multi-word expressions from file."""
        try:
            with open(mwe_file, 'r', encoding='utf-8') as f:
                self.mwes = [line.strip() for line in f if line.strip()]
            # Sort by length (longest first) for greedy matching
            self.mwes.sort(key=len, reverse=True)
        except FileNotFoundError:
            print(f"Warning: MWE file '{mwe_file}' not found. Continuing without MWEs.")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Turkish text into tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Store special entities with placeholders
        entities = []
        processed_text = text

        # Extract and replace special entities with placeholders
        for pattern_name, pattern in [
            ('URL', self.url_pattern),
            ('EMAIL', self.email_pattern),
            ('HASHTAG', self.hashtag_pattern),
            ('MENTION', self.mention_pattern),
        ]:
            for match in pattern.finditer(processed_text):
                entity = match.group()
                placeholder = f"___{pattern_name}_{len(entities)}___"
                entities.append(entity)
                processed_text = processed_text.replace(entity, placeholder, 1)

        # Handle multi-word expressions
        processed_text_lower = processed_text.lower()
        for mwe in self.mwes:
            mwe_lower = mwe.lower()
            if mwe_lower in processed_text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = processed_text_lower.find(mwe_lower, start)
                    if pos == -1:
                        break

                    # Check if it's a word boundary match
                    before_ok = pos == 0 or not processed_text[pos - 1].isalnum()
                    after_pos = pos + len(mwe)
                    after_ok = after_pos >= len(processed_text) or not processed_text[after_pos].isalnum()

                    if before_ok and after_ok:
                        # Replace with underscore-connected version
                        mwe_token = processed_text[pos:pos + len(mwe)].replace(' ', '_')
                        processed_text = (
                                processed_text[:pos] +
                                mwe_token +
                                processed_text[pos + len(mwe):]
                        )
                        processed_text_lower = processed_text.lower()
                        start = pos + len(mwe_token)
                    else:
                        start = pos + 1

        # Split by whitespace
        tokens = processed_text.split()

        # Separate punctuation into individual tokens
        tokens_with_punct = []
        for token in tokens:
            # Skip if token is a placeholder (will be restored later)
            if '___' in token and any(p in token for p in ['URL', 'EMAIL', 'HASHTAG', 'MENTION']):
                tokens_with_punct.append(token)
                continue

            # Separate leading and trailing punctuation
            current = token
            result = []

            # Extract leading punctuation
            while current and not current[0].isalnum() and current[0] != '_':
                result.append(current[0])
                current = current[1:]

            # Find the core word/token
            if current:
                # Extract trailing punctuation
                trailing_punct = []
                while current and not current[-1].isalnum() and current[-1] != '_':
                    trailing_punct.insert(0, current[-1])
                    current = current[:-1]

                # Add core token if it exists
                if current:
                    result.append(current)

                # Add trailing punctuation
                result.extend(trailing_punct)

            tokens_with_punct.extend(result)

        # Restore special entities
        final_tokens = []
        for token in tokens_with_punct:
            # Check if token contains a placeholder
            restored_token = token
            for i, entity in enumerate(entities):
                for pattern_name in ['URL', 'EMAIL', 'HASHTAG', 'MENTION']:
                    placeholder = f"___{pattern_name}_{i}___"
                    if placeholder in restored_token:
                        restored_token = restored_token.replace(placeholder, entity)
            final_tokens.append(restored_token)

        return final_tokens


# Example usage
if __name__ == "__main__":
    # Initialize tokenizer with MWE file
    tokenizer = TurkishTokenizer('../data/multiword.txt')

    # Example sentences
    examples = [
        "Bugün hava çok güzel.",
        "Web sitemiz https://example.com adresinde.",
        "Bana info@example.com adresinden yazabilirsiniz.",
        "Bu konuya değer vermek gerekir.",
        "#Python ile #NLP çalışması yapıyorum.",
        "Arkadaşım @ahmet ile görüştüm, değil mi?",
        "Fiyat 123.45 TL olarak belirlendi.",
        "Deşifre etmek için şifreyi bilmek gerekir."
    ]

    print("Turkish Tokenizer Examples\n" + "=" * 50)
    for text in examples:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput:  {text}")
        print(f"Tokens: {tokens}")