# Turkish Sentence Splitter

A sophisticated sentence splitter specifically designed for Turkish text that handles complex cases including quotations, abbreviations, and various punctuation patterns.

## Features

- **Multiple End-of-Sentence Markers**: Handles periods (.), exclamation marks (!), question marks (?), and their combinations
- **Quotation Handling**: Properly processes sentences within quotations using various quote styles (" ", ' ', « »)
- **Abbreviation Detection**: Recognizes common Turkish abbreviations (Dr., Prof., Ltd., A.Ş., etc.)
- **Decimal Number Support**: Correctly identifies decimal points (e.g., 3.14) vs. sentence-ending periods
- **Ellipsis Recognition**: Handles ellipsis (...) without breaking sentences prematurely
- **Multiple Punctuation**: Manages sequences like "!!", "!?", etc.

## Usage

### Standalone

```python
from sentence_splitter import TurkishSentenceSplitter

splitter = TurkishSentenceSplitter()
text = "Bugün hava çok güzel. Yarın da güzel olacak."
sentences = splitter.split(text)

# Output: ['Bugün hava çok güzel.', 'Yarın da güzel olacak.']
```

### API Endpoint

The sentence splitter is available via the Flask API:

```bash
curl -X POST http://localhost:5001/api/split_sentences \
  -H "Content-Type: application/json" \
  -d '{"text": "Bugün hava çok güzel. Yarın da güzel olacak."}'
```

Response:
```json
{
  "sentence_count": 2,
  "sentences": [
    "Bugün hava çok güzel.",
    "Yarın da güzel olacak."
  ]
}
```

## Examples

### Basic Sentences
Input: `Bugün hava çok güzel. Yarın da güzel olacak.`
Output:
- `Bugün hava çok güzel.`
- `Yarın da güzel olacak.`

### Quotations
Input: `Ali "Bugün hava çok güzel." dedi. Ayşe de "Evet, haklısın." diye cevap verdi.`
Output:
- `Ali "Bugün hava çok güzel." dedi. Ayşe de "Evet, haklısın." diye cevap verdi.`

Note: Quotations within dialogue are kept together with their attribution ("dedi", "diye sordu", etc.)

### Abbreviations
Input: `Dr. Ahmet Yılmaz geldi. Prof. Ayşe Demir de oradaydı.`
Output:
- `Dr. Ahmet Yılmaz geldi.`
- `Prof. Ayşe Demir de oradaydı.`

### Decimal Numbers
Input: `Fiyat 3.14 TL idi. Yeni fiyat 5.99 TL oldu.`
Output:
- `Fiyat 3.14 TL idi.`
- `Yeni fiyat 5.99 TL oldu.`

### Multiple Punctuation
Input: `Ne yapıyorsun?! Dur bir dakika!!`
Output:
- `Ne yapıyorsun?!`
- `Dur bir dakika!!`

## Implementation Details

The sentence splitter uses a state-based approach that:

1. Tracks quotation state using a stack to handle nested quotes
2. Checks for abbreviations against a predefined list of common Turkish abbreviations
3. Distinguishes between decimal points and sentence-ending periods
4. Handles ellipsis by detecting sequences of three or more dots
5. Processes multiple consecutive punctuation marks correctly
6. Considers capitalization patterns to determine sentence boundaries

## Supported Abbreviations

The splitter recognizes these common Turkish abbreviations:
- Academic titles: Dr., Prof., Doç., Yrd.
- Business: Ltd., A.Ş., Şti., Inc.
- General: No., Tel., vs., vb., ör., etc.
- Address: Cad., Sok., Mah., Blv.
- And many more...

## Testing

Run the test suite:

```bash
# Test the module directly
python sentence_splitter/sentence_splitter.py

# Test the API endpoint
python test_sentence_api.py
```
