from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizer.tokenizer_rule_based import TurkishTokenizer
from tokenizer.tokenizer_inference import load_trained_tokenizer, tokenize as ml_tokenize
from stemmer.stemmer_v2 import ImprovedTurkishStemmer
from normalization.normalization_pipeline import NormalizationPipeline
from stopword_removal.static_stopword_removal import remove_stopwords, load_stopwords
from sentence_splitter.sentence_splitter_rule_based import TurkishSentenceSplitter
from sentence_splitter.sentence_splitter_naive_bayes import TurkishSentenceSplitter as NaiveBayesSentenceSplitter

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize tokenizer, stemmer, normalization pipeline, and sentence splitters
tokenizer = TurkishTokenizer('data/multiword.txt')
stemmer = ImprovedTurkishStemmer(
    suffix_categories_file='data/suffix_categories.csv',
    words_file="data/Turkish_Corpus_3M.txt",
    min_stem_length=2
)
normalizer = NormalizationPipeline(lexicon_path='data/Turkish_Corpus_3M.txt', use_morpho_parser=True)

# Initialize rule-based sentence splitter
sentence_splitter = TurkishSentenceSplitter()

# Initialize Naive Bayes sentence splitter and load trained model
nb_sentence_splitter = NaiveBayesSentenceSplitter()
nb_sentence_splitter.load_model('data/sentence_splitter_nb_model.pkl')

# Load ML-based tokenizer model and vectorizer
ml_tokenizer_model, ml_tokenizer_vectorizer = load_trained_tokenizer(
    model_path='tokenizer/tokenizer_model.pkl',
    vectorizer_path='tokenizer/tokenizer_vectorizer.pkl'
)

# Load TF-IDF stopwords at startup
def load_tfidf_stopwords(filepath):
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word.lower())
    return stopwords

# Load TF-IDF stopwords (static stopwords are loaded via the static_stopword_removal module)
tfidf_stopwords = load_tfidf_stopwords('stopword_removal/tf-idf_stopwords.txt')

@app.route('/api/split_sentences', methods=['POST'])
def split_sentences():
    """Split text into sentences using rule-based sentence splitter"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Split text into sentences using rule-based splitter
    sentences = sentence_splitter.split(text)

    return jsonify({
        'sentences': sentences,
        'sentence_count': len(sentences)
    })

@app.route('/api/split_sentences_nb', methods=['POST'])
def split_sentences_nb():
    """Split text into sentences using Naive Bayes sentence splitter"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Split text into sentences using Naive Bayes splitter
    sentences = nb_sentence_splitter.split_sentences(text)

    return jsonify({
        'sentences': sentences,
        'sentence_count': len(sentences)
    })

@app.route('/api/tokenize_rule', methods=['POST'])
def tokenize_rule():
    """Rule-based tokenization"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Get tokens using rule-based tokenizer
    tokens = tokenizer.tokenize(text)

    # Get unique tokens
    unique_tokens = list(dict.fromkeys(tokens))  # Preserves order

    return jsonify({
        'tokens': tokens,
        'unique_tokens': unique_tokens
    })

@app.route('/api/tokenize_ml', methods=['POST'])
def tokenize_ml():
    """ML-based tokenization using trained model"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Get tokens using ML-based tokenizer
    tokens = ml_tokenize(text, ml_tokenizer_model, ml_tokenizer_vectorizer)

    # Get unique tokens
    unique_tokens = list(dict.fromkeys(tokens))  # Preserves order

    return jsonify({
        'tokens': tokens,
        'unique_tokens': unique_tokens
    })

@app.route('/api/stem', methods=['POST'])
def stem():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize the text first
    tokens = tokenizer.tokenize(text)

    # Stem each token and create results
    results = []
    for token in tokens:
        # Only stem alphabetic tokens (skip punctuation)
        if token.isalpha():
            stemmed = stemmer.stem(token)
            results.append(f"{token} → {stemmed}")
        else:
            # Keep punctuation as-is
            results.append(token)

    return jsonify({
        'results': results
    })

@app.route('/api/remove_stopwords_static', methods=['POST'])
def remove_stopwords_static_endpoint():
    """Static stopword removal using predefined stopword list"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Use the static_stopword_removal module
    stopword_file = 'data/stop-words.tr.txt'

    # Load stopwords from the file
    stopwords_list = load_stopwords(stopword_file)

    if not stopwords_list:
        return jsonify({'error': 'Could not load stopwords file'}), 500

    # Tokenize the text first (using the same tokenizer for consistency with other endpoints)
    tokens = tokenizer.tokenize(text)

    # Filter out stopwords and punctuation using static stopwords
    filtered_tokens = []
    removed_tokens = []

    for token in tokens:
        # Check if token is alphabetic and not a stopword
        if token.isalpha():
            if token.lower() not in stopwords_list:
                filtered_tokens.append(token)
            else:
                removed_tokens.append(token)

    # Get unique filtered tokens
    unique_filtered = list(dict.fromkeys(filtered_tokens))

    return jsonify({
        'filtered_tokens': filtered_tokens,
        'unique_filtered_tokens': unique_filtered,
        'removed_tokens': removed_tokens,
        'original_count': len([t for t in tokens if t.isalpha()]),
        'filtered_count': len(filtered_tokens),
        'removed_count': len(removed_tokens)
    })

@app.route('/api/remove_stopwords_tfidf', methods=['POST'])
def remove_stopwords_tfidf():
    """TF-IDF based stopword removal"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize the text first
    tokens = tokenizer.tokenize(text)

    # Filter out stopwords and punctuation using TF-IDF stopwords
    filtered_tokens = []
    removed_tokens = []

    for token in tokens:
        # Check if token is alphabetic and not a stopword
        if token.isalpha():
            if token.lower() not in tfidf_stopwords:
                filtered_tokens.append(token)
            else:
                removed_tokens.append(token)

    # Get unique filtered tokens
    unique_filtered = list(dict.fromkeys(filtered_tokens))

    return jsonify({
        'filtered_tokens': filtered_tokens,
        'unique_filtered_tokens': unique_filtered,
        'removed_tokens': removed_tokens,
        'original_count': len([t for t in tokens if t.isalpha()]),
        'filtered_count': len(filtered_tokens),
        'removed_count': len(removed_tokens)
    })

@app.route('/api/normalize', methods=['POST'])
def normalize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize the text first
    tokens = tokenizer.tokenize(text)

    # Filter tokens with alphabetic characters for normalization
    alphabetic_tokens = [token for token in tokens if any(c.isalpha() for c in token)]

    # Normalize all alphabetic tokens together (enables morpho parser to work)
    normalized_tokens = normalizer.normalize_sequence(alphabetic_tokens)

    # Create results mapping original to normalized
    normalized_map = dict(zip(alphabetic_tokens, normalized_tokens))

    results = []
    for token in tokens:
        if any(c.isalpha() for c in token):
            normalized = normalized_map[token]
            results.append(f"{token} → {normalized}")
        else:
            # Keep punctuation as-is
            results.append(token)

    return jsonify({
        'results': results
    })

if __name__ == '__main__':
    # host='0.0.0.0' allows external connections (not just localhost)
    # Set debug=False in production for security
    app.run(host='0.0.0.0', debug=True, port=5001)
