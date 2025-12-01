import nltk
from nltk.corpus import stopwords
from static_stopword_removal import load_stopwords, remove_stopwords, tokenize
from collections import Counter
import math

# Download Turkish stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stopword_file = "../data/turkce-stop-words.txt"
tfidf_stopword_file = "../data/tf-idf_stopwords.txt"
corpus_path = "../data/tr_boun-ud/tr_boun-ud-train_text.txt"

def remove_nltk_stopwords(document):
    nltk_stopwords = set(stopwords.words('turkish'))
    tokens = tokenize(document)
    filtered_tokens = [token for token in tokens if token not in nltk_stopwords]
    return ' '.join(filtered_tokens)

#Calculate unique vocabulary size
def calculate_vocabulary_size(documents):
    all_tokens = []
    for doc in documents:
        all_tokens.extend(tokenize(doc))
    return len(set(all_tokens))

#Calculate vocabulary reduction percentage
def calculate_vocabulary_reduction(original_vocab, filtered_vocab):
    reduction = ((original_vocab - filtered_vocab) / original_vocab) * 100
    return reduction
#Calculate average document length and total tokens
def calculate_document_length_stats(documents):
    lengths = [len(tokenize(doc)) for doc in documents]
    total_tokens = sum(lengths)
    avg_length = total_tokens / len(documents) if documents else 0
    return avg_length, total_tokens

#Calculate token reduction percentage
def calculate_token_reduction(original_tokens, filtered_tokens):
    reduction = ((original_tokens - filtered_tokens) / original_tokens) * 100
    return reduction

#Calculate what percentage of corpus tokens are stopwords
def calculate_stopword_coverage(documents, stopword_set):
    total_tokens = 0
    stopword_tokens = 0
    
    for doc in documents:
        tokens = tokenize(doc)
        total_tokens += len(tokens)
        stopword_tokens += sum(1 for token in tokens if token in stopword_set)
    
    coverage = (stopword_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    return coverage, stopword_tokens, total_tokens


def evaluation():
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        test_corpus = [line.strip() for line in f if line.strip()]
    
    print(f"Corpus size: {len(test_corpus)} documents\n")
    
    # Load stopword lists
    static_stopwords = set(load_stopwords(stopword_file))
    tfidf_stopwords = set(load_stopwords(tfidf_stopword_file))
    nltk_stopwords = set(stopwords.words('turkish'))
    
    # Apply stopword removal
    static_cleaned = [remove_stopwords(doc, stopword_file) for doc in test_corpus]
    tfidf_cleaned = [remove_stopwords(doc, tfidf_stopword_file) for doc in test_corpus]
    nltk_cleaned = [remove_nltk_stopwords(doc) for doc in test_corpus]
    
    print("EVALUATION RESULTS" + "-" * 60)
    
    #Vocabulary Reduction Analysis
    print("VOCABULARY REDUCTION" + "-" * 60)
    original_vocab = calculate_vocabulary_size(test_corpus)
    static_vocab = calculate_vocabulary_size(static_cleaned)
    tfidf_vocab = calculate_vocabulary_size(tfidf_cleaned)
    nltk_vocab = calculate_vocabulary_size(nltk_cleaned)
    print(f"Original vocabulary size: {original_vocab:,}")
    print(f"Static stopwords vocabulary: {static_vocab:,} (reduction: {calculate_vocabulary_reduction(original_vocab, static_vocab):.2f}%)")
    print(f"TF-IDF stopwords vocabulary: {tfidf_vocab:,} (reduction: {calculate_vocabulary_reduction(original_vocab, tfidf_vocab):.2f}%)")
    print(f"NLTK stopwords vocabulary: {nltk_vocab:,} (reduction: {calculate_vocabulary_reduction(original_vocab, nltk_vocab):.2f}%)")
    
    # Document Length Reduction Analysis
    print("DOCUMENT LENGTH REDUCTION" + "-" * 60)
    orig_avg, orig_total = calculate_document_length_stats(test_corpus)
    static_avg, static_total = calculate_document_length_stats(static_cleaned)
    tfidf_avg, tfidf_total = calculate_document_length_stats(tfidf_cleaned)
    nltk_avg, nltk_total = calculate_document_length_stats(nltk_cleaned)
    print(f"Original - Avg tokens/doc: {orig_avg:.2f}, Total tokens: {orig_total:,}")
    print(f"Static - Avg tokens/doc: {static_avg:.2f}, Total tokens: {static_total:,} (reduction: {calculate_token_reduction(orig_total, static_total):.2f}%)")
    print(f"TF-IDF - Avg tokens/doc: {tfidf_avg:.2f}, Total tokens: {tfidf_total:,} (reduction: {calculate_token_reduction(orig_total, tfidf_total):.2f}%)")
    print(f"NLTK - Avg tokens/doc: {nltk_avg:.2f}, Total tokens: {nltk_total:,} (reduction: {calculate_token_reduction(orig_total, nltk_total):.2f}%)")
    
    # Stopword Coverage Analysis
    print("STOPWORD COVERAGE" + "-" * 60) 
    static_cov, static_sw_tokens, _ = calculate_stopword_coverage(test_corpus, static_stopwords)
    tfidf_cov, tfidf_sw_tokens, _ = calculate_stopword_coverage(test_corpus, tfidf_stopwords)
    nltk_cov, nltk_sw_tokens, _ = calculate_stopword_coverage(test_corpus, nltk_stopwords)
    print(f"Static stopwords - Coverage: {static_cov:.2f}% ({static_sw_tokens:,} stopword tokens)")
    print(f"TF-IDF stopwords - Coverage: {tfidf_cov:.2f}% ({tfidf_sw_tokens:,} stopword tokens)")
    print(f"NLTK stopwords - Coverage: {nltk_cov:.2f}% ({nltk_sw_tokens:,} stopword tokens)")
    

if __name__ == "__main__":
    evaluation()