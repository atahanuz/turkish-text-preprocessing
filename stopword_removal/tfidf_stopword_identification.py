import math
import re
from collections import defaultdict

def save_stopwords(stopwords, filename):
    sorted_stopwords = sorted(stopwords)
    with open(filename, 'w', encoding='utf-8') as f:
        for word in sorted_stopwords:
            f.write(word + '\n')

def tokenize(text):
    text = text.replace('İ', 'i') 
    text = text.replace('I', 'ı')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = re.findall(r'\b[a-zçğıöşü]+\b', text)
    return tokens

def calculate_term_frequency(document):
    words = tokenize(document)
    word_count = len(words)
    
    if word_count == 0:
        return {}
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] = tf[word] / word_count
    return tf

def calculate_document_frequency(corpus):
    df = defaultdict(int)
    for document in corpus:
        words = set(tokenize(document))
        for word in words:
            df[word] += 1    
    return df

def calculate_idf(df, total_documents):
    idf = {}
    for word, doc_count in df.items():
        idf[word] = math.log(total_documents / doc_count)
    return idf

def calculate_tfidf(corpus):
    total_documents = len(corpus)
    df = calculate_document_frequency(corpus)
    idf = calculate_idf(df, total_documents)
    tfidf_corpus = []
    for document in corpus:
        tf = calculate_term_frequency(document)
        tfidf = {}
        for word, tf_score in tf.items():
            tfidf[word] = tf_score * idf[word]
        tfidf_corpus.append(tfidf)
    return tfidf_corpus, idf, df

def identify_stop_words_by_idf(idf, threshold=0.5):
    stop_words = []
    for word, idf_score in idf.items():
        if idf_score < threshold:
            stop_words.append((word, idf_score))
    stop_words.sort(key=lambda x: x[1])
    stop_words = [word for word, score in stop_words]
    return stop_words

def identify_stop_words_by_df(df, total_documents, threshold=0.5):
    stop_words = []
    for word, doc_count in df.items():
        if (doc_count / total_documents) > threshold:
            stop_words.append((word, (doc_count / total_documents)))
    stop_words.sort(key=lambda x: x[1])
    stop_words = [word for word, score in stop_words]
    return stop_words

def calculate_average_tf(corpus):
    term_tf_sum = defaultdict(float)
    term_count = defaultdict(int)
    for document in corpus:
        tf = calculate_term_frequency(document)
        for word, score in tf.items():
            term_tf_sum[word] += score
            term_count[word] += 1
    avg_tf = {}
    for word in term_tf_sum:
        avg_tf[word] = term_tf_sum[word] / term_count[word]
    return avg_tf

def identify_stop_words_by_tf(avg_tf, threshold=0.01):
    stop_words = []
    for word, avg_score in avg_tf.items():
        if avg_score < threshold:
            stop_words.append((word, avg_score))
    stop_words.sort(key=lambda x: x[1], reverse=True)
    stop_words = [word for word, score in stop_words]
    return stop_words

def calculate_average_tfidf(tfidf_corpus):
    term_tfidf_sum = defaultdict(float)
    term_count = defaultdict(int)
    for tfidf_doc in tfidf_corpus:
        for word, score in tfidf_doc.items():
            term_tfidf_sum[word] += score
            term_count[word] += 1
    avg_tfidf = {}
    for word in term_tfidf_sum:
        avg_tfidf[word] = term_tfidf_sum[word] / term_count[word]
    return avg_tfidf

def identify_stop_words_by_avg_tfidf(avg_tfidf, threshold=0.1):
    stop_words = []
    for word, avg_score in avg_tfidf.items():
        if avg_score < threshold:
            stop_words.append((word, avg_score))
    stop_words.sort(key=lambda x: x[1])
    stop_words = [word for word, score in stop_words]
    return stop_words

def print_statistics(idf, df, avg_tfidf, total_documents, top_n=10):

    print(f"Total unique words: {len(idf)}")
    print(f"Total documents: {total_documents}\n")
    
    if len(idf) == 0:
        print("No words found in corpus after tokenization!")
        return

    sorted_by_idf = sorted(idf.items(), key=lambda x: x[1])
    display_n = min(top_n, len(sorted_by_idf))
    
    print(f"Top {display_n} words with LOWEST IDF (most likely stop words):")
    for word, score in sorted_by_idf[:display_n]:
        doc_freq = df[word]
        doc_percentage = (doc_freq / total_documents) * 100
        print(f"  {word}: IDF={score:.4f}, appears in {doc_freq}/{total_documents} docs ({doc_percentage:.1f}%)")
    
    print(f"\nTop {display_n} words with HIGHEST IDF (most discriminative):")
    for word, score in sorted_by_idf[-display_n:]:
        doc_freq = df[word]
        print(f"  {word}: IDF={score:.4f}, appears in {doc_freq}/{total_documents} docs")


# Example usage
if __name__ == "__main__":

    save_filename = "tf-idf_stopwords.txt"
    with open('../data/tr_boun-ud-train_text.txt', 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f]

    #corpus = corpus[:20]
    print("=" * 60 + "TF-IDF BASED STOP WORD EXTRACTION" + "=" * 60 + "\n")
    
    # Calculate TF-IDF
    tfidf_corpus, idf, df = calculate_tfidf(corpus)
    avg_tfidf = calculate_average_tfidf(tfidf_corpus)
    
    # Print statistics
    print_statistics(idf, df, avg_tfidf, len(corpus), top_n=20)
    # Ortalama TF hesapla
    avg_tf = calculate_average_tf(corpus)

    # Stopword'leri belirle (threshold=0.01 veya istediğiniz değer)
    stop_words_tf = identify_stop_words_by_tf(avg_tf, threshold=0.01)
    print(f"Method 1 - Stop words by TF")
    print(f" {len(stop_words_tf)} words: {stop_words_tf}\n")
    #idf_threshold = 5.65 #the most optimal one   
    idf_threshold = 5.65 #the most optimal one for all_text
    df_threshold = 0.15           
    avg_tfidf_threshold = 0.20 

    print("=" * 60 + "STOP WORD IDENTIFICATION METHODS" + "=" * 60 + "\n")

    stop_words_idf = identify_stop_words_by_idf(idf, threshold=idf_threshold)
    print(f"Method 1 - Stop words by IDF < {idf_threshold:.4f}:")
    print(f" {len(stop_words_idf)} words: {stop_words_idf}\n")
    save_stopwords(stop_words_idf, save_filename)

    stop_words_df = identify_stop_words_by_df(df, len(corpus), threshold=df_threshold)
    print(f"Method 2 - Stop words appearing in >{df_threshold*100:.0f}% of documents:")
    print(f" {len(stop_words_df)} words: \n")

    """
    stop_words_avg = identify_stop_words_by_avg_tfidf(avg_tfidf, threshold=avg_tfidf_threshold)
    print(f"Method 3 - Stop words by average TF-IDF < {avg_tfidf_threshold:.4f}:")
    print(f" {len(stop_words_avg)} words: {stop_words_avg} \n")
    
    # Combined approach
    combined_stop_words = set(stop_words_idf) & set(stop_words_df)
    print(f"Combined - Words identified by both IDF and DF methods:")
    print(f" {len(combined_stop_words)} words: ")
    """