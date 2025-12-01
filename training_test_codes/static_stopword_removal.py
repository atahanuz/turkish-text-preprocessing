from tfidf_stopword_identification import tokenize

stopword_file = "../data/turkce-stop-words.txt"
tfidf_stopword_file = "../data/tf-idf_stopwords.txt"
tfidf_stopword_all_file = "../data/tf-idf_stopwords_all.txt"
corpus_path = "../data/tr_boun-ud/tr_boun-ud-train_text.txt"

#read stopword list from static zemberek stop-words.tr.txt file 
#stop-words.tr.txt source is here: https://github.com/ahmetaa/zemberek-nlp/blob/master/experiment/src/main/resources/stop-words.tr.txt
#turkce-stop-words.txt source is here: https://github.com/ahmetax/trstop/blob/master/dosyalar/turkce-stop-words
def load_stopwords(file="turkce-stop-words.txt"):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            stopwords = [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return set()
    return stopwords

def remove_stopwords(document, file):
    stopwords = load_stopwords(file)
    
    if not stopwords:
        return document
    
    tokens = tokenize(document)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(filtered_tokens)

# Example usage
if __name__ == "__main__":
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f]

    #corpus = corpus[:5]
    stopword_cleaned = [remove_stopwords(document, stopword_file) for document in corpus]
    tfidf_stopword_cleaned = [remove_stopwords(document, tfidf_stopword_file) for document in corpus]

    diff = set(stopword_cleaned) - set(tfidf_stopword_cleaned)
    print(len(diff))
    #print(stopword_cleaned)
    #print(tfidf_stopword_cleaned)

    stopwords = load_stopwords(stopword_file)
    tfidf_stopwords = load_stopwords(tfidf_stopword_file)

    diff = set(stopwords) - set(tfidf_stopwords)
    print(len(diff))
