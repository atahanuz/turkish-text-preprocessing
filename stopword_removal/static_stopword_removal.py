from stopword_removal.tfidf_stopword_identification import tokenize

stopword_file = "../data/stop-words.tr.txt"
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
    text=input()
    stopword_cleaned = remove_stopwords(text, stopword_file)

 
