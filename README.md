# kalenjin-word-corpus
import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk

try:
    nltk.download('punkt')
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

kalenjin_sentences = [
    "Kiptaiyat ak muren eng kisumet.",
    "Koee inendet ab kasit ne kikoomi.",
    "Kimnai lagok che mi konom kongoi.",
    "Kapkutuny nebo boiyot komwa.",
    "Amun kalyet ne bo kotik kiptendeny."
]

def preprocess_text(sentences):
    """
    Clean and tokenize sentences.
    """
    clean_sentences = []
    for sentence in sentences:
        # Remove punctuation and convert to lowercase
        sentence = re.sub(r"[^\w\s]", "", sentence).lower()
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        clean_sentences.append(tokens)
    return clean_sentences

print("\nPreprocessing sentences...")
try:
    preprocessed_corpus = preprocess_text(kalenjin_sentences)
    print("Preprocessed Corpus:", preprocessed_corpus)
except Exception as e:
    print("Error during preprocessing:", e)

if preprocessed_corpus:
    print("\nTraining Word2Vec model...")
    try:
        model = Word2Vec(
            sentences=preprocessed_corpus,  
            vector_size=100,                
            window=5,                       
            min_count=1,                   
            workers=4                       
        )
        model.save("kalenjin_word2vec.model")
        print("Word2Vec model saved as 'kalenjin_word2vec.model'.")
    except Exception as e:
        print("Error during Word2Vec training:", e)
else:
    print("No preprocessed data available for training Word2Vec.")

try:
    word_vector = model.wv['kiptaiyat'] 
    print("\nVector for 'kiptaiyat':\n", word_vector)
except KeyError:
    print("Word 'kiptaiyat' not found in the vocabulary.")
except NameError:
    print("Model is not defined. Ensure Word2Vec training was successful.")

try:
    similar_words = model.wv.most_similar('kiptaiyat')  
    print("\nWords similar to 'kiptaiyat':", similar_words)
except KeyError:
    print("Word 'kiptaiyat' not found in the vocabulary.")
except NameError:
    print("Model is not defined. Ensure Word2Vec training was successful.")
