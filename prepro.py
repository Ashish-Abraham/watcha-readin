#preprocessing
from sklearn.preprocessing import OrdinalEncoder
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from nltk.stem.porter import PorterStemmer
import re

#stem words
def stemm(data):
    ps=PorterStemmer()
    corpus=[]
    review=re.sub('[^a-zA-Z]',' ',data)
    review=review.lower()
    review=review.split()
    #remove html tag by removing <br> also
    review=[ps.stem(word) for word in review if not word in stopwords.words('english') and not word in ['br']]
    review=' '.join(review)
    corpus.append(review)
    return corpus

#one hot encoding and padding
def preprocess(data):
    corpus=stemm(data)
    onehot_corpus=[one_hot(words,10000) for words in corpus]
    sent_length = 2470
    padded_corpus=pad_sequences(onehot_corpus,padding='pre',maxlen=sent_length)
    return padded_corpus

