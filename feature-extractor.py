import json
import re
import collections
import tensorflow as tf
import numpy as np
import string
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

NUM_WORDS = 10000

"""
TODO: 
- Padding on input texts (find max length of input string (in terms of # of words), pad by this.)
- Deal with two inputs (headline and description)

"""


def main():
    datasetBuilder()
    # news_types = readNewsTypes()
    # vocab, inverse_vocab = getVocab()
    # print(vocab)
    # base_dataset = buildBaseDataset(vocab, news_types)
    # print(base_dataset[0])
    # getGloveEmbeddings()

def readNewsTypes(categories):
    news_types = set(categories)
    return dict((news_type, index) for index, news_type in dict(enumerate(news_types)).items())

def getVocab():
    translation_table = str.maketrans({key: None for key in string.punctuation})
    words = list()
    vocab = dict()
    with open('News_Category_Dataset.json', 'r') as file:
        for line in file:
            line_as_json = json.loads(line)
            description = line_as_json["short_description"]
            headline = line_as_json["headline"]
            words.extend(cleanText(description, translation_table))
            words.extend(cleanText(headline, translation_table))

    count = [['BLK', 0],['UNK', 1]]  # indicating UNKNOWN
    count.extend(collections.Counter(words).most_common(NUM_WORDS - 2))
    for word, _ in count:
        vocab[word] = len(vocab)
    inverse_vocab = dict(zip(vocab.values(), vocab.keys()))
    return vocab, inverse_vocab
        
def encodeText(vocab, words):
    res = []
    for word in words:
        if(word in vocab.keys()): res.append(vocab[word])
        else: res.append(vocab['UNK'])
    return res

def decodeText(inverse_vocab, integers):
    res = []
    for i in integers:
        res.append(inverse_vocab[i])
    return res

def datasetBuilder():
    translation_table = str.maketrans({key: None for key in string.punctuation})
    df = pd.read_json('News_Category_Dataset.json', lines=True)
    df = df.drop(['date','link','authors'], axis=1)
    df = df.dropna()
    categories = readNewsTypes(df["category"].tolist())

    df["category"] = df["category"].map(lambda x: categories[x])
    df["short_description"] = df["short_description"].map(lambda x: cleanText(x, translation_table))
    df["headline"] = df["headline"].map(lambda x: cleanText(x, translation_table))

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(df["short_description"].tolist() + df["headline"].tolist())
    sequences_headline = tokenizer.texts_to_sequences(df['headline'])
    sequences_description = tokenizer.texts_to_sequences(df['short_description'])
    

# def buildBaseDataset(vocab, news_types):
#     translation_table = str.maketrans({key: None for key in string.punctuation})

#     dataset = []
#     with open('News_Category_Dataset.json', 'r') as file:
#         for line in file:
#             line = json.loads(line)
#             description = line["short_description"]
#             headline = line["headline"]
#             category = line["category"]
#             entry = [cleanText(description, translation_table), cleanText(headline, translation_table), news_types[category]]
#             dataset.append(entry)
#     return dataset

def getGloveEmbeddings():
    embeddings_index = dict()
    with open('glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def cleanText(text, table, stem=False):
    text = text.translate(table)
    text = text.lower().split()
    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    # Clean text
    if(stem):
        stemmer = SnowballStemmer('english')
        text = [stemmer.stem(word) for word in text]
    return " ".join(text)

main()