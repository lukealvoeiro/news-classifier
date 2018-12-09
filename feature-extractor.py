import json
import re
import collections
import tensorflow as tf
import numpy as np
import string
import pandas as pd
from math import floor

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import pickle

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

NUM_WORDS = 10000
NUM_CATEGORIES = 31
TRAIN_SET = 0.7
"""
TODO: 
- Deal with two inputs (headline and description)
"""


def main():
    # Validation Accuracies: Headlines - 0.9703, Headlines+Desc - 0.9700, Desc - 0.9681
    # 0.9707 - Val split 0.4, 150 neurons, 0.2 dropout, lr=3
    
    """
    # RUN ONCE - creates files so they don't have to be recreated every run
    
    #create dataset in pandas dataframe
    df = datasetBuilder()
    with open('df.pd', 'wb') as data:
        pickle.dump(df, data)

    # create tokenized word sequence
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(df["short_description"].tolist() + df["headline"].tolist())
    data_headline = pad_sequences(tokenizer.texts_to_sequences(df['headline']), maxlen=19) # max sequence size of 19
    data_description = pad_sequences(tokenizer.texts_to_sequences(df['short_description']), maxlen=123) # max sequence size of 123
    data_all = pad_sequences(tokenizer.texts_to_sequences(df['short_description'] + df['headline']), maxlen=142)
    embeddings_matrix = createEmbeddingMatrix(tokenizer, getGloveEmbeddings())

    # save large objects as files 
    with open('headlines', 'wb') as headlines:
        pickle.dump(data_headline, headlines)
    with open('descriptions', 'wb') as descriptions:
        pickle.dump(data_description, descriptions)
    with open('emb_matrix', 'wb') as emb_matrix:
        pickle.dump(embeddings_matrix, emb_matrix)
    with open('data_all', 'wb') as all:
        pickle.dump(data_all, all)

    # count blank headlines - 37
    count=0
    for i in df['headline']:
        if i == "":
            count += 1
    print('count=', count)
    """
    # load saved files
    with open('df.pd', 'rb') as data:
        df = pickle.load(data)
    with open('data_all', 'rb') as data_all:
        data = pickle.load(data_all)
    with open('headlines', 'rb') as headlines:
        data_headline = pickle.load(headlines)
    with open('descriptions', 'rb') as descriptions:
        data_description = pickle.load(descriptions)
    with open('emb_matrix', 'rb') as emb_matrix:
        embeddings_matrix = pickle.load(emb_matrix)
    
    ## Get output array
    cats = np.array(df['category'])
    y = np.zeros((len(cats), NUM_CATEGORIES))
    for i, ex in enumerate(cats):
        y[i][ex] = 1
    
    # split off test set
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.15)  
    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=NUM_WORDS, output_dim=100, input_length=142, weights=[embeddings_matrix], trainable=False))
    
    model.add(Conv(64, 3, input_shape=(142, 100), activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.4))
    
    model.add(LSTM(100))    
    
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    # sgd = SGD(lr=3)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs = 3, batch_size=10, shuffle=True)
    #model.summary()

    print(model.evaluate(X_test, y_test))



    #     # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
        

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

    return df

def getGloveEmbeddings():
    embeddings_index = dict()
    with open('glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def createEmbeddingMatrix(tokenizer, embeddings_index):
    embedding_matrix = np.zeros((NUM_WORDS, 100))
    for word, index in tokenizer.word_index.items():
        if index > NUM_WORDS - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix

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
