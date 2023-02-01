import numpy as np
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import Dense
import matplotlib.pyplot as plt

import preprocessing
# Defining the window for context
window = 2

# Creating a placeholder for the scanning of the word list
word_lists = []
all_text = []

# open file and add text to the list
with open('sentences') as f:
    for text in f:
        text = preprocessing.clean_text(text)
        all_text.append(text)

# Context dictionary
for sentence in all_text:
    sentence = sentence.split()
    for i, word in enumerate(sentence):
        for next in range(window):
            if i+1+next < len(sentence):
                word_lists.append((word, sentence[i+1+next]))
            if i-1-next >= 0:
                word_lists.append((word, sentence[i-1-next]))

def unique_word(text:list):
    list_text = []
    for line in text:
        for word in line.split():
            list_text.append(word)
    words = list(set(list_text))
    words.sort()
    unique_word = {}
    for i, word in enumerate(words):
        unique_word.update({word: i})
    return unique_word

# number of features
unique_word_dict = unique_word(all_text)
n_words = len(unique_word_dict)
words = list(unique_word_dict.keys())

# Creating the X and Y matrices using one hot encoding
X = []
Y = []
for i, word_list in enumerate(word_lists):
    main_word_index = unique_word_dict.get(word_list[0])
    context_word_index = unique_word_dict.get(word_list[1])

    # Creating the placeholders
    X_row = np.zeros(n_words)
    Y_row = np.zeros(n_words)

    # One hot encoding the main and the context word
    X_row[main_word_index] = 1
    Y_row[context_word_index] = 1

    # Main matrices
    X.append(X_row)
    Y.append(Y_row)

# Converting the matrices into an array
X = np.asarray(X)
Y = np.asarray(Y)

# Deep learning

# The size of the embedding = the hidden layer dimension
embed_size = 2

# Defining the size of the embedding
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Optimizing the network weights
model.fit(
    x=X,
    y=Y,
    batch_size=256,
    epochs=1000
    )
# Obtaining the weights from the neural network.
# These are the so called word embeddings

# The input layer
weights = model.get_weights()[0]

# Creating a dictionary to store the embeddings in. The key is a unique word and the value is the numeric vector
embedding_dict = {}
for word in words:
    embedding_dict.update({word: weights[unique_word_dict.get(word)]})

print(embedding_dict)
plt.figure(figsize=(10, 10))
for word in list(unique_word_dict.keys()):
    coord = embedding_dict.get(word)
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))
plt.show()
plt.savefig('Visualization of the embeddings.png')