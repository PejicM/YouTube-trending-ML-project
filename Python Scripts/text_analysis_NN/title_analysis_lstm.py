# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:05:51 2020

@author: Marko Pejić
"""

#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SpatialDropout1D, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

print(tf.__version__)

#%%
# Hyperparameters
vocab_size = 8000
embedding_dim = 64
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

#%%
# Data Loading
videos_df = pd.read_pickle('../US_trending.pkl')
videos_df.head()

#%%
# VIDI KAKVO JOS PRETPROCESIRANJE BI TREBALO URADITI

# Removing stopwords from video titles
def filter_stopwords(text):
    text = text.lower()
    word_list = nltk.word_tokenize(text)
    filtered_words = [word for word in word_list if word not in STOPWORDS]
    return ' '.join(filtered_words)

# Make X (video titles) and y (video categories) vectors
video_titles = videos_df['title'].apply(lambda var: filter_stopwords(var))
#labels = videos_df['category_name']
#labels = videos_df['category_id']
#categories_num = len(set(labels))
category_dummies = pd.get_dummies(videos_df['category_name'])
categories = list(category_dummies.columns)
labels = category_dummies.values
print('Shape of label tensor:', labels.shape)
categories_num = labels.shape[1]

#%%
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(video_titles, labels, test_size=0.2, random_state=33)
print('X_train length: {}'.format(len(X_train)))
print('y_train length: {}'.format(len(y_train)))
print('X_test length: {}'.format(len(X_test)))
print('y_test length: {}'.format(len(y_test)))

#%%
# Find vocab_size (5000) most common words, and replace all other words with oov_tok (<OOV>)
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

# Turn tokens into list of sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
print(train_sequences[10])

lengths = []
for seq in train_sequences:
    lengths.append(len(seq))

print('Maximum length of sequences: {}'.format(max(lengths)))

# Padding - make sequences of same length
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

#%%
# Same job for test (validation) set
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(test_padded.shape)

#%%
### LSTM building
def create_model_RNN():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Dense(embedding_dim, activation='tanh'))
    model.add(Dense(categories_num, activation='softmax'))
    print(model.summary())
    return model

model = create_model_RNN()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
# Training
#history = model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test), verbose=2)

epochs = 50
batch_size = 64

history = model.fit(train_padded, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Visualizations
accr = model.evaluate(test_padded, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#%%
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

#%%
# Predict with new data
new_video = ['WATCH: President Trump and the Coronavirus Task Force holds a news conference at the White House']
seq = tokenizer.texts_to_sequences(new_video)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)

print(pred, categories[np.argmax(pred)])
