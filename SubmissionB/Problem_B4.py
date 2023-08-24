# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import numpy as np
# from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import csv


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    # def remove_stopwords(sentence):
    #     # List of stopwords
    #     stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    #                  "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
    #                  "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    #                  "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    #                  "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    #                  "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of",
    #                  "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    #                  "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
    #                  "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    #                  "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    #                  "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
    #                  "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    #                  "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    #                  "yourself", "yourselves"]
    #
    #     # Sentence converted to lowercase-only
    #     sentence = sentence.lower()
    #
    #     words = sentence.split()
    #     no_words = [w for w in words if w not in stopwords]
    #     sentence = " ".join(no_words)
    #
    #     return sentence

    sentences = []
    labels = []
    for _, row in bbc.iterrows():
        labels.append(row[0])
        sentences.append(row[1])

    train_size = int(len(sentences) * training_portion)
    training_sentences = sentences[:train_size]
    training_labels = labels[:train_size]
    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    # training_sentences, validation_sentences = #YOUR CODE HERE
    # training_labels, validation_labels = #YOUR CODE HERE

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    def seq_pad_trunc(sen, tokenizer, padding, truncating, maxlen):
        sequences = tokenizer.texts_to_sequences(sen)
        pad_trunc_seq = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
        return pad_trunc_seq

    train_pad_seq_trunc = seq_pad_trunc(training_sentences, tokenizer, padding_type, trunc_type, max_length)
    val_pad_seq_trunc = seq_pad_trunc(validation_sentences, tokenizer, padding_type, trunc_type, max_length)

    # You can also use Tokenizer to encode your label.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    train_label_seq = np.array(label_tokenizer.texts_to_sequences(training_labels))
    val_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(train_pad_seq_trunc, train_label_seq, epochs=100, validation_data=(val_pad_seq_trunc, val_label_seq))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
