from word_tf.models.glove import Glove
from datapy.data.datasets import IMDBDataset
from sentence_cnn import SentenceCNN
import pickle
import tensorflow as tf
import numpy as np

WINDOW_SIZE = 15
EMBEDDING_SIZE = 300
VOCAB_SIZE = 50000
LEARNING_RATE = 0.05
EPOCHS = 30
ALPHA = 0.75
X_MAX = 100
BATCH_SIZE = 64
MAX_LENGTH = 60
INFO_STEP = 100

#dataset = IMDBDataset(corpus_only=True, load_unsup=False)
#corpus = dataset.process()

#gloveO = Glove(verbose=1)
#glove.fit_corpus(corpus, 15, 50000)
#glove.save('imdb')

sess = tf.Session()
print("Loading vocabulary")
gloveO = pickle.load(open('glove-imdb.pkl', 'rb'))
del gloveO.cooccurence_dict
#gloveO.train_tf(sess, EMBEDDING_SIZE, LEARNING_RATE, EPOCHS, ALPHA, X_MAX, batch_size=512, info_step=10000)
print("Loading embedding")
embedding_matrix = pickle.load(open('glove-embedding-imdb.pkl', 'rb'))
first_row = np.array([[0.0]*300])
embedding_matrix = np.concatenate((first_row, embedding_matrix))
# print(gloveO.most_similar('poop'))
# gloveO.save_embedding('imdb')

first_tok = gloveO.id2tok[0]
gloveO.id2tok[len(gloveO.id2tok)] = first_tok
gloveO.id2tok[0] = 'PAD'
gloveO.tok2id[first_tok] = len(gloveO.id2tok)-1
gloveO.tok2id['PAD'] = 0
dataset = IMDBDataset(token2idx=gloveO.tok2id, idx2token=gloveO.id2tok, load_unsup=False)
del gloveO

dataset.process()

sentence_cnn = SentenceCNN(sess, 2, LEARNING_RATE, BATCH_SIZE, [3,4,5], 100, embedding_matrix, MAX_LENGTH, 0.5)

batch_loss = 0

for e in range(EPOCHS):
    X, _, y = dataset.get_batches_sequence(BATCH_SIZE, MAX_LENGTH, pad_token=0)
    num_batches = len(X)
    print("Epoch {}/{}".format(e+1, EPOCHS))
    epoch_loss = 0
    for b in range(num_batches):
        x_b = X[b]
        y_b = y[b]
        loss, accuracy = sentence_cnn.train(x_b, y_b)
        batch_loss += loss
        epoch_loss += loss

        if (b+1) % INFO_STEP == 0:
            print("\tBatch {}/{}:".format(b+1, num_batches))
            print("\t\tLoss: {} Accuracy {}".format(batch_loss/INFO_STEP, accuracy))
            batch_loss = 0
            batch_accuracy = 0

    print("Loss {} Accuracy {}".format(epoch_loss/num_batches, accuracy))


X_test, _, y_test = dataset.get_batches_sequence(BATCH_SIZE, MAX_LENGTH, pad_token=0, train=False)
num_test_batches = len(X_test)
test_loss = 0
for b in range(num_test_batches):
    x_b = X_test[b]
    y_b = y_test[b]
    loss, accuracy = sentence_cnn.test(x_b, y_b)
    test_loss += loss

print("Test loss {} Accuracy {}".format(test_loss / num_test_batches, accuracy))