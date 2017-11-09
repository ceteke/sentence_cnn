from word_tf.models.glove import Glove
from datapy.data.datasets import IMDBDataset

WINDOW_SIZE = 15
VOCAB_SIZE = 50000

dataset = IMDBDataset(corpus_only=True, load_unsup=False)
corpus = dataset.process()

glove = Glove(verbose=1)
glove.fit_corpus(corpus, 15, 50000)
glove.save('imdb')