from classify import ClassificationTrainer
import numpy as np
import os
from chainer import cuda


def readFileIntoArray(filename):
    ret = [];
    with open(filename, 'r') as df:
        for line in df:
            ret.append(line[:-1])
    return ret


class PhraseClassificationTrainer(object):
    """ XXX: comment """
    def __init__(self, data_file, target_file):
        """ XXX: comment """

        """ Read in the raw data & targets """
        raw_data = readFileIntoArray(data_file)
        raw_targets = readFileIntoArray(target_file)

        """ Initialize the underlying vocabulary by assigning vectors to letters """
        base_vocab = "!abcdefghijklmnopqrstuvwqxyz' " # Maybe generate this procedurally
        self.vocab = {}
        for i in range(0, len(base_vocab)):
            k = np.zeros(len(base_vocab))
            k[i] = 1
            self.vocab[base_vocab[i]] = k

        """ Convert the targets to a vector """
        self.targetTranslate = set(raw_targets)
        optDict = dict(zip(self.targetTranslate, range(0, len(self.targetTranslate))))
        self.targets = np.ndarray([len(raw_targets)])
        for i in range(len(raw_targets)):
            self.targets[i] = optDict[raw_targets[i]]
        self.targets = self.targets.astype(np.int32)

        """ Calculate the max vector length """
        # (we won't need this once we fix our underlying chainer model)
        self.max_phrase_len = 0
        for phrase in raw_data:
            if (len(phrase) > self.max_phrase_len):
                self.max_phrase_len = len(phrase)
        self.max_vector_len = self.max_phrase_len * len(base_vocab)

        """ Convert data to vectors """
        k = []
        for phrase in raw_data:
            k.append(self._toVector(phrase))
        self.data = np.asarray(k)

        """ Do not yet initialize the trainer -- we can retrain it later. """
        self.trainer = None

    def _toVector(self, phrase):
        phrase_vec = []
        for char in phrase:
            phrase_vec.extend(self.vocab[char])
        return np.asarray(phrase_vec + [0] * (self.max_vector_len - len(phrase_vec))).astype(np.float32)


    def train(self, net_sizes, epochs, batchsize):
        """ Initialize the base trainer """
        self.trainer = ClassificationTrainer(self.data, self.targets, net_sizes)
        self.trainer.learn(epochs, batchsize)
        return self.trainer.evaluate(batchsize)

    def classify(self, phrase, cut_to_len=True):
      if (len(phrase) > self.max_phrase_len):
          if not cut_to_len:
              raise Exception("Phrase too long.")
          phrase = phrase[0:self.max_phrase_len]
      if (self.trainer == None):
          raise Exception("Must train the classifier at least once before classifying")

      ### XXXX Use target dict to return something reasonable
      return self.trainer.classify(self._toVector(phrase))

    def save(self, directory, label):
       if (self.trainer == None):
           raise Exception("Must train the classifier at least once before saving")

       if not os.path.exists(directory):
           os.makedirs(directory)
       model_file_name = os.path.join(directory, label + ".model")
       state_file_name = os.path.join(directory, label + ".state")
       self.trainer.save(model_file_name, state_file_name)
       ### XXX: save other pre-genned parameters as well
