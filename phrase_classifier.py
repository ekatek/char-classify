from base_classifier import ClassificationTrainer, Classifier
import numpy as np
import os
import json

def readFileIntoArray(filename):
    ret = [];
    with open(filename, 'r') as df:
        for line in df:
            ret.append(line[:-1])
    return ret


def stringToVector(phrase, vocab, maxlen):
    phrase_vec = []
    for char in phrase:
        phrase_vec.extend(vocab[char])
    return np.asarray(phrase_vec + [0] * (maxlen - len(phrase_vec))).astype(np.float32)


def generateVocabVectors(base_vocab):
    vec_vocab = {}
    for i in range(0, len(base_vocab)):
        k = np.zeros(len(base_vocab))
        k[i] = 1
        vec_vocab[base_vocab[i]] = k
    return vec_vocab


class PhraseClassificationTrainer(object):
    """Given a file of phrases and a file of targets, train a phrase classifier to
       recognize the phrases."""
    def __init__(self, data_file, target_file):
        """ Read in the raw data & targets """
        raw_data = readFileIntoArray(data_file)
        raw_targets = readFileIntoArray(target_file)

        """ Initialize the underlying vocabulary by assigning vectors to letters """
        base_vocab = "!abcdefghijklmnopqrstuvwqxyz' " # Maybe generate this procedurally
        self.vocab = generateVocabVectors(base_vocab)

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
            k.append(stringToVector(phrase, self.vocab, self.max_vector_len))
        self.data = np.asarray(k)

        """ Do not yet initialize the trainer -- we can retrain it later. """
        self.trainer = None

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
      return self.trainer.classify(stringToVector(phrase, self.vocab, self.max_vector_len))

    def save(self, directory, label):
       if (self.trainer == None):
           raise Exception("Must train the classifier at least once before saving")

       if not os.path.exists(directory):
           os.makedirs(directory)
       model_filename = os.path.join(directory, label + ".model")
       state_filename = os.path.join(directory, label + ".state")
       self.trainer.save(model_filename, state_filename)

       save_file = {}
       save_file["max_vector_len"] = self.max_vector_len
       save_file["max_phrase_len"] = self.max_phrase_len
       save_file["vocab"] = list(base_vocab)
       save_file["sizes"] = self.trainer.sizes
       other_filename = os.path.join(directory, label + ".other")
       with open(other_filename, 'w') as v:
           v.write(json.dumps(save_file))


class PhraseClassifier(object):
    """ Classify a phrase based on a pre-existing model """
    def __init__(self, directory, label):
       other_filename = os.path.join(directory, label + ".other")
       with open(other_filename, 'r') as v:
           all_vars = json.load(v)
           self.vocab =  generateVocabVectors(all_vars["vocab"])
           self.max_vector_len = all_vars["max_vector_len"]
           self.max_phrase_len = all_vars["max_phrase_len"]
           self.net_sizes = all_vars["sizes"]

       model_filename = os.path.join(directory, label + ".model")
       state_filename = os.path.join(directory, label + ".state")
       self.classifier = Classifier(self.net_sizes, model_filename, state_filename)


    def classify(self, phrase, cut_to_len=True):
      """ Classify a phrase based on the loaded model. If cut_to_len is True, cut to
          desired length."""
      if (len(phrase) > self.max_phrase_len):
          if not cut_to_len:
              raise Exception("Phrase too long.")
          phrase = phrase[0:self.max_phrase_len]

      return self.classifier.classify(stringToVector(phrase, self.vocab, self.max_vector_len))
