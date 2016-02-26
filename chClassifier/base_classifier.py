""" Classify the output based on the choice neuron in the last layer with the highest probability."""
import numpy as np

from chainer import cuda, Variable, Chain, optimizers, serializers
import chainer.links as L
import chainer.functions as F


class BaseNetwork(Chain):
    """
      BaseNetwork is the underlying NN.  Takes in the
      following parameters:
       - inputSize: Total length of an input vector
       - hidden_layers: An array of ints, corresponding to number of neurons in each layer.
       - choices: Total number of possible choices.
    """
    def __init__(self, sizes):
        if len(sizes) < 2:
            raise Exception("Net must have input and output layers")
        layers = [];
        for i in range(0, len(sizes)-1):
            layers.append(L.Linear(sizes[i], sizes[i+1]))
        super(BaseNetwork, self).__init__(
            l0 = layers[0],
            l1 = layers[1]
        );
        for i in range(2, len(layers)):
            super(BaseNetwork, self).add_link(str(i), layers[i])

    """Call the network -- given x (an input vector), call each layer and return
     the final result."""
    def __call__(self, x):
        layer_result = x
        for layer in self.children():
            layer_result = F.relu(layer(layer_result))
        return layer_result

class ClassificationTrainer(object):
    """Train a classifier on some labeled data.
    """
    def __init__(self, data, target, hidden_layers):
        """ Must submit either a net configuration, or something to load from """
        if hidden_layers == [] and model_filename == "":
            raise Exception("Must provide a net configuration or a file to load from")

        """ Divide the data into training and test """
        self.trainsize = int(len(data) * 5 / 6)
        self.testsize = len(data) - self.trainsize
        self.x_train, self.x_test = np.split(data, [self.trainsize])
        self.y_train, self.y_test = np.split(target, [self.trainsize])

        """ Create the underlying neural network model """
        self.sizes = [len(data[0])]
        self.sizes.extend(hidden_layers)
        self.sizes.append(len(set(target)))
        self.model = L.Classifier(BaseNetwork(self.sizes))

        """ Create the underlying optimizer """
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def learn(self, numEpochs, batchsize):
        """Train the classifier for a given number of epochs, with a given batchsize"""
        for epoch in range(numEpochs):
            print('epoch %d' % epoch)
            indexes = np.random.permutation(self.trainsize)
            for i in range(0, self.trainsize, batchsize):
                x = Variable(self.x_train[indexes[i: i + batchsize]])
                t = Variable(self.y_train[indexes[i: i + batchsize]])
                self.optimizer.update(self.model, x, t)

    def evaluate(self, batchsize):
        """Evaluate how well the classifier is doing. Return mean loss and mean accuracy"""
        sum_loss, sum_accuracy = 0, 0
        for i in range(0, self.testsize, batchsize):
            x = Variable(self.x_test[i: i + batchsize])
            y = Variable(self.y_test[i: i + batchsize])
            loss = self.model(x, y)
            sum_loss += loss.data * batchsize
            sum_accuracy += self.model.accuracy.data * batchsize
        return sum_loss / self.testsize, sum_accuracy / self.testsize


    def save(self, model_filename, optimizer_filename):
        """ Save the state of the model & optimizer to disk """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(optimizer_filename, self.optimizer)


class Classifier(object):
    """ Load a state from disk, and use it to classify submissions"""
    def __init__(self, net_size, model_filename, optimizer_filename):
        """ Create the underlying neural network model """
        self.model = L.Classifier(BaseNetwork(net_size))
        if (model_filename != ""):
            serializers.load_hdf5(model_filename, self.model)

        """ Create the underlying optimizer """
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        if (optimizer_filename != ""):
            serializers.load_hdf5(optimizer_filename, self.optimizer)

    def classify(self, phrase_vector):
        """ Run this over an input vector and see the result """
        x = Variable(np.asarray([phrase_vector]))
        return self.model.predictor(x).data[0]
