Char Classifier
---------

Given a short phrase, train a neural network to classify it based on the characters that it contains.

The trainer takes in a file of phrases (separated by new line) and a file of target values (separated by new line).

To train on the sample data (parts of speech classification): :: 

  import chClassifier
  p = chClassifier.Trainer("sample/words", "sample/parts-of-speech")
  p.train([100, 100], 20, 100)
  p.save("sample", "example-run")

This will save a trained model with the tag 'example-run' in the sample directory. To use that model, run: ::

  import chClassifier
  k = chClassifier.Classifier("sample", "example-run")
  print k.classify("dog")


This will return an array of tuples of original label + likelyhood that the label is correct, like so: ::

  >> print k.classify("dog")
  [(u'VB', 0.050349433), (u'NN', 3.8027303)]


The higher the number, the more sure we are of the classification (`dog` is definitely a noun, for example, and probably not a verb). 
