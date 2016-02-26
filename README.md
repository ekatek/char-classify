Char Classifier
---------

Given a short phrase, train a neural network to classify it based on the characters that it contains.

The trainer takes in a file of phrases (separated by new line) and a file of target values (separated by new line).

To train on the sample data (parts of speech classification):

```
from phrase_classifier import PhraseClassificationTrainer, PhraseClassifier
p = PhraseClassificationTrainer("sample/words", "sample/parts-of-speech")
p.train([100, 100], 20, 100)
p.save("sample", "example-run")
```

This will save a trained model with the tag 'example-run' in the sample directory. To use that model, run:

```
k = PhraseClassifier("sample", "example-run")
print k.classify("dog")
```
