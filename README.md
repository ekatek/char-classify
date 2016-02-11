Char Classifier
---------

Given a short phrase, train a neural network to classify it based on the characters that it contains.

To train on the sample data (parts of speech classification):

```
from phrase_classifier import PhraseClassificationTrainer, PhraseClassifier
p = PhraseClassificationTrainer("sample/words", "sample/parts-of-speech")
p.train([100, 100], 20, 100)
p.save("sample", "example-run")
```

To use the result, run:

```
k = PhraseClassifier("sample", "example-run")
print k.classify("dog")
```
