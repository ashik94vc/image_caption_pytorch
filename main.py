import torch

from scripts.build_vocab import Vocabulary
from core.classifier import Classifier

classifier = Classifier()
num_epochs = 50

for i in range(num_epochs):
    classifier.train(i)
    classifier.test(i)