import torch
import argparse
import pickle
from scripts.build_vocab import Vocabulary
from core.classifier import Classifier

def train_model():
    classifier = Classifier()
    num_epochs = 50
    for i in range(num_epochs):
        classifier.train(i)
        classifier.test(i)


parser = argparse.ArgumentParser()
parser.add_argument("--load", action='store_true', help="Load Model Flag set to True to Load Pretrained model")
parser.add_argument("-encoder",help="Encoder model path")
parser.add_argument("-decoder",help="Decoder model path")
parser.add_argument("-sample", help="Sample Image")
args = parser.parse_args()
print(args)
if args.load:
    encoder_path = args.encoder
    decoder_path = args.decoder
    sample_path = args.sample
    # with open(encoder_path) as f:
    #     encoder_params = pickle.load(f)
    # with open(decoder_path) as f:
    #     decoder_params = pickle.load(f)
    classifier = Classifier(encoder_path, decoder_path)
    result = classifier.sample(sample_path)
    print(result)
else:
    train_model()