import torch
import sys
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision import transforms

from tqdm import tqdm

from PIL import Image

from dataset.imagenet_dataset import ImagenetDataset
from core.resnet import ResNet
from core.image import ImageDataGenerator
from core.lib import loadModel
from core.coco import CocoCaptions, collate_fn
from core.model import ImageEncoder, CaptionDecoder


class Classifier(object):
    """
    Classifier Wrapper to Train Data
    """

    def __init__(self, encoder_params=None, decoder_params=None):
        
        self.embed_size = 256
        self.hidden_size = 512
        self.num_layers = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device %s" % self.device)

        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        vocab = loadModel('dataset/vocab.pkl')
        self.vocab = vocab
        self.vocab_size = len(vocab)
        coco_train = CocoCaptions(root="dataset/coco2014/train2014", annFile="dataset/coco2014/annotations/captions_train2014.json", vocab=vocab, transform=train_transform)
        coco_test = CocoCaptions(root="dataset/coco2014/val2014", annFile="dataset/coco2014/annotations/captions_val2014.json",vocab=vocab , transform=test_transform)
        self.trainloader = DataLoader(dataset=coco_train, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
        self.testloader = DataLoader(dataset=coco_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
        self.encoder = ImageEncoder(self.embed_size)
        self.decoder = CaptionDecoder(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if encoder_params and decoder_params:
            self.encoder.load_state_dict(torch.load(encoder_params))
            self.decoder.load_state_dict(torch.load(decoder_params))
        self.train_len = len(coco_train)
        self.test_len = len(coco_test)
        # if self.device == 'cuda':
        #     self.encoder = nn.DataParallel(self.encoder)
        #     self.decoder = nn.DataParallel(self.decoder)
        #     cudnn.benchmark = True

    def train(self, epoch):
        criterion = nn.CrossEntropyLoss()
        params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
        optimizer = optim.Adagrad(params, lr=1e-3)
        train_loss = 0
        correct = 0
        total = 0
        acc = 0
        barformat = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} {postfix[loss]}[{remaining} {rate_fmt} accuracy:{postfix[accuracy]}%]"
        pbar = tqdm(total=self.train_len, bar_format=barformat, postfix={"accuracy":acc, "loss":0, 5:0})
        for batch_idx, (data,target,length) in enumerate(self.trainloader):
            pbar.set_description("epoch {0}".format(epoch+1))
            data,target = data.to(self.device), target.to(self.device)
            targets = pack_padded_sequence(target, length, batch_first=True)[0]
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            outputs = self.encoder(data)
            outputs = self.decoder(outputs, target, length)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.0*correct/total
            pbar.postfix["accuracy"] = "{0:.2f}".format(acc)
            pbar.postfix["loss"] = "{0:10.3f}".format(train_loss)
            if self.device == "cuda":
                if batch_idx % 50 == 0:
                    pbar.update(50)
            else:
                pbar.update(1)
            if batch_idx >= self.train_len:
                break
            # print(batch_idx, train_loss)

        print("\nEpoch %d complete... " % epoch)

    def test(self, epoch):
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            barformat = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{remaining} {rate_fmt} {postfix}%]"
            pbar = tqdm(total=self.test_len, bar_format=barformat, postfix={"accuracy":acc})
            for _, (data, target, length) in enumerate(self.testloader):
                pbar.set_description("epoch {0}".format(epoch+1))
                data,target = data.to(self.device), target.to(self.device)
                targets = pack_padded_sequence(target, length, batch_first=True)[0]
                outputs = self.encoder(data)
                outputs = self.decoder(outputs, target, length)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.0*correct/total
                pbar.postfix["accuracy"] = "{0:.2f}".format(acc)
        print("Saving Model...")
        state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'epoch': epoch,
            'accuracy': acc
        }
        torch.save(state, 'checkpoint/checkpoint.t7')

    def sample(self, image):
        toTensor = transforms.ToTensor()
        input_image = toTensor(Image.open(image).resize([224,224], Image.LANCZOS))
        input_image = input_image.unsqueeze(0)
        self.encoder.eval()
        output = self.encoder(input_image)
        sampled_ids = self.decoder.sample(output)
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            if word != '<start>' and word != '<end>':
                sampled_caption.append(word)
            if word == '<end>':
                break
        print(' '.join(sampled_caption))
        
            