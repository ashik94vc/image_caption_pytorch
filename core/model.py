import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence

class ImageEncoder(nn.Module):
    def __init__(self, size):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, size)
        self.bn = nn.BatchNorm1d(size, momentum=0.01)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.Tensor(x.data)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x

class CaptionDecoder(nn.Module):
    def __init__(self, size, hidden_size, vocab_size, num_layers):
        super(CaptionDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, size)
        self.lstm = nn.LSTM(size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, length):
        embeddings = self.embeddings(captions)
        embeddings = self.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, length, batch_first=True)
        hiddens,_ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, feature, states=None):
        sampled_ids = []
        inputs = feature.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()