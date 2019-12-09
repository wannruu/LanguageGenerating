import torch
import torch.nn.functional as F
from . import utils
import numpy as np

class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('LanguageModel.predict_all')
        

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            """
            Implement a Causal convolution followed by a ReLU, and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            super().__init__()
            self.c1 = torch.nn.ConstantPad1d((2*dilation, 0), 0)
            self.c2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
            self.c3 = torch.nn.ConstantPad1d((2*dilation, 0), 0)
            self.c4 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
            self.c5 = torch.nn.ConstantPad1d((2*dilation, 0), 0)
            self.c6 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
            self.skip = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            identity = self.skip(x)

            x = self.c1(x)
            x = F.relu(self.c2(x))
            x = self.c3(x)
            x = F.relu(self.c4(x))
            x = self.c5(x)
            x = F.relu(self.c6(x))
            
            x = x + identity

            return x

    def __init__(self, layers=[32, 32, 50, 50, 64, 64], kernel_size=3):
        super().__init__()
        L = []
        c = len(utils.vocab)
        total_dilation = 1
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, kernel_size, dilation=total_dilation))
            c = l
            total_dilation *= 2
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, len(utils.vocab), 1)
        self.first_char = torch.nn.Parameter(torch.zeros(len(utils.vocab), 1))

    def forward(self, x):
        """
        x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if x.size(2) == 0: # empty string to predict its first character
            result = self.first_char[None]
            result = result.repeat(x.size(0), 1, 1)
            return result
        else:
            result = self.first_char[None]
            result = result.repeat(x.size(0), 1, 1)
            output = torch.cat((result, self.classifier(self.network(x))), dim=2)
            output = output.to(device)
            return output

    def predict_all(self, some_text):
        """
        some_text: a string
        return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods
        """
        one_hot = utils.one_hot(some_text)
        result = self.forward(one_hot[None])
        final = F.log_softmax(result.squeeze(0), dim=0)
        return final


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r

