from .models import LanguageModel, load_model
from . import utils
import torch
import numpy as np


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Evaluate the log-likelihood of a given string

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    log_prob = model.predict_all(some_text)
    one_hot = utils.one_hot(some_text)
    log_sum = 0
    for i in range(len(some_text)):
        one_hot_column = one_hot[:, i]
        log_column = log_prob[:, i]
        each_log = torch.matmul(one_hot_column.unsqueeze(0), log_column.unsqueeze(1))
        log_sum += each_log.item()
    return log_sum


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Sample a random sentence from the language model
    Terminate once reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    S = ""
    for i in range(max_length):
        o = model.predict_all(S)[:, -1].unsqueeze(0)
        s = torch.distributions.Categorical(logits=o).sample()
        S += utils.vocab[s]
        if utils.vocab[s] == '.':
            return S
    return S


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Use beam search for find the highest likelihood generations

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    # for each beam search, first get the log prob for the next of the first char
    S = ""
    log_prob = model.predict_all(S)[:, -1]

    # sample element: (sum of log, string)
    h = TopNHeap(beam_size)
    for j in range(log_prob.size(0)):
        h.add([log_prob[j], utils.vocab[j]])
    # now sample contains beam size of elements, each element is a tuple: (log, a char, from which previous char)
    sample = [each for each in h.elements]

    #the list to store all top beam size strings ever selected
    result_sample = []
    for element in sample:
        result_sample.append(element)

    # loop from the second char
    for i in range(max_length-1):
        new_h = TopNHeap(beam_size)
        for n in range(len(sample)):
            if sample[n][1][-1] == '.':
                new_h.add(sample[n])
            else:
                log_prob = model.predict_all(sample[n][1])[:, -1]
                for l in range(log_prob.size(0)):
                    new_string = sample[n][1] + utils.vocab[l]
                    if average_log_likelihood:
                        new_prob = (sample[n][0]*len(sample[n][1]) + log_prob[l]) / (len(sample[n][1]) + 1)
                    else:
                        new_prob = sample[n][0] + log_prob[l]
                    new_h.add([new_prob, new_string])

        # keep top beam_size samples in sample
        sample = [each for each in new_h.elements]
        
        # add all beam search selected string into the result sample list:
        for data in sample:
            result_sample.append(data)

        # check if all elements end with period
        count = 0
        for k in range(len(sample)):
            if sample[k][1][-1] == '.':
                count += 1
        if count == len(sample): #all elements end with period
            sample_dict = dict(sample)
            sample_sorted = sorted(sample_dict.items(), reverse=True)
            final = [data[1] for data in sample_sorted[:n_results]]
            return final
    
    result_sample_dict = dict(result_sample)
    result_sample_sorted = sorted(result_sample_dict.items(), reverse=True)
    final = [data[1] for data in result_sample_sorted[:n_results]]

    return final


    
