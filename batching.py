import itertools
import torch
from .constants import PAD_token, EOS_token, SMALL_BATCH_SIZE  # pylint: disable=relative-beyond-top-level

##########################
### BATCHING FUNCTIONS ###
##########################


def indexesFromSentence(voc, sentence):
    ''' Convert 'sentence' into 'voc' indexes '''
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    ''' Transpose 'l' to size (max_length, batch_size) padded with 'fillvalue' '''
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    ''' Convert 'l' to binary boolean matrix '''
    return [[token == value for token in seq] for seq in l]


def inputVar(l, voc):
    ''' Return padded input tensor AND batch sequence lengths tensor given 'l' and 'voc' '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    lengths = torch.tensor(lengths)  # pylint: disable=not-callable
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)  # pylint: disable=no-member
    return padVar, lengths


def outputVar(l, voc):
    ''' Return padded target tensor, padded mask, and max target length '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = torch.BoolTensor(binaryMatrix(padList))  # pylint: disable=no-member
    padVar = torch.LongTensor(padList)  # pylint: disable=no-member
    return padVar, mask, max_target_len


########################
### BATCHING PROCESS ###
########################

def batch2TrainData(voc, pair_batch):
    ''' Return ALL (padded input, lengths, padded output, mask, max_target_len) for given 'pair_batch' and 'voc' '''
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    [input_batch, output_batch] = list(
        zip(*[(pair[0], pair[1]) for pair in pair_batch]))
    return inputVar(input_batch, voc) + outputVar(output_batch, voc)
