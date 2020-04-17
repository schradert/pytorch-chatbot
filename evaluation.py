########################
### EVALUATION CLASS ###
########################
import torch
import torch.nn as nn
from .constants import SOS_token, DEVICE, MAX_LENGTH


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):

        # forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # prepare encoder's final hidden layer to be first hidden decoder input
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=DEVICE, dtype=torch.long) * SOS_token

        # initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)

        # iteratively decode one word token at a time
        for _ in range(max_length):

            # decoder forward pass
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs)

            # obtain most likely word token and softmax score
            decoder_scores, decoder_input = torch.max(
                decoder_output,
                dim=1)

            # record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # prepare current token to be next decoder input (add dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # return collections of word tokens and scores
        return all_tokens, all_scores

##########################
### EVALUATE FUNCTIONS ###
##########################


def evaluate(indexesFromSentence, encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):

    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]

    # create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # transpose batch dimensions to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    # use appropriate CUDA device
    input_batch = input_batch.to(DEVICE)
    lengths = lengths.to(DEVICE)

    # decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)

    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words

########################
### EVALUATE PROCESS ###
########################


def evaluateInput(indexesFromSentence, normalizeStr, encoder, decoder, searcher, voc):
    input_sentence = ''
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')

            # check for quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break

            # normalize sentence
            input_sentence = normalizeStr(input_sentence)

            # evaluate sentence
            output_words = evaluate(indexesFromSentence,
                                    encoder, decoder, searcher, voc, input_sentence)

            # format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print('Error: Encountered unknown word.')
