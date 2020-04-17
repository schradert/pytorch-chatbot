import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True)

    '''
    PARAMS
    input_seq: batch of input sentences (shape=(max_length, batch_size))
    input_lengths: list of sentence lengths (shape=(batch_size))
    hidden: hidden state (shape=(n_layers x num_directions, batch_size, hidden_size))
    RETURN
    outputs: features from last hidden GRU layer (shape=(max_length, batch_size, hidden_size))
    hidden: updated hidden state (shape=(n_layers x num_directions, batch_size, hidden_size))
    '''

    def forward(self, input_seq, input_lengths, hidden=None):
        # convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # pack padded batch of embeddings for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]

        return outputs, hidden
