import torch
import torch.nn as nn

# Luong attention layer


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, 'is not an appropriate attention method.')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        # element-wise multiplication of current decoder hidden with encoder output and sum them
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    '''
    hidden: (shape=(1, batch_size, hidden_size))
    encoder_outputs: (shape=(max_length, batch_size, hidden_size))
    '''

    def forward(self, hidden, encoder_outputs):
        # calculate attention weights (energies)
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        # return softmax normalized probability scores (with added dimension)
        return nn.F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):

        # local variable initialization
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # initialize attention model
        self.attn = Attn(attn_model, hidden_size)

    '''
    PARAMETERS
    input_step: 1 timestep/word of input sequence batch (shape=(1, batch_size))
    last_hidden: final hidden layer of GRU (shape=(n_layers x num_dirs, batch_size, hidden_size))
    encoder_outputs: encoder model's output (shape=(max_length, batch_size, hidden_size))
    RETURN
    output: softmax normalized tensor giving probabilities of each word being the correct next word in decoded sequence
    hidden: final hidden state of GRU
    '''

    def forward(self, input_step, last_hidden, encoder_outputs):

        # STEP 1: get embedding of current input word
        embedded = self.embedding_dropout(self.embedding(input_step))

        # STEP 2: forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # STEP 3: calculate attention weights from current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # STEP 4: "weighted sum" context vector = attn_weights * encoder_outputs
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # STEP 5: concatenate context and GRU output
        # column concatenation
        concat_input = torch.cat(
            (rnn_output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # STEP 6: predict next word with Luong eq. 6
        output = nn.F.softmax(self.out(concat_output), dim=1)

        return output, hidden
