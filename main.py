from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import sys
import os
import torch
import torch.nn as nn
from . import constants as c, encoder, decoder, evaluation, batching, training, loading, formatting
from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from evaluation import GreedySearchDecoder, evaluateInput
from batching import batch2TrainData, indexesFromSentence
from training import trainIters
from loading import loadPrepareData, trimRareWords, normalizeStr, Vocabulary
from formatting import loadLines, loadConvos, extractSentencePairs, printLines


def format_data():
    # (1) Load lines and process conversations
    print('\nProcessing corpus...')
    lines = loadLines(c.LINES_FILEPATH, c.LINES_FIELDS)
    print('\nLoading conversations...')
    convos = loadConvos(c.CONVOS_FILEPATH, lines, c.CONVOS_FIELDS)

    # (2) Write new csv file
    print('\nWriting newly formatted file...')
    with open(c.DATAFILE, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=c.DELIMITER,
                            lineterminator='\n')
    for pair in extractSentencePairs(convos):
        writer.writerow(pair)

    # (3) Print a sample of lines for confirmation
    print('\nSample lines from file:')
    printLines(c.DATAFILE)


def load_data():
    # (1) Load/Assemble voc and pairs
    voc, pairs = loadPrepareData(
        c.CORPUS,
        c.CORPUS_NAME,
        c.DATAFILE,
        c.SAVE_DIR)
    # (2) trim voc and pairs
    pairs = trimRareWords(voc, pairs, c.MIN_COUNT)
    return voc, pairs


def initialize_models(voc):
    print('Building encoder and decoder ...')
    # initialize word embeddings
    embedding = nn.Embedding(
        voc.num_words,
        c.HIDDEN_SIZE)
    if c.LOAD_FILENAME:
        embedding.load_state_dict(embedding_sd)
    # initialize encoder & decoder models
    encoder = EncoderRNN(
        c.HIDDEN_SIZE,
        embedding,
        c.ENCODER_N_LAYERS,
        c.DROPOUT)
    decoder = LuongAttnDecoderRNN(
        c.ATTN_MODEL,
        embedding,
        c.HIDDEN_SIZE,
        voc.num_words,
        c.DECODER_N_LAYERS,
        c.DROPOUT)
    if c.LOAD_FILENAME:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(c.DEVICE)
    decoder = decoder.to(c.DEVICE)
    print('Models built and ready to go!')
    return embedding, encoder, decoder


def pre_training_configuration(encoder, decoder):

    # (1) ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # (2) initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=c.LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=c.LEARNING_RATE * c.DECODER_LEARNING_RATIO)

    # (3) load states if mid-way
    if c.LOAD_FILENAME:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # (4) configure CUDA to optimized state tensors
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return encoder_optimizer, decoder_optimizer


def train_data(voc, pairs, embedding, encoder, decoder, enc_optim, dec_optim):
    # run training iterations
    print('Starting Training!')
    trainIters(
        batch2TrainData,
        checkpoint,
        c.MODEL_NAME,
        voc,
        pairs,
        encoder,
        decoder,
        enc_optim,
        dec_optim,
        embedding,
        c.ENCODER_N_LAYERS,
        c.DECODER_N_LAYERS,
        c.SAVE_DIR,
        c.N_ITERATION,
        c.BATCH_SIZE,
        c.PRINT_EVERY,
        c.SAVE_EVERY,
        c.CLIP,
        c.CORPUS_NAME,
        c.LOAD_FILENAME)


def evaluate(encoder, decoder, voc):
    # (1) set dropout layers to eval mode
    encoder.eval()
    # (2) initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)
    # (3) initialize real-time chatbot
    evaluateInput(
        indexesFromSentence,
        normalizeStr,
        encoder,
        decoder,
        searcher,
        voc)


if __name__ == '__main__':
    # (1) data formatting and pairs generation
    format_data()
    # (1) load data into vocabulary and pairs
    voc, pairs = load_data()
    # (2) load old state in if applicable
    loadFileTag = False
    if loadFileTag:
        loadFileTag = os.path.join(
            c.SAVE_DIR,
            c.MODEL_NAME,
            c.CORPUS_NAME,
            f'{c.ENCODER_N_LAYERS}-{c.DECODER_N_LAYERS}_{c.HIDDEN_SIZE}',
            f'{c.CHECKPOINT_ITER}_checkpoint.tar')
        # if loading on same machine model was trained on
        checkpoint = torch.load(c.LOAD_FILENAME)
        # if loading model trained on GPU to CPU
        checkpoint = torch.load(
            c.LOAD_FILENAME, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc = Vocabulary(c.CORPUS_NAME)
        voc.__dict__ = checkpoint['voc_dict']
    # (3) format

    # (3) initialize all models
    embedding, encoder, decoder = initialize_models(voc)
    # (4) prepare models for the training process
    enc_optim, dec_optim = pre_training_configuration(encoder, decoder)
    # (5) train
    train_data(voc, pairs, embedding, encoder, decoder, enc_optim, dec_optim)

    # (5) evaluate
    print('Beginning Evaluation!')
    evaluate(encoder, decoder, voc)
