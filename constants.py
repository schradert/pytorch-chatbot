import torch
import os
import codecs

##############
### DEVICE ###
##############
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")  # pylint: disable=no-member

###############
### LOADING ###
###############
PAD_token = 0  # used for padding short sentences
SOS_token = 1  # start of sentence token <START>
EOS_token = 2  # end of sentence token <END>
MAX_LENGTH = 10  # max sentence length to consider

SAVE_DIR = os.path.join('data', 'save')

MIN_COUNT = 3  # minimum word count threshold for trimming

CORPUS_NAME = 'cornell-moviedialog-corpus'
CORPUS = os.path.join('../input', CORPUS_NAME)

##################
### FORMATTING ###
##################

LINES_FILEPATH = os.path.join(CORPUS, 'movie_lines.txt')
CONVOS_FILEPATH = os.path.join(CORPUS, 'movie_conversations.txt')
LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
CONVOS_FIELDS = ['characterID', 'character2ID', 'movieID', 'utteranceIDs']

DATAFILE = os.path.join('..', 'formatted_movie_lines.txt')
DELIMITER = str(codecs.decode('\t', 'unicode_escape'))
################
### BATCHING ###
################
SMALL_BATCH_SIZE = 5

# configure models
MODEL_NAME = 'cb_model'
ATTN_MODEL = 'dot'     # ['dot', 'general', 'concat']
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64

LOAD_FILENAME = None
CHECKPOINT_ITER = 4000

# configure training/optimization
CLIP = 50.0
TEACHER_FORCING_RATIO = 1.0
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_ITERATION = 4000
PRINT_EVERY = 1
SAVE_EVERY = 500
