#########################
### LOADING CONSTANTS ###
#########################
import unicodedata
from .constants import PAD_token, SOS_token, EOS_token, MAX_LENGTH, MIN_COUNT, SAVE_DIR

#######################
### LOADING CLASSES ###
#######################


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: 'PAD',
            SOS_token: 'SOS',
            EOS_token: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # remove words below a count threshold
    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words),
            len(self.word2index),
            len(keep_words) / len(self.word2index)
        ))

        # reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}  # counts frequency of words
        self.index2word = {
            PAD_token: 'PAD',
            SOS_token: 'SOS',
            EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

#########################
### LOADING FUNCTIONS ###
#########################


def unicodeToAscii(s):
    ''' turn unicode string to pure ASCII '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalizeStr(s):
    ''' lowercase, trim, and remove non-letter characters '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


def readVocs(datafile, corpus_name):
    ''' read query/response pairs and return VOC object '''
    print('Reading lines...')
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeStr(s) for s in pair.split('\t')] for pair in lines]
    voc = Vocabulary(corpus_name)
    return voc, pairs


def filterPair(p):
    ''' returns True iff both sentences in pair 'p' are under the MAX_LENGTH threshold '''
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    ''' filter 'pairs' with filterPair condition '''
    return [pair for pair in pairs if filterPair(pair)]


#######################
### LOADING PROCESS ###
#######################

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    ''' returns populated VOC and pairs list (filtered and formatted) '''
    print('Start preparing training data ...')
    voc, pairs = readVocs(datafile, corpus_name)
    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filterPairs(pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Counted words:', voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    ''' Trim VOC words under MIN_COUNT, and filter out pairs in 'pairs' with trimmed words '''
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        # only keep pairs WITHOUT trimmed word(s) in input/output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print('Trimmed from {} pairs to {}, {:.4f} of total'.format(
        len(pairs),
        len(keep_pairs),
        len(keep_pairs) / len(pairs)))
    return keep_pairs
