from .constants import CORPUS_NAME, LINES_FILEPATH, LINES_FIELDS, CONVOS_FILEPATH, CONVOS_FIELDS, DATAFILE, DELIMITER  # pylint: disable=relative-beyond-top-level

####################################
### FORMATTING FUNCTIONS/PROCESS ###
####################################


def printLines(file, n=10):
    ''' Print out n (10) lines from 'file' '''
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def loadLines(filepath, fields):
    ''' split each line of 'filepath' into dictionary of 'fields' '''
    lines = {}
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {field: values[i] for i, field in enumerate(fields)}
            lines[lineObj['lineID']] = lineObj
    return lines


def loadConvos(filepath, lines, fields):
    ''' group 'fields' of 'lines' in 'filepath' into conversations '''
    convos = []
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            convObj = {field: values[i] for i, field in enumerate(fields)}
            # convert str from split to list '['a', 'b', ...]'
            lineIds = eval(convObj['utteranceIDs'])
            convObj['lines'] = [lines[lineId] for lineId in lineIds]
            convos.append(convObj)
    return convos


def extractSentencePairs(convos):
    ''' extract sentence pairs from 'convos' '''
    qa_pairs = []
    for convo in convos:
        # iterate over all conversation lines
        for i in range(len(convo['lines']) - 1):
            inputLine = convo['lines'][i]['text'].strip()
            targetLine = convo['lines'][i+1]['text'].strip()
            # filter wrong samples if one of lists is empty
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs
