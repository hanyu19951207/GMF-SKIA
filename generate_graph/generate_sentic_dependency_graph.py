# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


def load_sentic_word():
    """
    load senticNet
    """
    path = '../senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    text_list =text.split()
    sentic_all = 1
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)

    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)])
        else:
            sentic = 0
        sentic_all += abs(sentic)

# sentic&dependency tree
#     for token in document:
#         # print('token:', token)
#         if str(token) in senticNet:
#             sentic = float(senticNet[str(token)])
#         else:
#             sentic = 0
#         # if str(token) in aspect:
#         #     sentic += 1
#         if token.i < seq_len:
#             matrix[token.i][token.i] = 1 * sentic/(sentic_all)
#             # https://spacy.io/docs/api/token
#             for child in token.children:
#                 # if str(child) in aspect:
#                 #     sentic += 1
#                 if child.i < seq_len:
#                     matrix[token.i][child.i] = 1 * sentic/(sentic_all)
#                     matrix[child.i][token.i] = 1 * sentic/(sentic_all)
    for token in document:
        #print('token:', token)
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in aspect:
                    sentic += 1
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic
    return matrix

def process(filename):
    senticnet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph_sdat', 'wb')
    graph_idx = 0
    for i in range(len(lines)):
        aspects, polarities, positions, text = lines[i].split('\t')
        aspect_list = aspects.split('||')
        polarity_list = polarities.split('||')
        position_list = positions.split('||')
        text = text.lower().strip()
        aspect_graphs = {}
        aspect_positions = {}
        for aspect, position in zip(aspect_list, position_list):
            aspect_positions[aspect] = position
        for aspect, position in zip(aspect_list, position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            # other_aspects
            adj_matrix = dependency_adj_matrix(text, aspect, senticnet)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph, fout)
    print('done !!!' + filename)
    fout.close()


if __name__ == '__main__':
    process('../con_datasets/rest14/rest14_train.raw')
    process('../con_datasets/rest14/rest14_test.raw')
    # process('../con_datasets/lap14/lap14_train.raw')
    # process('../con_datasets/lap14/lap14_test.raw')
    # process('../con_datasets/rest15/rest15_train.raw')
    # process('../con_datasets/rest15/rest15_test.raw')
    # process('../con_datasets/rest16/rest16_train.raw')
    # process('../con_datasets/rest16/rest16_test.raw')