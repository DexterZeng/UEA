import tensorflow as tf
from include.Model import build_SE, training
from include.utils import get_hits_gen, getsim_matrix_cosine, get_hits_ma
import time
from include.Load import *
import json
import scipy
from scipy import spatial
import copy
from collections import defaultdict


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def make_print_to_file(fileName, path='./'):
#     import sys
#     import os
#     import sys
#     import datetime
#
#     class Logger(object):
#         def __init__(self, filename="Default.log", path="./"):
#             self.terminal = sys.stdout
#             self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
#
#         def write(self, message):
#             self.terminal.write(message)
#             self.log.write(message)
#
#         def flush(self):
#             pass
#     sys.stdout = Logger(fileName + '.log', path=path)
#     print(fileName.center(60,'*'))

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UEA')
    parser.add_argument('--lan', type=str, default='zh_en')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--thres', type=float, default=0.05) # initial threshold
    parser.add_argument('--inc', type=float, default=0.1)  # the increment of threshold
    parser.add_argument('--stopThres', type=float, default=0.45)  # the maximum of threshold

    parser.add_argument('--adj', type=bool, default=True) # whether dynamically adjust the threshold
    parser.add_argument('--fixedThres', type=float, default=0.45) # if adj is false, one should set a fixed weight


    args = parser.parse_args()
    print(args)

    language = args.lan
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    r1 = 'data/' + language + '/rel_ids_1'
    r2 = 'data/' + language + '/rel_ids_2'
    ill = 'data/' + language + '/ref_ent_ids'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'
    # e1_trans = 'data/' + language + '/ent_ids_1_trans_goo'
    sup = 'data/' + language + '/sup_ent_ids'
    epochs_se = 300
    epochs_ae = 600
    se_dim = 300
    ae_dim = 100
    act_func = tf.nn.relu
    gamma = 3.0  # margin based loss
    k = 25  # number of negative samples for each positive one
    seed = 3  # 30% of seeds
    beta = 0.9  # weight of SE

    t = time.time()
    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1))) # print(e)
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    test = ILL
    test_left = []; test_right = []
    inf = open(e1)
    for i, line in enumerate(inf):
        strs = line.strip().split('\t')
        if i<10500 or i>=15000:
            test_left.append(int(strs[0]))
    inf = open(e2)
    for i, line in enumerate(inf):
        strs = line.strip().split('\t')
        if i < 10500:# or i >= 15000:
            test_right.append(int(strs[0]))

    seedss = loadfile(sup, 2)
    KG1 = loadfile(kg1, 3)
    KG2 = loadfile(kg2, 3)

    path = 'data' #'entity-alignment-full-data'
    lang = language.split('_')[0]
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
    ne_vec = np.array(embedding_list)

    str_sim = np.load('./'+path+'/' + language + '/string_mat.npy')
    str_sim = 1 - str_sim
    aep_n = getsim_matrix_cosine(ne_vec, test_left, test_right)
    text_combine = aep_n * args.alpha + str_sim * (1 - args.alpha)

    clenold = 0
    id2confi = dict()
    confi = []
    correct = []
    if args.adj:
        thres = args.thres
    else:
        thres = args.fixedThres

    confi, correct, id2confi = get_hits_gen(text_combine, test_left, test_right, id2confi, confi, correct, thres)
    countt = 0

    if len(confi) < 10499:
        while len(confi) - clenold > 30:
            print('ROUND ' + str(countt))
            train = copy.deepcopy(confi)
            train = np.array(train)
            clenold = len(confi)

            test1 = []
            test2 = []
            for ee in test_left:
                if ee not in id2confi.keys():
                    test1.append(ee)
            for ee in test_right:
                if ee not in id2confi.values():
                    test2.append(ee)

            print("Generating structural embeddings.... ")
            output_layer, loss, = build_SE(se_dim, act_func, gamma, k, e, train, KG1 + KG2)
            se_vec, J = training(output_layer, loss, 25, epochs_se, train, e, k)
            countt += 1

            aep = getsim_matrix_cosine(se_vec, test_left, test_right)
            combine = aep * (1-args.beta) + text_combine * (args.beta)
            if args.adj:
                if thres >= args.stopThres:
                    thres = args.stopThres
                else:
                    thres = thres + args.inc
            else:
                thres = args.fixedThres

            confi, correct, id2confi = get_hits_gen(combine, test1, test2, id2confi, confi, correct,thres)
            print()

    conf = 0
    for item in confi:
        if item[0] < 10500:
            conf += 1
    corr = 0
    for item in correct:
        if item[0] < 10500:
            corr += 1

    print("Confi: " + str(len(confi)))
    print("Matchable: " + str(conf))
    print("Correct: " + str(corr))

    print("Precision: " + str(corr*1.0/len(confi)))
    print("Recall: " + str(corr * 1.0 / 10500))
    print("total time elapsed: {:.4f} s".format(time.time() - t))


