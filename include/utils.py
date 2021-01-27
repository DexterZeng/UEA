import tensorflow as tf
from include.Model import build_SE, training
import time
from include.Load import *
import json
import scipy
from scipy import spatial
import copy
from collections import defaultdict

def get_hits_gen(simM, test1, test2, id2confi, confi, correct, gap):
    rowindex = []
    columnindex = []
    for iii in test1:
        if iii>=25500:
            rowindex.append(iii-15000)
        else:
            rowindex.append(iii)
    for iii in test2:
        columnindex.append(iii-10500)
    partialsim = simM[rowindex]
    partialsim = partialsim[:, columnindex]

    sim = partialsim
    counn = 0

    for i in range(len(rowindex)):
        rank = sim[i, :].argsort()
        scores = copy.deepcopy(sim[i, :])
        scores.sort()
        minrank = rank[0]
        minscore = scores[0]
        minscoregap = scores[1] - scores[0]

        scores_col = copy.deepcopy(sim[:, minrank])
        scores_col.sort()
        minscoregap_col = scores_col[1] - scores_col[0]

        rank_col = sim[:, minrank].argsort()
        minrank_col = rank_col[0]
        # if minscore<gap:
        if minrank_col == i and minscore<gap:
        # if minrank_col == i and minscoregap > gap and minscoregap_col > gap:
            confi.append([test1[i], test2[minrank]])
            id2confi[test1[i]] = test2[minrank]
            if test1[i] + 10500 == test2[minrank]:
                correct.append([test1[i], test2[minrank]])
        counn += 1

    matchable= 0
    for item in confi:
        if item[0] < 10500:
            matchable += 1

    print("Evaluated: " + str(counn))
    print("Confi " + str(len(confi)))
    print("Among which matchable " + str(matchable))
    print("Correct " + str(len(correct)))
    return confi, correct, id2confi


def get_hits_gen_nochange(simM, test1, test2, id2confi, correct, gap):
    rowindex = []
    columnindex = []
    for iii in test1:
        if iii>=25500:
            rowindex.append(iii-15000)
        else:
            rowindex.append(iii)
    for iii in test2:
        columnindex.append(iii-10500)
    partialsim = simM[rowindex]
    partialsim = partialsim[:, columnindex]

    sim = partialsim
    counn = 0
    confi = []
    for i in range(len(rowindex)):
        rank = sim[i, :].argsort()
        scores = copy.deepcopy(sim[i, :])
        scores.sort()
        minrank = rank[0]
        # get column-wise results
        minscore = scores[0]

        scores_col = copy.deepcopy(sim[:, minrank])
        scores_col.sort()
        minscoregap_col = scores_col[1] - scores_col[0]

        rank_col = sim[:, minrank].argsort()
        minrank_col = rank_col[0]
        if  minrank_col == i and minscore<gap:
            confi.append([test1[i], test2[minrank]])
            id2confi[test1[i]] = test2[minrank]
            if test1[i] + 10500 == test2[minrank]:
                correct.append([test1[i], test2[minrank]])
        counn += 1
    print("Evaluated: " + str(counn))
    print("Confi " + str(len(confi)))
    print("Correct " + str(len(correct)))

    return confi, correct, id2confi

def getsim_matrix_cosine(vec, test_left, test_right):
    Lvec = tf.placeholder(tf.float32, [None, vec.shape[1]])
    Rvec = tf.placeholder(tf.float32, [None, vec.shape[1]])
    he = tf.nn.l2_normalize(Lvec, dim=-1)
    norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
    aep = tf.matmul(he, tf.transpose(norm_e_em))

    sess = tf.Session()
    Lv = np.array([vec[e1] for e1 in test_left])
    Rv = np.array([vec[e2] for e2 in test_right])

    aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
    aep = 1-aep
    return aep

def get_hits_ma(sim, test_pair, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        if i < 10500:
            rank_index = np.where(rank == i)[0][0]
            mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)
    msg = 'Hits@1:%.3f\n' % (top_lr[0] / 14888)
    print(msg)
