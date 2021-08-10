import Levenshtein
import re
import numpy as np

# could not remove comas, as variants...
# if '_V1' in Config.language:
#     lowbound = 0; highbound = 10500
# elif Config.language == 'fb_dbp':
#     lowbound = 7662; highbound = 25542
# else:
#     lowbound = 0; highbound = 10500

lowbound = 0; highbound = 10500; lowbound1 = 15000; lowbound2 = 15000

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UEA')
    parser.add_argument('--lan', type=str, default='zh_en')

    args = parser.parse_args()
    print(args)

    language = args.lan
    e1 = 'data/' + language + '/ent_ids_1'
    e1_trans = 'data/' + language + '/ent_ids_1_trans_goo'
    e2 = 'data/' + language + '/ent_ids_2'
    r1 = 'data/' + language + '/rel_ids_1'
    r2 = 'data/' + language + '/rel_ids_2'
    ill = 'data/' + language + '/ref_ent_ids'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'

    # inf1 = open(e1)
    inf1 = open(e1_trans)
    id2name1_test = dict()
    for i1, line in enumerate(inf1):
        strs = line.strip().split('\t')
        wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
        wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
        if (i1>=lowbound and i1<highbound) or i1>=lowbound1:
            id2name1_test[len(id2name1_test)] = wordline
    print(len(id2name1_test))

    inf2 = open(e2)
    #inf2 = open(Config.e2_trans)
    id2name2_test = dict()
    for i1, line in enumerate(inf2):
        strs = line.strip().split('\t')
        wordline = strs[1].replace('http://dbpedia.org/resource/','').lower().replace('(','').replace(')','')
        wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
        if (i1>=lowbound and i1<highbound):# or i1>=lowbound2:
            id2name2_test[len(id2name2_test)] = wordline
    print(len(id2name2_test))

    overallscores = []
    for item in range(len(id2name1_test)):
        # print(item)
        name1 = id2name1_test[item]
        scores = []
        for item in range(len(id2name2_test)):
            name2 = id2name2_test[item]
            scores.append(Levenshtein.ratio(name1, name2))
        overallscores.append(scores)

    print(np.array(overallscores))

    np.save('./data/'+ language + '/string_mat.npy', np.array(overallscores))



