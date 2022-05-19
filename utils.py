# -*- coding: utf-8 -*-
# @Time : 2022/5/16 21:45
# @Author : Bruno
# @File : utils.py
from config.config import get_params
import re
import gensim
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

random.seed(10)


def dict2arrary(uid_iid_rating: dict) -> np.array:
    np_uid, np_iid = list(zip(*list(uid_iid_rating.keys())))
    np_ratings = list(uid_iid_rating.values())
    return np.array(np_uid), np.array(np_iid), np.array(np_ratings)


def compt_num(uid_iid_rating: dict):
    uids, iids = list(zip(*list(uid_iid_rating.keys())))
    num_uids = len(set(uids))
    num_iids = len(set(iids))
    return num_uids, num_iids


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


# 划分数据集
def split_dict(data: dict, ratio=0.7) -> dict:
    train = list(data.items())[:int(ratio * len(data))]
    test = list(data.items())[int(ratio * len(data)):]
    return dict(train), dict(test)


def review2vec(params, reviews):
    """
    The processed words are converted into the corresponding vectors
    by pre-training and setting the length of a review to 100.
    Original review -> remove useless symbols -> convert to lower case -> split word -> get word2vector

    Args:
        params: dict;
        reviews: list;
    Returns:
        vec_reviews: list; A vector of review.
    """
    r3 = "[^A-Za-z]+"
    vec_reviews = []
    pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(params['PRE_W2V_BIN_PATH'], binary=True)
    stopwordslist = [k.strip() for k in open(params['stopwords'], encoding='utf8').readlines() if k.strip() != '']
    print("=" * 10, "pretrained vector has been introduced", "=" * 10)

    for sentence in tqdm(reviews):
        s2v = []
        sentence = sentence[:400]
        sentence = re.sub(r3, ' ', sentence)
        sentence = sentence.lower()
        for w in sentence.split(" "):
            if w not in stopwordslist:
                if w in pre_word2v:
                    s2v.append(pre_word2v[w])
        # filter_sentence= [w for w in sentence.split(" ") if w not in stopwords.words('english') if w in pre_word2v]
        # for word in filter_sentence:
        #     if word in pre_word2v:
        #         s2v.append(pre_word2v[word])
        s2v = padding_s2v(s2v, params["sentence_len"])
        # if len(s2v) < params["sentence_len"]:
        #     padding_s2v(s2v,params["sentence_len"])
        # else:
        #     s2v = s2v[:80]
        vec_reviews.append(s2v)
    return vec_reviews


def padding_s2v(s2v, length):
    """Complete sentences less than required length in length.

    Args:
        s2v: list; s2v.shape = [num of words, 300]
        len: int; required length
    Returns:
        s2v: list; completed sentence
    """
    if len(s2v) < length:
        a = length - len(s2v)
        while a:
            s2v.append(np.zeros(300, ))
            a = a - 1
    else:
        s2v = s2v[:80]
    return s2v


def id2num(id):
    """
    Converts character titles to ordinal titles.

    Args:
        id: list; id[i] = 'EFFEFEVCSCE'
    Returns:
        num_id: list; num_id[i] = 1
    """
    id2num = dict()
    flag = 0
    for i in id:
        if i not in id2num.keys():
            id2num[i] = flag
            flag = flag + 1
        else:
            pass
    num_users_id = []
    for i in id:
        num_users_id.append(id2num[i])
    return num_users_id


def build_dataset(spam_iid_uid, gen_iid_uid, uiid_rating, ratio):
    """

    :param spam_iid_uid:
    :param gen_iid_uid:
    :param uiid_rating:
    :param ratio: 控制训练数据集中的虚假行为数据的比例
    :return:
    """
    spam_data = dict2list(uiid_rating, spam_iid_uid)
    gen_data = dict2list(uiid_rating, gen_iid_uid)
    random.shuffle(spam_data)
    random.shuffle(gen_data)
    testdata = gen_data[:int(0.3 * len(gen_data))]
    traindata = gen_data[int(0.3 * len(gen_data)):] + spam_data[:int(ratio * (len(spam_data)))]
    print('test', np.average(np.array(testdata)[:, 2]))
    print('train%s' % ratio, np.average(np.array(traindata)[:, 2]))
    return traindata, testdata


def build_dataset_ratio(spam_iid_uid, gen_iid_uid, uiid_rating, spam_sample, gen_sample):
    """
    Construct training and test data sets with positive and negative sample ratios.

    Args:
        spam_iid_uid: ditc
        gen_iid_uid: dict
    Returns:
        dict_traindata_x
        dict_testdata_x
    """
    dict_traindata_x = defaultdict(list)  # dict {itemid:[1,2,3,4]}
    dict_testdata_x = defaultdict(list)
    for key in spam_iid_uid.keys():
        if len(list(spam_iid_uid[key])) >= spam_sample:
            train_spam = list(spam_iid_uid[key])[:spam_sample]
            train_gen = list(gen_iid_uid[key])[:gen_sample]
            train_spam.extend(train_gen)
            dict_traindata_x[key] = train_spam
            test_gen = list(gen_iid_uid[key])[30:40]
            dict_testdata_x[key] = test_gen
    traindata = dict2list(uiid_rating, dict_traindata_x)
    testdata = dict2list(uiid_rating, dict_testdata_x)
    return traindata, testdata


def dict2list(uiid_rating, dict_x):
    """
    {0:[1,2,3,4]}  ——>   [[0, 1], [0, 2], [0, 3], [0, 4]]

    Args:
        dict_x: dict,{0:[1,2,3,4]}
        uiid_rating: dict, {(1,0):5}
    Returns:
        dataset: [u,i,r]
    """
    dataset = []
    for key, value in dict_x.items():
        for i in range(len(value)):
            dataset.append([value[i], key, float(uiid_rating[(value[i], key)])])
    return dataset


def single_plot(great, params, fix_value, total):
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    x1 = list(range(len(great)))
    y1 = great

    plt.plot(x1, y1, linewidth=0.5, label='differnce')  # label为设置图例标签，需要配合legend（）函数才能显示出
    # plt.xlabel('item_id')
    plt.ylabel('rating')
    plt.title('%s difference > %s, total:%s' % (params['data_name'], fix_value, total))

    plt.legend()  # 需要配合这个才能显示图例标签
    fig.tight_layout(h_pad=2)  # 设置子图间的间隙，还有参数w_pad等
    plt.savefig('%s_diff>%s.png' % (params['data_name'], fix_value), dpi=500)
    plt.show()


def single_plot_v2(great, params, total):
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    x1 = list(range(len(great)))
    y1 = great

    plt.plot(x1, y1, linewidth=1, label='spam_num')  # label为设置图例标签，需要配合legend（）函数才能显示出
    # plt.xlabel('item_id')
    plt.ylabel('rating')
    plt.title('%s spam num, total:%s' % (params['data_name'], total))

    plt.legend()  # 需要配合这个才能显示图例标签
    fig.tight_layout(h_pad=2)  # 设置子图间的间隙，还有参数w_pad等
    plt.savefig('%s_spam_num.png' % params['data_name'], dpi=500)
    plt.show()


def double_plot(mix_aver_rating, gen_aver_rating, params):
    """
    draw two images

    :param mix_aver_rating:
    :param gen_aver_rating:
    :param params:
    :return:
    """
    fig = plt.figure(figsize=(10, 3))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    x1 = list(gen_aver_rating.keys())
    y1 = list(gen_aver_rating.values())
    x2 = list(mix_aver_rating.keys())
    y2 = list(mix_aver_rating.values())

    plt.plot(x1, y1, linewidth=0.2, label='real behavior')  # label为设置图例标签，需要配合legend（）函数才能显示出
    plt.plot(x2, y2, linewidth=0.2, label='mixed behavior')
    plt.xlabel('item_id')
    plt.ylabel('rating')
    plt.title('%s compare rating' % params['data_name'])

    plt.legend()  # 需要配合这个才能显示图例标签
    fig.tight_layout(h_pad=2)  # 设置子图间的间隙，还有参数w_pad等
    plt.savefig('%s.png' % params['data_name'], dpi=500)
    plt.show()


def split_data(data, ratio, part=False):
    """
    split data to train and test
    :param data: completed data
    :return: train,test
    """
    if part:
        train = data[:8]
        test = data[8:10]
    else:
        train = data[:int(len(data) * ratio)]
        test = data[int(len(data) * ratio):]
    return torch.from_numpy(np.array(train)), torch.from_numpy(np.array(test))


if __name__ == '__main__':
    s = ['deaf', 'deda', 'deaf']
    print(id2num(s))
    d = {(1, 23): 2, (2, 3): 1}
    print(list(d.keys())[0])
