# -*- coding: utf-8 -*-
# @Time : 2022/5/16 21:57
# @Author : Bruno
# @File : config_vedios_games.py
import argparse
import os


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    amazon = os.path.abspath("..") + "/Amazondataset"
    selfdoc = os.path.abspath(".")
    parser.add_argument("--amazon_path", type=str, default=amazon, help="Amazon dataset directory")
    parser.add_argument("--meta_path", type=str, default='/reviews_Video_Games_5.json', help="chi metadata directory")
    parser.add_argument("--data_name", type=str, default='vedio_games')
    parser.add_argument("--stopwords", type=str, default=selfdoc + '/stopword.txt')

    # info
    parser.add_argument("--metrics", type=str, default="['auc', 'mean_mrr', 'ndcg@5', 'ndcg@10']")
    parser.add_argument("--col_spliter", type=str, default='\t')

    # data
    parser.add_argument("--sentence_len", type=int, default=100)
    parser.add_argument("--his_len", type=int, default=50)
    parser.add_argument("--word_num", type=int, default=34304)
    parser.add_argument("--PRE_W2V_BIN_PATH", type=str, default='/hezhuangzhuang/GoogleNews-vectors-negative300.bin')

    # model:
    parser.add_argument("--dropout", type=float, default=0.2)

    # train
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--optim", type=str, default='SGD')
    parser.add_argument("--lamda", type=float, default='0.5')

    args, _ = parser.parse_known_args()
    return args
