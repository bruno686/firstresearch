# -*- coding: utf-8 -*-
# @Time : 2022/5/16 22:13
# @Author : Bruno
# @File : main.py
from models import DetectNet, GCF, LFM
from data.load_data import load_all
from datasets import AmazonDataset
from config.config import get_params
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import split_dict, random_dic
# from trainers import Trainer
from trainers_lmf import Trainer
# from profiler_trainer import Trainer
from modules import my_func
import torch

def main():
    params = vars(get_params())
    uid_iid_label_dict = defaultdict(int)
    uid_iid_rating_dict = defaultdict(int)
    uid_iid_fea, uid_iid_label, uid_iid_rating = load_all(params)
    uid_iid_fea_dict = random_dic(uid_iid_fea)
    for i in uid_iid_fea_dict.keys():
        uid_iid_label_dict[i] = uid_iid_label[i]
        uid_iid_rating_dict[i] = uid_iid_rating[i]
    x_fea_train, x_fea_test = split_dict(uid_iid_fea_dict)
    y_label_train, y_label_test = split_dict(uid_iid_label_dict)
    y_rating_train, y_rating_test = split_dict(uid_iid_rating_dict)
    x_uid_train = list(zip(*list(x_fea_train.keys())))[0]
    x_iid_train = list(zip(*list(x_fea_train.keys())))[1]
    x_uid_test = list(zip(*list(x_fea_test.keys())))[0]
    x_iid_test = list(zip(*list(x_fea_test.keys())))[1]
    y_label_train = list(y_label_train.values())
    y_label_test = list(y_label_test.values())
    y_rating_train = list(y_rating_train.values())
    y_rating_test = list(y_rating_test.values())
    x_fea_train = list(x_fea_train.values())
    x_fea_test = list(x_fea_test.values())
    train_dataset = AmazonDataset(x_uid_train,x_iid_train,x_fea_train,y_label_train,y_rating_train)
    test_dataset = AmazonDataset(x_uid_test,x_iid_test,x_fea_test,y_label_test,y_rating_test)
    train_dataloader = DataLoader(train_dataset, collate_fn=my_func, batch_size=params['train_batch_size'])
    test_dataloader = DataLoader(test_dataset, collate_fn=my_func, batch_size=params['test_batch˚_size'])
    trainer = Trainer(DetectNet, LFM, uid_iid_rating)
    trainer.train(train_dataloader,test_dataloader)
    trainer.test(test_dataloader)


main()
