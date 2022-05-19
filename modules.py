# -*- coding: utf-8 -*-
# @Time : 2022/5/17 00:09
# @Author : Bruno
# @File : modules.py
import torch
def weighted_MSEloss(pred_y, real_y, weight):  # @save
    """均方损失"""
    return ((pred_y - real_y.reshape(pred_y.shape)) ** 2 / 2) * weight

def MSEloss(pred_y, real_y):  # @save
    """均方损失"""
    return (pred_y - real_y.reshape(pred_y.shape)) ** 2 / 2


def my_func(batch):
    with torch.autograd.profiler.record_function("collate"):
        uids = []
        iids = []
        features = []
        ratings = []
        labels = []
        for i in batch:
            uids.append(i[0])
            iids.append(i[1])
            features.append(i[2])
            labels.append(i[3])
            ratings.append(i[4])
    return torch.tensor(uids), torch.tensor(iids), torch.tensor(features),torch.tensor(labels),torch.tensor(ratings)
