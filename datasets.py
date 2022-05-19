# -*- coding: utf-8 -*-
# @Time : 2022/5/16 21:43
# @Author : Bruno
# @File : datasets.py.py
import torch
from torch.utils.data import Dataset


class AmazonDataset(Dataset):
    def __init__(self, uid: tuple, iid: tuple, fea: list, label: list, rating: list):
        self.uid = uid
        self.iid = iid
        self.fea = fea
        self.label = label
        self.rating = rating

    def __len__(self):
        return len(self.fea)

    def __getitem__(self, index: int) -> torch.tensor:
        return self.uid[index], self.iid[index], self.fea[index], self.label[index], self.rating[index]
