# -*- coding: utf-8 -*-
# @Time : 2022/5/16 21:44
# @Author : Bruno
# @File : models.py
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse import identity
from scipy.sparse import coo_matrix

"""
Design two models.
RecognNet: Identifying the truth of interaction data
GCF: Predict the scoring of interaction data
"""


class DetectNet(torch.nn.Module):
    def __init__(self, input_features=28):
        super(DetectNet, self).__init__()
        self.linear_3 = torch.nn.Linear(input_features, 8)
        self.linear_4 = torch.nn.Linear(8, 2)
        self.dropout_1 = torch.nn.Dropout(0.5)
        self.dropout_2 = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.linear_3(x))
        x = torch.softmax(self.linear_4(x), dim=1)
        return x


class GCF(torch.nn.Module):
    def __init__(self, uid, iid, ratings, n_users, n_items, device, emb_size=100, average=1):
        super(GCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = emb_size
        self.n_layers = 4
        self.uid = uid
        self.iid = iid
        self.ratings = ratings
        self.device = device
        self.norm_adj_matrix = self.get_norm_adj_mat().to(device=self.device)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.transForm1 = torch.nn.Linear(in_features=200, out_features=128)
        self.transForm2 = torch.nn.Linear(in_features=64, out_features=32)
        self.transForm3 = torch.nn.Linear(in_features=32, out_features=1)
        self.transForm4 = torch.nn.Linear(in_features=128, out_features=64)
        self.uBias = torch.nn.Embedding(int(n_users), 1)
        self.iBias = torch.nn.Embedding(int(n_items + n_users), 1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.av = average

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # get adjency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        uiMat = coo_matrix(
            (self.ratings, (self.uid.astype(dtype='int'), self.iid.astype(dtype='int'))))
        inter_M = uiMat
        inter_M_t = uiMat.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # add self-loop
        selfloop = identity(A.shape[0])
        A = A + selfloop
        # calculate diag matrix
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        # get LaplacianMat
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row.tolist(), col.tolist()])
        data = torch.FloatTensor(L.data.tolist())
        SparseL = torch.sparse.FloatTensor(i, data)
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, userIdx, itemIdx):
        itemIdx = itemIdx + self.n_users
        ubias = self.uBias(userIdx.long())
        ibias = self.iBias(itemIdx.long())
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        finalEmbd = lightgcn_all_embeddings
        # user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        # finalEmbd = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd, itemEmbd], dim=1)
        biases = ubias + ibias
        embd = nn.ReLU()(self.transForm1(embd))
        # embd = nn.Mish()(self.transForm1(embd))
        embd = self.transForm4(embd)
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)

        prediction = embd.flatten() + biases.flatten()

        return prediction


class LFM(nn.Module):
    def __init__(self,n_users,n_items,dim=10):
        super(LFM, self).__init__()
        # self.avg = torch.tensor(avg)
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.u_bias = nn.Embedding(n_users,1)
        self.i_bias = nn.Embedding(n_items,1)
        self.linear1 = nn.Linear(10,128)
        self.linear2 = nn.Linear(128,64)
        self.linear4 = nn.Linear(64,1)
        self.reset_para()

    def forward(self,u,v):
        param_u = self.users(u)
        param_v = self.items(v)
        uv = self.linear1(param_u*param_v)
        uv = torch.sigmoid(self.linear2(uv))
        uv = self.linear4(uv)
        uv = uv.reshape(uv.shape[0],)
        return uv+self.u_bias.weight[u.long()].squeeze()+self.i_bias.weight[v.long()].squeeze()

    def reset_para(self):
        nn.init.normal_(self.users.weight,0,1)
        nn.init.normal_(self.items.weight,0,1)
        nn.init.xavier_normal_(self.u_bias.weight)
        nn.init.xavier_normal_(self.i_bias.weight)