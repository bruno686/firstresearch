# -*- coding: utf-8 -*-
# @Time : 2022/5/17 21:59
# @Author : Bruno
# @File : profiler_trainer.py
import torch
from config.config import get_params
from modules import weighted_MSEloss
from utils import compt_num, dict2arrary

class Trainer(torch.nn.Module):
    def __init__(self, DetectNet, GCF, uid_iid_rating):
        super(Trainer, self).__init__()
        self.params = vars(get_params())
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.uid_iid_rating = uid_iid_rating
        self.num_uid, self.num_iid = compt_num(self.uid_iid_rating)
        self.uid, self.iid, self.ratings = dict2arrary(self.uid_iid_rating)
        self.detectNet = DetectNet().to(self.device)
        self.gcf = GCF(self.uid, self.iid, self.ratings, self.num_uid, self.num_iid, self.device).to(self.device)
        self.optim_detect = torch.optim.SGD(self.detectNet.parameters(), lr=self.params['learning_rate']) if \
                                            self.params['optim'] == 'SGD' else torch.optim.Adam(
                                            self.detectNet.parameters(), lr=self.params['learning_rate'])
        self.optim_rating = torch.optim.SGD(self.gcf.parameters(),lr=self.params['learning_rate']) if \
                                            self.params['optim'] == 'SGD' else torch.optim.Adam(
                                            self.gcf.parameters(),lr=self.params['learning_rate'])
        self.optim = torch.optim.SGD(list(self.gcf.parameters())+list(self.detectNet.parameters()),lr=self.params['learning_rate']) if \
                                            self.params['optim'] == 'SGD' else torch.optim.Adam(
                                            list(self.gcf.parameters())+list(self.detectNet.parameters()),lr=self.params['learning_rate'])
        self.crit_detect = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 0.6]).to(self.device))
        self.crit_rating = weighted_MSEloss
        self.lamda = self.params['lamda']

    def train(self, train_dataloader):
            for i in range(self.params['train_epochs']):
                total_loss_detect = 0
                total_loss_rating = 0
                total_loss = 0
                with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True,with_stack=True) as prof:
                    for index, batch in enumerate(train_dataloader):
                        with torch.autograd.profiler.record_function("batch[]"):
                            uid = batch[0].to(device=self.device)
                            iid = batch[1].to(device=self.device)
                            fea = batch[2].to(device=self.device)
                            label = batch[3].to(device=self.device)
                            rating = batch[4].to(device=self.device)
                        with torch.autograd.profiler.record_function("predict data"):
                            pred_detect = self.detectNet(fea)
                            pred_rating = self.gcf(uid, iid)
                        with torch.autograd.profiler.record_function("compute loss"):
                            weight = torch.argmax(pred_detect,dim=1)
                            loss_detect = self.crit_detect(pred_detect, label.long())
                            loss_rating = sum(self.crit_rating(pred_rating, rating, weight))
                            total_loss = (1 - self.lamda) * loss_detect + self.lamda * loss_rating
                        with torch.autograd.profiler.record_function("backward"):
                            total_loss.backward()
                        # self.optim_detect.step()
                        # self.optim_rating.step()
                        with torch.autograd.profiler.record_function("step"):
                            self.optim.step()
                            self.optim.zero_grad()
                        # self.optim_detect.zero_grad()
                        # self.optim_rating.zero_grad()
                        total_loss_detect += loss_detect
                        total_loss_rating += loss_rating
                        total_loss += total_loss
                    print('epoch{}, loss_detect={}, loss_rating={}, total_loss={}'.format(i, total_loss_detect / len(rating),
                                                                                          total_loss_rating / len(rating),
                                                                                       total_loss / len(rating)))
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                prof.export_chrome_trace('./profiler_main')
    def test(self):
        pass
