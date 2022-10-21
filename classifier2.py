import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler 
import sys

class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, pretrain_embed=''):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.attribute = data_loader.attribute
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.pretrain_embed = pretrain_embed
        self.criterion = nn.NLLLoss()
        
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.acc_seen, self.acc_unseen, self.H = self.fit()


    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])

            acc_seen, predicted_label = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, self.attribute)
            predicted = predicted_label
            acc_unseen, predicted_label = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, self.attribute)
            predicted = torch.cat((predicted, predicted_label), 0)
            real = torch.cat((self.test_seen_label, self.test_unseen_label), 0)

            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            #print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H


        return best_seen, best_unseen, best_H
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes, attribute):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_bysmt = torch.LongTensor(test_label.size())
        #predicted_nclass = torch.LongTensor(ntest, attribute.size(0))

        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(Variable(test_X[start:end].cuda()))
                else:
                    output = self.model(Variable(test_X[start:end]))
                # _, predicted_label[start:end] = torch.max(output.data, 1)
                predicted_nclass = output.data

                if self.pretrain_embed == '':
                    pass
                else:
                    if self.cuda:
                        output = self.pretrain_embed.model(Variable(test_X[start:end].cuda()))
                    else:
                        output = self.pretrain_embed.model(Variable(test_X[start:end]))

                    temp = torch.mm(output[1].cpu(), attribute.float().t()).float()
                    logic = nn.LogSoftmax(dim=1)
                    temp = logic(temp).long()

                    predicted_nclass += 0.01*temp.long()

            _, predicted_label[start:end] = torch.max(predicted_nclass, 1)

            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, predicted_label

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).item() / torch.sum(idx).item()
        acc_per_class /= target_classes.size(0)
        return acc_per_class 



class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
