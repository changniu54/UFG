import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.linear_model as models


def att_clb(attribute, seenclasses, unseenclasses, num, ratio):

    nclass = attribute.size(0) # all classes' number
    seen_nclass = seenclasses.size(0)
    unseen_nclass = unseenclasses.size(0)
    att = torch.zeros(nclass * num, attribute.size(1))
    att1 = torch.zeros(num * unseen_nclass, attribute.size(1))
    att2 = torch.zeros(seen_nclass * num, attribute.size(1))
    seen_att = attribute[seenclasses,]
    unseen_att = attribute[unseenclasses,]

    # similarity = cosine_similarity(attribute.numpy(), attribute.numpy())
    # similarity = torch.from_numpy(similarity).float()

    LASSO1 = models.Ridge(alpha=1)
    LASSO1.fit(seen_att.numpy().transpose(), unseen_att.numpy().transpose())  # use train to represent test
    similar1 = LASSO1.coef_
    similar1[similar1 < 1e-3] = 0
    tmp1 = np.sum(similar1, axis=1)
    tmp2 = np.tile(tmp1, (similar1.shape[1], 1)).transpose()
    similar1 = similar1 / tmp2
    similarity1 = torch.from_numpy(similar1).float() #（10,40）for AWA


    LASSO2 = models.Ridge(alpha=1)
    LASSO2.fit(unseen_att.numpy().transpose(), seen_att.numpy().transpose())  # use test to represent train
    similar2 = LASSO1.coef_
    similar2[similar2 < 1e-3] = 0
    tmp1 = np.sum(similar2, axis=1)
    tmp2 = np.tile(tmp1, (similar2.shape[1], 1)).transpose()
    similar2 = similar2 / tmp2
    similarity2 = torch.from_numpy(similar2.transpose()).float() #（40,10）for AWA

    # sim, idx = torch.sort(similarity, dim=1, descending=True)
    sim1, idx1 = torch.sort(similarity1, dim=1, descending=True)
    sim2, idx2 = torch.sort(similarity2, dim=1, descending=True)


   # expand unseen classes
    for i in range(unseen_nclass):
        att1[i * num, ] = unseen_att[i,]
        for j in range(num-1):
            att1[i*num + j+1,] = (1-ratio) * unseen_att[i,] + ratio * seen_att[idx1[i, j],] # 取前面num-1个近邻来做混淆
            # print('unseen: %d + seen: %d\n' %(i,idx1[i,j]))

   # expand seen classes
    for i in range(seen_nclass):
        att2[i * num, ] = seen_att[i,]
        for j in range(num-1):
            att2[i*num + j+1,] = (1-ratio) * seen_att[i,] + ratio * unseen_att[idx2[i, j],]

    att = torch.cat([att1,att2],dim=0)

    return att

