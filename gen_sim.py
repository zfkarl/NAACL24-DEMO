import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
from cmhdataset import CMDataset
import time
import torchvision.models as models
import torch.nn as nn

def cal_similarity(S1):
    batch_size = S1.size(0)

    S_pair = S1 * 1.
    pro = S_pair * 1.

    size = batch_size
    top_size = int(0.3*batch_size)
    m, n1 = pro.sort()

    pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
        -1)] = 0.
    pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
        -1)] = 0.
    pro = pro / (pro.sum(1).view(-1, 1) + 1e-6)
    pro_dis = torch.matmul(pro, pro.t())
    pro_dis = pro_dis * 4000
    S_i = (S_pair * (1 - 0.3) + pro_dis * 0.3)
    return S_i


def cal_ed(x):  # x shape:[bathcsize,nviews,dim] -> [10000,5,4096]
    x = F.normalize(x,dim = -1)
    output4 = torch.zeros(x.shape[0],x.shape[0])
    for i in range(x.shape[0]):
        for j in range(i,x.shape[0]):
            output4[i,j] = (1-torch.matmul(x[i],x[j].t())).mean()
            print(i,j)
            
    output4 = output4 + output4.T - torch.diag(output4.diag())
    
    diag = output4.diag().unsqueeze(1) 
    res = 2*output4 - diag - diag.t() 
    return res

def gen_sim_ed(data_name):
    start = time.time()
    trans_num = 5
    trainset = CMDataset(data_name,partition='train',num_transform=trans_num)       
    train_y = trainset.texts
    vgg19 = models.vgg19(pretrained=True)
    vgg19.classifier[6] = nn.Linear(4096, 4096)
    vgg19.to(torch.device("cuda"))
    vgg19.eval()
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=512,num_workers=10,shuffle=False,pin_memory=True,drop_last=False)
    features = torch.zeros(1,trans_num,4096)
    for idx, (imgs, txt, labels, index) in enumerate(train_loader):
        imgs =[img.cuda() for img in imgs]
        with torch.no_grad():
            feature_outputs = [vgg19(im) for im in imgs]
            feature_outputs = [item.cpu() for item in feature_outputs]
            feature = torch.cat([fi.unsqueeze(1) for fi in feature_outputs], dim=1)
            features = torch.cat([features,feature],dim=0)
    vgg19 = vgg19.to('cpu')
    torch.cuda.empty_cache()  
    train_x = features[1:]
    assert train_x.shape[0] == train_y.shape[0]
    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    
    print('train_x', train_x.shape) #train_x: [10000,5,4096]
    print('train_y', train_y.shape)
    
    D_I_ed = cal_ed(train_x) 

    F_I =  F.normalize(train_x.view(train_x.shape[0],-1))
    S_I = torch.matmul(F_I,F_I.t())
    F_T = F.normalize(train_y)
    S_T = torch.matmul(F_T,F_T.t())

    mt = 0.01
    tm = 0.2
    #im = 1.3
    
    if data_name.lower() == 'mirflickr25k':
        im=1.25
    elif data_name.lower() == 'nus_wide_tc10':
        im=0.5
    elif data_name.lower() == 'iapr':
        im=1.0
    elif data_name.lower() == 'mscoco':
        im=1.15

    sel_ed = (S_T > tm) * (D_I_ed < im) * 1.
    S_ed = sel_ed + (1 - sel_ed) * (S_I * (1 - mt) + S_T * mt)
    
    end = time.time()
    print('time', end-start)
    return S_ed


def gen_sim_ed2(data_name):
    start = time.time()
    trans_num = 5
    trainset = CMDataset(data_name,partition='train',num_transform=trans_num)       
    train_y = trainset.texts
    vgg19 = models.vgg19(pretrained=True)
    vgg19.classifier[6] = nn.Linear(4096, 4096)
    vgg19.to(torch.device("cuda"))
    vgg19.eval()
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=512,num_workers=10,shuffle=False,pin_memory=True,drop_last=False)
    features = torch.zeros(1,trans_num,4096)
    for idx, (imgs, txt, labels, index) in enumerate(train_loader):
        imgs =[img.cuda() for img in imgs]
        with torch.no_grad():
            feature_outputs = [vgg19(im) for im in imgs]
            feature_outputs = [item.cpu() for item in feature_outputs]
            feature = torch.cat([fi.unsqueeze(1) for fi in feature_outputs], dim=1)
            features = torch.cat([features,feature],dim=0)
    vgg19 = vgg19.to('cpu')
    torch.cuda.empty_cache()  
    train_x = features[1:]
    assert train_x.shape[0] == train_y.shape[0]
    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    
    print('train_x', train_x.shape) #train_x: [10000,5,4096]
    print('train_y', train_y.shape)
    
    D_I_ed = cal_ed(train_x) 

    F_I =  F.normalize(train_x.view(train_x.shape[0],-1))
    S_I = torch.matmul(F_I,F_I.t())
    F_T = F.normalize(train_y)
    S_T = torch.matmul(F_T,F_T.t())

    mt = 0.01
    tm = 0.2
    
    if data_name.lower() == 'mirflickr25k':
        im=1.25
    elif data_name.lower() == 'nus_wide_tc10':
        im=0.5
    elif data_name.lower() == 'iapr':
        im=1.0
    elif data_name.lower() == 'mscoco':
        im=1.15

    sel = (S_T > tm) * (S_I > im) * 1.
    S_ = sel + (1 - sel) * (S_I * (1 - mt) + S_T * mt)
    
    end = time.time()
    print('time', end-start)
    return S_

if __name__ == '__main__':
    
    # data_name = 'mirflickr25k'
    # s_ed2 = gen_sim_ed2(data_name)
    # with open('./similarity/vgg19_sim_ed3_mirflickr25k.pkl', 'wb') as f:
    #     pickle.dump(s_ed2, f)

    # data_name = 'nus_wide_tc10'
    # s_ed = gen_sim_ed2(data_name)
    # with open('./similarity/vgg19_sim_ed3_nus_wide_tc10.pkl', 'wb') as f:
    #     pickle.dump(s_ed, f)
        
    # data_name = 'iapr'
    # s_ed = gen_sim_ed2(data_name)
    # with open('./similarity/vgg19_sim_ed3_iapr.pkl', 'wb') as f:
    #     pickle.dump(s_ed, f)

    data_name = 'mscoco'
    s_ed = gen_sim_ed2(data_name)
    with open('./similarity/vgg19_sim_ed3_mscoco.pkl', 'wb') as f:
        pickle.dump(s_ed, f)

    print('finish gen similarity')

