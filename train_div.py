import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, p_topK, calculate_map_1
from models import ImgNet, TxtNet
import os.path as osp
import torchvision.transforms as transforms
# from load_data import get_loader, get_loader_wiki
import numpy as np
import pdb
import time
import logging
import pickle
import random
import cv2
import scipy.io as scio
import h5py
import pickle
import os
import argparse
import logging
from cmhdataset import CMDataset
#from gen_sim import gen_sim_cos

def sharpen(p, T=0.25):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

class KLD(nn.Module):

    def forward(self, targets, inputs):
        targets = F.softmax(targets, dim=-1)
        inputs = F.log_softmax(inputs, dim=-1)
        return F.kl_div(inputs, targets, reduction='batchmean')


class DistributionAlignLoss(nn.Module):

    def __init__(self,temp = 0.25):
        super(DistributionAlignLoss, self).__init__()
        self.temp = temp

    def forward(self, x, y):
        assert x.shape[0]==y.shape[0]
  
        sim1 = x.mm(y.t())
        sim11 = (sim1 / self.temp).exp()
        sim1 = sharpen(sim1).exp()
        distribution1 = (sim11 / sim1.sum(1))
        
        sim2 = y.mm(x.t())
        sim21 = (sim2 / self.temp).exp()
        sim2 = sharpen(sim2).exp()

        distribution2 = (sim21 / sim2.sum(1))
        kld = KLD()
        score1 = kld(distribution1,distribution2)
        score2 = kld(distribution2,distribution1)

        return 0.5*(score1 + score2)
    
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
    pro_dis = pro_dis * int(0.4*batch_size)
    S_i = (S_pair * (1 - 0.3) + pro_dis * 0.3)
    return S_i

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def calc_dis(query_L, retrieval_L, query_dis, top_k=32):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = query_dis[iter]
        ind = np.argsort(hamm)[:top_k]
        gnd = gnd[ind]
        tsum = np.int32(np.sum(gnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


class Session:
    def __init__(self, opt):
        self.opt = opt

        self.data_set = opt.data_name
        ### data processing

        self.train_dataset = CMDataset(
            opt.data_name,
            partition='train',
            num_transform = 1
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )

        self.retrieval_dataset = CMDataset(
            opt.data_name,
            partition='retrieval',
            num_transform = 1
        )
        self.retrieval_loader = torch.utils.data.DataLoader(
            self.retrieval_dataset,
            batch_size=opt.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=False
        )

        self.test_dataset = CMDataset(
            opt.data_name,
            partition='test',
            num_transform = 1
        )
        self.query_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=opt.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=False
        )

        print('train set: %d, retrieval set: %d, test set: %d' % (len(self.train_dataset),len(self.retrieval_dataset),len(self.test_dataset)))
    
        self.image_codes = torch.rand((len(self.train_dataset), opt.bit)).float()
        self.text_codes = torch.rand((len(self.train_dataset), opt.bit)).float()

        print('generating similarity started')
        start_time = time.time()
        

        if self.opt.sim_type == 'ed':
            file_path = './similarity/vgg19_sim_ed3_{}.pkl'.format(self.opt.data_name)
        else:
            raise ValueError('Wrong simlarity type, must be cos or swd or ed !!')
        
        with open(file_path, 'rb') as f:
            gs2 = pickle.load(f).detach().cpu()
        
        if self.opt.cal_sim:
            gs2 = cal_similarity(gs2)
            print('cal_similarity !')
        else:
            print('Without cal_similarity !')
        gs2 = 2.0 * gs2 - 1.
        self.gs = gs2
        end_time = time.time()
        print('generating similarity fnished, Time: ',end_time - start_time)
        
        self.CodeNet_I = ImgNet( opt.bit)
        self.CodeNet_T = TxtNet(opt.bit, self.train_dataset.text_dim)

        if self.opt.same_lr:
             self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay)
        else:
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.features.parameters(), 'lr': opt.learning_rate * 0.05},
            {'params': self.CodeNet_I.classifier.parameters(), 'lr': opt.learning_rate * 0.05},
            {'params': self.CodeNet_I.hashlayer.parameters(), 'lr': opt.learning_rate}], lr=opt.learning_rate, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)
        
        # self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(),  lr=opt.learning_rate, momentum=opt.momentum,
        #                              weight_decay=opt.weight_decay)
        # self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
        #                              weight_decay=opt.weight_decay)

        # self.scheduler_I = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_I, T_max=opt.num_epochs)
        # self.scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_T, T_max=opt.num_epochs)
        
        self.scheduler_I = torch.optim.lr_scheduler.MultiStepLR(self.opt_I, [15, 30, 45], gamma=0.1)
        self.scheduler_T = torch.optim.lr_scheduler.MultiStepLR(self.opt_T, [15, 30, 45], gamma=0.1)
        
        self.best = 0
        self.best_i2t = 0
        self.best_t2i = 0
        # pdb.set_trace()
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        stream_log = logging.StreamHandler()
        stream_log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_log.setFormatter(formatter)
        logger.addHandler(stream_log)
        self.logger = logger
    def loss_denoise(self, code_sim, S, weight_s):
        loss = (weight_s * (code_sim - S).pow(2)).mean()
        return loss
    def loss_cal(self, code_I, code_T, S, weight_s, I):
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())

        #loss_pair = torch.tensor(0.0)
        #loss_cons = torch.tensor(0.0)
       

        diagonal = BI_BT.diagonal()
        all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
        loss_pair = F.mse_loss(diagonal, 1.5 * all_1)

        loss_dis_1 = self.loss_denoise(BT_BT * (1 - I), S * (1 - I), weight_s)
        loss_dis_2 = self.loss_denoise(BI_BT * (1 - I), S * (1 - I), weight_s)
        loss_dis_3 = self.loss_denoise(BI_BI * (1 - I), S * (1 - I), weight_s)
        
        # loss_dis_1 = self.loss_denoise(BT_BT , S , weight_s)
        # loss_dis_2 = self.loss_denoise(BI_BT , S , weight_s)
        # loss_dis_3 = self.loss_denoise(BI_BI , S , weight_s)

        loss_cons = F.mse_loss(BI_BT, BI_BI) + \
                    F.mse_loss(BI_BT, BT_BT) + \
                    F.mse_loss(BI_BI, BT_BT) + \
                    F.mse_loss(BI_BT, BI_BT.t())

        DAL = DistributionAlignLoss()
        loss_dal = DAL(B_I,B_T)
        #loss_dal = torch.tensor(0.0)
        
        loss = loss_pair + (loss_dis_1 + loss_dis_2 + loss_dis_3) * self.opt.dw \
               + loss_cons * self.opt.cw  \
               + loss_dal * 0.001


        return loss, (loss_dal,loss_pair, loss_dis_1, loss_dis_2, loss_dis_3, loss_cons, loss_cons)


    def train(self, epoch, start_denoise=False):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        top_mAP = 0.0
        num = 0.0
        # self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
        #     epoch + 1, self.opt.num_epochs, self.CodeNet_I.alpha, self.CodeNet_T.alpha))
        for idx, (img, txt, labels, index) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

            batch_size = img.size(0)
            I = torch.eye(batch_size).cuda()


            code_I = self.CodeNet_I(img)
            code_T = self.CodeNet_T(txt)

            S0 = self.gs[index, :][:, index].cuda()
            if start_denoise:
                B_I = F.normalize(code_I)
                B_T = F.normalize(code_T)
                code_sim = (B_I.mm(B_T.t()) + B_I.mm(B_I.t()) + B_T.mm(B_T.t())) / 3.
                select_pos = (torch.abs(code_sim - S0) > self.opt.gamma) * 1.
                I = torch.eye(code_sim.size(0)).cuda()

                weight_s = (1 - torch.abs(S0)) * select_pos + 1 - select_pos

                S = (1 - select_pos) * S0 + select_pos * torch.sign(code_sim - S0)
                S = S * (1 - I) + I
            else:
                weight_s = 1.
                S = S0

            loss0, all_los0 = self.loss_cal(code_I, code_T, S.detach(), weight_s, I)
            loss = loss0
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            loss.backward(retain_graph=True)
            self.opt_I.step()
            self.opt_T.step()

            loss1, loss2, loss3, loss4, loss5, loss6, loss7 = all_los0

            top_mAP += calc_dis(labels.cpu().numpy(), labels.cpu().numpy(), -S0.cpu().numpy())

            num += 1.
            if (idx + 1) % (len(self.train_loader)) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss da : %.4f '
                    'code T: %.4f code I: %.4f '
                    'Total Loss: %.4f '
                    'mAP: %.4f'
                    % (
                        epoch + 1, self.opt.num_epochs, idx + 1,
                        len(self.train_loader) // self.opt.batch_size,
                        loss1.item(),
                        code_T.abs().mean().item(),
                        code_I.abs().mean().item(),
                        loss.item(),
                        top_mAP / num))
                
        self.scheduler_I.step()
        self.scheduler_T.step()
        
    def eval(self, step=0, num_epoch=0, last=False, adapt=False):
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        if self.opt.EVAL == False:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.retrieval_loader, self.query_loader, self.CodeNet_I,
                                                              self.CodeNet_T)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            self.logger.info('--------------------------------------------------------------------')
            print('--------------------Evaluation1: Calculate top MAP-------------------')
            print('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            print('--------------------------------------------------------------------')
            
            if MAP_I2Ta + MAP_T2Ia > self.best:
                num_epoch = 0

                if not adapt:
                    self.save_checkpoints(step=step, best=True)
                self.best = MAP_T2Ia + MAP_I2Ta
                self.best_i2t = MAP_I2Ta
                self.best_t2i = MAP_T2Ia
                self.logger.info("#########is best:%.3f #########" % ((self.best)/2))
            else:
                num_epoch += 1
        if self.opt.EVAL:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.retrieval_loader, self.query_loader, self.CodeNet_I,
                                                              self.CodeNet_T)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation1: Calculate top MAP-------------------')
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            self.logger.info('--------------------------------------------------------------------')
            print('--------------------Evaluation1: Calculate top MAP-------------------')
            print('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            print('--------------------------------------------------------------------')
            res = [MAP_I2Ta, MAP_T2Ia, (MAP_I2Ta+MAP_T2Ia)/2]

                
        return num_epoch ,res

    def save_checkpoints(self, step, path='',
                         best=False):
        ckp_path = path + '/model_ckp_ablation_text_' + self.data_set + '/our_model_bit_'+str(int(self.opt.bit)) + '.pth'
        if not os.path.exists(path + '/model_ckp_ablation_text_' + self.data_set):
            os.makedirs(path + '/model_ckp_ablation_text_' + self.data_set)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, path=''):
        ckp_path = path + '/model_ckp_ablation_text_' + self.data_set + '/our_model_bit_'+str(int(self.opt.bit)) + '.pth'
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])





