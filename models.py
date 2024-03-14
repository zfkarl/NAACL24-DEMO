import math

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# class ImgNet(nn.Module):
#     def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         """
#         super(ImgNet, self).__init__()
#         self.module_name = "img_model"

#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#         #self.apply(weights_init)
#         self.norm = norm

#     def forward(self, x):
#         out = self.fc(x).tanh()
#         if self.norm:
#             norm_x = torch.norm(out, dim=1, keepdim=True)
#             out = out / norm_x
#         return out

# class TxtNet(nn.Module):
#     def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         """
#         super(TxtNet, self).__init__()
#         self.module_name = "txt_model"

#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#         #self.apply(weights_init)
#         self.norm = norm

#     def forward(self, x):
#         out = self.fc(x).tanh()
#         if self.norm:
#             norm_x = torch.norm(out, dim=1, keepdim=True)
#             out = out / norm_x
#         return out


class ImgNet(nn.Module):
    def __init__(self, code_len,  pretrained=True):
        super(ImgNet, self).__init__()
        original_model = models.vgg19(pretrained)
        self.features = original_model.features
        cl1 = nn.Linear(25088, 4096)

        cl2 = nn.Linear(4096, 4096)
        if pretrained:
            # ## vgg16
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias


        self.classifier = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.hashlayer = nn.Sequential(
            nn.Linear(4096, code_len)
        )
        torch.nn.init.normal(self.hashlayer[0].weight, mean=0.0, std=0.1)

        self.iter_num = 0
        self.step_size = 800000
        self.gamma = 0.000125
        self.power = 0.5
        self.init_scale = 1.0
        self.alpha = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), -1)

        hid = self.classifier(x)
        hid1 = self.hashlayer(hid)
        if self.iter_num % self.step_size == 0:
            self.alpha = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        code = torch.tanh(self.alpha * hid1)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len,txt_feat_len ):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        # torch.nn.init.normal(self.fc2.weight, mean=0.0, std= 0.3)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)
        self.iter_num = 0
        self.step_size = 800000
        self.gamma = 0.000125
        self.power = 0.5
        self.init_scale = 1.0
        self.alpha = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        feat = self.relu(self.fc1(x))
        # feat = self.relu(self.fc2(self.dropout(feat)))
        hid = self.fc_encode(self.dropout(feat))
        if self.iter_num % self.step_size == 0:
            self.alpha = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

