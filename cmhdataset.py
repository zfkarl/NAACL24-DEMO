# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import h5py
import cv2
logger = getLogger()

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
    
class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)

def text_transform(text):
    return text

def load_mat(img_mat_url, tag_mat_url, label_mat_url):
    img_names = sio.loadmat(img_mat_url)['imgs']  # type: np.ndarray
    img_names = img_names.squeeze()
    all_img_names = img_names
    all_txt = np.array(sio.loadmat(tag_mat_url)['tags'], dtype=np.float32)
    all_label = np.array(sio.loadmat(label_mat_url)['labels'], dtype=np.float32)
    return all_img_names, all_txt, all_label

class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        data_name,
        return_index=False,
        partition='train',
        num_transform = 1
    ):
        self.data_name = data_name
        self.partition = partition
        training = 'train' in partition.lower()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        train_transforms = transforms.Compose([
                                        transforms.ToPILImage(), 
                                        transforms.Resize((256, 256)),
                                        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.7),
                                        transforms.RandomGrayscale(p=0.2),
                                        GaussianBlur(3),    
                                        transforms.ToTensor(),         
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
        transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_transforms_coco = transforms.Compose([ 
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.7),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(3),    
            transforms.ToTensor(),         
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        transformations_coco = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if training:
            if self.data_name.lower()=='mscoco':
                trans = train_transforms_coco
            else:
                trans = train_transforms 
            # trans = transforms.Compose([
            #         transforms.ToTensor(),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.RandomResizedCrop(224),
            #         # transforms.CenterCrop(224),
            #         transforms.Normalize(mean=mean, std=std)])
            
        else:
            if self.data_name.lower()=='mscoco':
                trans = transformations_coco
            else:
                trans = transformations
            # trans = transforms.Compose([
            #         transforms.ToTensor(),
            #         # transforms.Resize(256),
            #         transforms.CenterCrop(224),
            #         transforms.Normalize(mean=mean, std=std)])
    
        self.trans = trans
        self.num_transform = num_transform
        self.return_index = return_index
        self.open_data()

    
    def open_data(self):
        # mirflickr25k mirflickr25k_fea MSCOCO_fea nus_wide_tc10_fea IAPR-TC12_fea
        if self.data_name.lower() == 'mirflickr25k':
            data = MIRFlickr25K(self.partition)
        elif self.data_name.lower() == 'mirflickr25k_1k':
            data = MIRFlickr25K_1k(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc10':
            data = NUSWIDETC10(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc21':
            data = NUSWIDETC21(self.partition)
        elif self.data_name.lower() == 'iapr':
            data = IAPR(self.partition)
        elif self.data_name.lower() == 'mscoco':
            data = MSCOCO(self.partition)
            
        elif self.data_name.lower() == 'mirflickr25k_fea':
            data = MIRFlickr25K_fea(self.partition)
        elif self.data_name.lower() == 'iapr_fea':
            data = IAPR_fea(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc10_fea':
            data = NUSWIDETC10_fea(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc21':
            data = NUSWIDETC21_fea(self.partition)
        elif self.data_name.lower() == 'mscoco_fea':
            data = MSCOCO_fea(self.partition)

        print(self.data_name)
        
        self.data_size = len(data)
        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
            self.imgs = self.imgs
        else:
            (self.imgs, self.texts, self.labels, self.root) = data
            #self.imgs = Sampler(root, self.img_names)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]

    def read_img(self, item):
        image_url = os.path.join(self.root, self.imgs[item].strip())
        image = Image.open(image_url).convert('RGB')
        return image
    
    def __getitem__(self, index):
        if self.data_size ==3:
            image = self.imgs[index]
        else:
            #image = Image.open(os.path.join(self.root,self.imgs[index].strip())).convert('RGB')
            image =self.read_img(index)
            
        text = self.texts[index]
        label = self.labels[index]
        if self.num_transform > 1:
            if self.data_name.lower() in ['mirflickr25k','mirflickr25k_1k','nus_wide_tc10','nus_wide_tc21','iapr','mscoco']:
                multi_crops = [self.trans(image) for i in range(self.num_transform)]
        elif self.num_transform == 1:
            if self.data_name.lower() in ['mirflickr25k','mirflickr25k_1k','nus_wide_tc10','nus_wide_tc21','iapr','mscoco']:
                multi_crops = self.trans(image)
        else:
            multi_crops = image
        text = text
        
        if self.return_index:
            return index, multi_crops, text, label
        
        return multi_crops, text, label ,index
        # return multi_crops, text, index

    def __len__(self):
        return self.length

def MIRFlickr25K(partition):
    import h5py
    imgs = h5py.File('/home/zf/dataset/data_cmh/MIRFLICKR25K/IAll/mirflickr25k-iall.mat', mode='r')['IAll'][()]
    tags = sio.loadmat('/home/zf/dataset/data_cmh/MIRFLICKR25K/YAll/mirflickr25k-yall.mat')['YAll']
    labels = sio.loadmat('/home/zf/dataset/data_cmh/MIRFLICKR25K/LAll/mirflickr25k-lall.mat')['LAll']

    inx = np.arange(imgs.shape[0])
    np.random.seed(42)
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            imgs, tags, labels =imgs[0: train_size], tags[0: train_size], labels[0: train_size]
        
    return imgs.transpose(0,3,2,1), tags, labels


def MIRFlickr25K_1k(partition):
    import h5py
    imgs = h5py.File('/home/zf/dataset/data_cmh/MIRFLICKR25K/IAll/mirflickr25k-iall.mat', mode='r')['IAll'][()]
    tags = sio.loadmat('/home/zf/dataset/data_cmh/MIRFLICKR25K/YAll/mirflickr25k-yall.mat')['YAll']
    labels = sio.loadmat('/home/zf/dataset/data_cmh/MIRFLICKR25K/LAll/mirflickr25k-lall.mat')['LAll']

    inx = np.arange(imgs.shape[0])
    np.random.seed(42)
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 1000
            imgs, tags, labels =imgs[0: train_size], tags[0: train_size], labels[0: train_size]
        
    return imgs.transpose(0,3,2,1), tags, labels

def NUSWIDETC10(partition):
    imgs =  h5py.File("/home/zf/dataset/data_cmh/NUS-WIDE-TC10/IAll/IAll/nus-wide-tc10-iall.mat", mode='r')['IAll']
    print('transfer nus-wide-tc10 file to numpy array, very slow...')
    imgs = np.asarray(imgs).transpose(0,3,2,1)
    print('transfer finished')
    tags = h5py.File('/home/zf/dataset/data_cmh/NUS-WIDE-TC10/nus-wide-tc10-yall.mat', mode='r')['YAll']
    tags = np.asarray(tags).transpose(1,0)
    labels = sio.loadmat('/home/zf/dataset/data_cmh/NUS-WIDE-TC10/nus-wide-tc10-lall.mat')['LAll']

    inx = np.arange(labels.shape[0])
    np.random.seed(42)
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
 
    test_size = 2100
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            imgs, tags, labels =imgs[0: train_size], tags[0: train_size], labels[0: train_size]
            
    return imgs, tags, labels

def NUSWIDETC21(partition):
    imgs =  h5py.File("/home/zf/dataset/data_cmh/NUS-WIDE-TC21/IAll/IAll/nus-wide-tc21-iall.mat", mode='r')['IAll']
    print('transfer nus-wide-tc21 file to numpy array, very slow...')
    imgs = np.asarray(imgs).transpose(0,3,2,1)
    print('transfer finished')
    tags = sio.loadmat('/home/zf/dataset/data_cmh/NUS-WIDE-TC21/nus-wide-tc21-yall.mat')['YAll']
    labels = sio.loadmat('/home/zf/dataset/data_cmh/NUS-WIDE-TC21/nus-wide-tc21-lall.mat')['LAll']

    inx = np.arange(labels.shape[0])
    np.random.seed(42)
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
 
    test_size = 5000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            imgs, tags, labels =imgs[0: train_size], tags[0: train_size], labels[0: train_size]
            
    return imgs, tags, labels

def MSCOCO(partition):
    abs_dir = "/home/zf/CMR/img_text_baselines/CMH/deep-cross-modal-hashing/torchcmh/dataset/"
    root = "/home/zf/dataset/data_gnn4cmr/MS-COCO/coco_raw_imgs/"
    
    default_img_mat_url = os.path.join(abs_dir, "data", "coco2014", "imgList.mat")
    default_tag_mat_url = os.path.join(abs_dir, "data", "coco2014", "tagList.mat")
    default_label_mat_url = os.path.join(abs_dir, "data", "coco2014", "labelList.mat")
    
    img_names, tags, labels = load_mat(img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url)
    
    inx = np.arange(tags.shape[0])
    
    np.random.seed(42)
    np.random.shuffle(inx)
    img_names, tags, labels = img_names[inx], tags[inx], labels[inx]
    test_size = 5000
    if 'test' in partition.lower():
        img_names, tags, labels = img_names[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        img_names ,tags, labels = img_names[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            img_names, tags, labels = img_names[0: train_size], tags[0: train_size], labels[0: train_size]
            
    return img_names, tags, labels, root


def IAPR(partition):
    iaprtc = h5py.File(os.path.join('/home/zf/dataset/data_cmh/','iaprtc.h5'), 'r')
    database_x,database_y, database_l = iaprtc['data_set'],iaprtc['dataset_y'],iaprtc['dataset_L']
    test_x,test_y,test_l = iaprtc['test_data'],iaprtc['test_y'],iaprtc['test_L']
    imgs, tags, labels = np.concatenate([database_x,test_x]),np.concatenate([database_y,test_y]),np.concatenate([database_l,test_l])

    inx = np.arange(tags.shape[0])
    
    np.random.seed(42)
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs ,tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            #print('train size:',train_size)
            imgs, tags, labels =imgs[0: train_size], tags[0: train_size], labels[0: train_size]
            
    return imgs, tags, labels

def MIRFlickr25K_fea(partition):
    root = '/home/zf/dataset/UCCH_data/MIRFLICKR25K/'
    data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']

    test_size = 2000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
            
    return data_img, data_txt, labels

def IAPR_fea(partition):
    root = '/home/zf/dataset/UCCH_data/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)

    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']

    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']

    data_img, data_txt, labels = np.concatenate([valid_img, test_img]), np.concatenate([valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])

    test_size = 2000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
            
    return data_img, data_txt, labels

def NUSWIDETC10_fea(partition):
    root = '/home/zf/dataset/UCCH_data/NUS-WIDE-TC10/'
    test_size = 2100
    data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll']
    data_txt = sio.loadmat(root + 'nus-wide-tc10-yall.mat')['YAll']
    labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']

    test_size = 2100
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
            
    return data_img, data_txt, labels

def NUSWIDETC21_fea(partition):
    root = '/home/zf/dataset/data_cmh/NUS-WIDE-TC21/'
    test_size = 5000
    data_img = sio.loadmat(root + 'nus-wide-tc21-xall-vgg.mat')['XAll'] ## 500 dim, not 4096
    data_txt = sio.loadmat(root + 'nus-wide-tc21-yall.mat')['YAll']
    labels = sio.loadmat(root + 'nus-wide-tc21-lall.mat')['LAll']

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
            
    return data_img, data_txt, labels

def MSCOCO_fea(partition):
    root = '/home/zf/dataset/UCCH_data/'
    path = root + 'MSCOCO_deep_doc2vec_data_rand.h5py'
    data = h5py.File(path)
    data_img = data['XAll'][()]
    data_txt = data['YAll'][()]
    labels = data['LAll'][()]

    test_size = 5000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
        if 'train' in partition.lower(): 
            train_size = 10000
            data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
            
    return data_img, data_txt, labels


if __name__ == '__main__':
    # imgs, tags, labels = NUSWIDE('train')
    # print(imgs.shape,tags.shape,labels.shape)
    import torch
    train_dataset = CMDataset(
        'nus_wide_tc21',
        partition='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1024,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    for img, txt, labels,index in train_loader:
        print(len(img),img.shape,txt.shape,labels.shape)
        
    print(len(train_dataset))
