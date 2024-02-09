import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
def to_image(x):
    return Image.fromarray(x)
def to_np(x):
    return np.asarray(x)
class test_Dataset(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    transforms.Lambda(to_image),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Lambda(to_image),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        # if self.pretrain:
        #     self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
        # else:
        #     self.file_pattern = 'miniImageNet_category_split_%s.pickle'
        # self.data = {}
        # with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     self.imgs = data['data']
        #     self.labels = data['labels']

        # pre-process for contrastive sampling
        self.k = k
        # self.is_sample = is_sample
        # if self.is_sample:
        #     self.labels = np.asarray(self.labels)
        #     self.labels = self.labels - np.min(self.labels)
        #     num_classes = np.max(self.labels) + 1
        #
        #     self.cls_positive = [[] for _ in range(num_classes)]
        #     for i in range(len(self.imgs)):
        #         self.cls_positive[self.labels[i]].append(i)
        #
        #     self.cls_negative = [[] for _ in range(num_classes)]
        #     for i in range(num_classes):
        #         for j in range(num_classes):
        #             if j == i:
        #                 continue
        #             self.cls_negative[i].extend(self.cls_positive[j])
        #
        #     self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        #     self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        #     self.cls_positive = np.asarray(self.cls_positive)
        #     self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        # img=Image.fromarray(img)
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class MVImgNet_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 istrain: bool,
                 img_list,
                 label_list,
                 data_size: int,
                 return_index: bool = False,
                 is_train_aug=False,
                 is_save=False,
                 save_folder=None,
                 filename=None,
                 transform=None,
                 isgrayscale=False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.img_list = img_list
        self.label_list=label_list
        self.data_size = data_size
        self.return_index = return_index
        self.is_save=is_save
        self.is_train_aug=is_train_aug
        self.save_folder=save_folder
        self.filename=filename
        self.transforms=transform
        """ declare data augmentation """
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.istrain=istrain
        self.isgrayscale=isgrayscale
        if self.transforms is None:
            if self.istrain: #self.partition == 'train' and self.data_aug:
                self.transforms = transforms.Compose([
                    # transforms.Lambda(to_image),
                    transforms.Resize([384, 384]),
                    #transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize([384, 384]),
                    # transforms.Lambda(to_image),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transforms = transform

        """ read all data information """
        data_infos=[]
        for i,img_path in enumerate(self.img_list):
            data_infos.append({"path": img_path, "label": label_list[i]})
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)
    # def save_image(self,image,index):
    #     """
    #     save image
    #
    #     Args:
    #         image : image file
    #         index : index of image
    #     """
    #     save_path=os.path.join(self.save_folder)
    #     os.makedirs(name=save_path, mode=0o777, exist_ok=True)
    #     save_image(image, os.path.join(save_path,f'{self.filename}_{index}.jpg'),normalize=True)

    def __getitem__(self, index):

        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]

        # img=cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # img = img[:, :, ::-1] # BGR to RGB.
        img = Image.open(image_path)
        if self.isgrayscale:
            img=img.convert('L')
        img = np.asarray(img)
        if self.isgrayscale:
            img = np.repeat(img[..., np.newaxis], 3, -1)
        img=Image.fromarray(img)
        img = self.transforms(img)

        if self.is_save:
            self.save_image(img, index)

        if self.return_index:
            return index, img, label

        return img, label,index