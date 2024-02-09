import os,random
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.combine_bg import combine_hands
def get_train_image_list(dataset_path,translate_class2num):
    # get image list
    img_path_list = []
    img_classes_list = []
    skiped_list = []
    classes=[name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    for class_ in classes:
        if not os.path.isdir(os.path.join(dataset_path ,class_)):
            continue
        for img in os.listdir(os.path.join(dataset_path ,class_)):
            if (img[-3:] != "jpg") and (img[-3:] != "png"):
                skiped_list.append(img)
                continue
            img_path = os.path.join(dataset_path, class_ , img)
            img_path_list.append(img_path)
            class_name = class_.split('.iam')[0].split('.ipt')[0]
            if '-' in class_name:
                class_name = class_name.split('-')[1]
            img_classes_list.append(translate_class2num[class_name])

    return img_path_list,img_classes_list
def get_class2num(path):
    """
    get part class2num dict

    Args:
        path : dataset path

    Returns:
        class2num: part class2num dict
    """

    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    model_list.sort()
    class2num = {}
    for idx, item in enumerate(model_list):
        class_name = item.split('.iam')[0].split('.ipt')[0].split('-A')[0]
        if '-' in class_name:
            class_name=class_name.split('-')[1]
        class2num[class_name] = idx
    return class2num
def get_num2class(path):
    """
    get part class2num dict

    Args:
        path : dataset path

    Returns:
        class2num: part class2num dict
    """
    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    model_list.sort()
    num2class = {}
    for idx, item in enumerate(model_list):
        class_name = item.split('.iam')[0].split('.ipt')[0].split('-A')[0]  # .split('-')[1]
        if '-' in class_name:
            class_name = class_name.split('-')[1]
        num2class[idx] = class_name
    return num2class

def build_loader(args,is_train_aug=False,add_hands=False):
    class2num = get_class2num(args.train_root)
    num_classes = len(class2num)
    print("[dataset] class number:", num_classes)
    #split train valid data
    print(args.train_root)
    original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
    print(len(original_img_path_list))
    x_train, x_valid, y_train, y_valid = train_test_split(original_img_path_list, original_img_classes_list,
                                                          test_size=0.2,
                                                          random_state=32, stratify=original_img_classes_list,
                                                          shuffle=True)


    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, img_list=x_train,label_list=y_train, data_size=args.data_size, return_index=True)
        if is_train_aug:
            if add_hands:
                # create hands dataset
                if not os.path.isdir(os.path.join(args.combine_hands_path)):
                    print(f'create hand data...')
                    combine_hands(x_train)
                else:
                    print(f'hand data exist...')
                hands_img_path_list, hands_img_classes_list = get_train_image_list(os.path.join(args.combine_hands_path), class2num)
                train_aug_set = ImageDataset(istrain=True, img_list=hands_img_path_list, label_list=hands_img_classes_list,
                                             data_size=args.data_size,
                                             return_index=True, is_train_aug=is_train_aug)
                print(f'use hand data,{len(hands_img_path_list)}')
            else:
                train_aug_set = ImageDataset(istrain=True, img_list=x_train, label_list=y_train, data_size=args.data_size,
                                     return_index=True,is_train_aug=is_train_aug)
            train_set = torch.utils.data.ConcatDataset([train_set, train_aug_set])
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, img_list=x_valid,label_list=y_valid, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, img_list=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 img_list,
                 label_list,
                 data_size: int,
                 return_index: bool = False,
                 is_train_aug=False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.img_list = img_list
        self.label_list=label_list
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        if is_train_aug:
            self.transforms=A.Compose(
            [
                A.Resize(data_size, data_size),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.7,-0.3), rotate_limit=45,border_mode=cv2.BORDER_CONSTANT,value=0,p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,p=0.5),
                A.GaussianBlur (blur_limit=(3, 7),p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        elif istrain:
            self.transforms=A.Compose(
            [
                A.Resize(data_size, data_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        else:
            self.transforms = A.Compose(
                [
                    A.Resize(data_size, data_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

        """ read all data information """
        data_infos=[]
        for i,img_path in enumerate(self.img_list):
            data_infos.append({"path": img_path, "label": label_list[i]})
        self.data_infos = data_infos


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = os.path.join(root+folder,file)
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        random.seed(index)

        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img=cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img = img[:, :, ::-1] # BGR to RGB.
        img = self.transforms(image=img)['image']
        
        if self.return_index:
            return index, img, label
        return img, label
