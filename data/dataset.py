import os,random
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.combine_bg import combine_hands
from sklearn.model_selection import KFold, StratifiedKFold
from torchvision.utils import save_image
def get_train_image_list(dataset_path,translate_class2num):
    # get image list
    img_path_list = []
    img_classes_list = []
    skiped_list = []
    #classes = translate_class2num.keys()
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
            class_name = class_ #class_.split('.iam')[0].split('.ipt')[0]
            # if '-' in class_name:
            #     class_name = class_name.split('-')[1]
            img_classes_list.append(translate_class2num[class_name])

    return img_path_list,img_classes_list
def get_MVImgNet_train_image_list(dataset_path,translate_class2num):
    # get image list
    img_path_list = []
    img_classes_list = []
    skiped_list = []
    # classes = translate_class2num.keys()
    classes = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    for class_ in classes:
        if not os.path.isdir(os.path.join(dataset_path, class_)):
            continue
        for img in os.listdir(os.path.join(dataset_path, class_,'images')):
            if (img[-3:] != "jpg") and (img[-3:] != "png"):
                skiped_list.append(img)
                continue
            img_path = os.path.join(dataset_path, class_,'images', img)
            img_path_list.append(img_path)
            class_name = class_
            img_classes_list.append(translate_class2num[class_name])
    return img_path_list, img_classes_list
def get_MVImgNet_train_image_list_ori(dataset_path,translate_class2num):
    # get image list
    img_path_list = []
    img_classes_list = []
    skiped_list = []
    #classes = translate_class2num.keys()
    classes=os.listdir(dataset_path)
    classes.sort()
    for class_ in classes:
        if not os.path.isdir(os.path.join(dataset_path ,class_)):
            continue
        second_folder_list=os.listdir(os.path.join(dataset_path, class_))
        second_folder_list.sort()
        for k,i in enumerate(second_folder_list):
            if k==3: break
            for img in os.listdir(os.path.join(dataset_path ,class_,i,'images')):
                if (img[-3:] != "jpg") and (img[-3:] != "png"):
                    skiped_list.append(img)
                    continue
                img_path = os.path.join(dataset_path, class_ ,i,'images', img)
                img_path_list.append(img_path)
                class_name = i #class_.split('.iam')[0].split('.ipt')[0]
                # if '-' in class_name:
                #     class_name = class_name.split('-')[1]
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

    #model_list = os.listdir(path)
    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    model_list.sort()
    class2num = {}
    for idx, item in enumerate(model_list):
        class_name = item  # class_.split('.iam')[0].split('.ipt')[0]
        # if '-' in class_name:
        #     class_name = class_name.split('-')[1]
        class2num[class_name] = idx
    return class2num
def get_class2num_mv(path):
    """
    get part class2num dict

    Args:
        path : dataset path

    Returns:
        class2num: part class2num dict
    """

    #model_list = os.listdir(path)
    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    model_list.sort()
    class2num = {}
    for i, item in enumerate(model_list):
        second_list = [name for name in os.listdir(os.path.join(path,item)) if os.path.isdir(os.path.join(path,item, name))]
        second_list.sort()
        for idx, item in enumerate(second_list):
            if idx == 3: break
            class_name = item  # class_.split('.iam')[0].split('.ipt')[0]
            # if '-' in class_name:
            #     class_name = class_name.split('-')[1]
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

    #model_list = os.listdir(path)
    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    model_list.sort()
    num2class = {}
    for idx, item in enumerate(model_list):
        class_name = item  # class_.split('.iam')[0].split('.ipt')[0]
        # if '-' in class_name:
        #     class_name = class_name.split('-')[1]
        num2class[idx] = class_name
    return num2class
def build_data_list(args,):
    class2num = get_class2num(args.train_root)
    num_classes = len(class2num)
    print("[dataset] class number:", num_classes)
    # split train valid data
    print(args.train_root)
    original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
    print(len(original_img_path_list))
    if args.is_using_cross_validation:
        pass
        skf = StratifiedKFold(n_splits=3)
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(original_img_path_list, original_img_classes_list,
                                                              test_size=0.2,
                                                              random_state=32, stratify=original_img_classes_list,)
    return x_train, x_valid, y_train, y_valid
def build_loader(args,is_train_aug=False,add_hands=False,fold_img_label=None):
    if args.isgrayscale==True:
        print('HERBS dataset is grayscale')
    else:
        print('HERBS dataset is RGB')
    class2num = get_class2num(args.train_root)
    num_classes = len(class2num)
    original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
    if fold_img_label == None :
        print("[dataset] class number:", num_classes)
        # #split train valid data
        # print(args.train_root)
        print(len(original_img_path_list))
        x_train, x_valid, y_train, y_valid = train_test_split(original_img_path_list, original_img_classes_list,
                                                              test_size=0.2,
                                                              random_state=32, stratify=original_img_classes_list,
                                                              shuffle=True)
    else:
        print('using_cross_validation')
        print("[dataset] class number:", num_classes)
        print(len(original_img_path_list))
        x_train = [original_img_path_list[i] for i in fold_img_label[0]]
        y_train = [original_img_classes_list[i] for i in fold_img_label[0]]
        x_valid = [original_img_path_list[i] for i in fold_img_label[1]]
        y_valid = [original_img_classes_list[i] for i in fold_img_label[1]]


    train_set, train_loader = None, None
    if x_train is not None:
        train_set = ImageDataset(istrain=True, img_list=x_train,label_list=y_train, data_size=args.data_size, return_index=True,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='train_aug',filename='ori')
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
                                             return_index=True, is_train_aug=is_train_aug,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='train_aug',filename='aug')
                print(f'use hand data,{len(hands_img_path_list)}')
            else:
                train_aug_set = ImageDataset(istrain=True, img_list=x_train, label_list=y_train, data_size=args.data_size,
                                     return_index=True,is_train_aug=is_train_aug,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='train_aug',filename='aug')
            train_set = torch.utils.data.ConcatDataset([train_set, train_aug_set])
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if x_valid is not None:
        val_set = ImageDataset(istrain=False, img_list=x_valid,label_list=y_valid, data_size=args.data_size, return_index=True,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='valid',filename='valid')
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

# def get_dataset(args):
#     if args.train_root is not None:
#         train_set = ImageDataset(istrain=True, img_list=args.train_root, data_size=args.data_size, return_index=True)
#         return train_set
#     return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self,
                 istrain: bool,
                 img_list,
                 label_list,
                 data_size: int,
                 return_index: bool = False,
                 is_train_aug=False,
                 isgrayscale=False,
                 is_save=False,
                 save_folder=None,
                 filename=None):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.img_list = img_list
        self.label_list=label_list
        self.data_size = data_size
        self.return_index = return_index
        self.isgrayscale=isgrayscale
        self.is_save = is_save
        self.is_train_aug = is_train_aug
        self.save_folder = save_folder
        self.filename = filename
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
    def save_image(self,image,index):
        """
        save image

        Args:
            image : image file
            index : index of image
        """
        save_path=os.path.join(self.save_folder)
        os.makedirs(name=save_path, mode=0o777, exist_ok=True)
        save_image(image, os.path.join(save_path,f'{self.filename}_{index}.jpg'),normalize=True)
    def __getitem__(self, index):
        random.seed(index)

        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # =============================
        # read image by opencv.
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # isgrayscale = False
        if self.isgrayscale:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.repeat(img[..., np.newaxis], 3, -1)
        else:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, :, ::-1]  # BGR to RGB.

        img = self.transforms(image=img)['image']
        if self.is_save:
            self.save_image(img, index)
        if self.return_index:
            return index, img, label
        return img, label
