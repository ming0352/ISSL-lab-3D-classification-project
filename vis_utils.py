import cv2
import numpy as np
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImgLoader(object):

    def __init__(self, img_size: int,text=None):
        self.text=text
        self.img_size = img_size
        self.transform = A.Compose(
            [
                A.Resize(640, 640),
                A.CenterCrop(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),#mean=0.5, std=1),
                ToTensorV2(),
                ])

    def load(self, image_path: str):
        ori_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        assert ori_img.shape[2] == 3, "3(RGB) channels is required."
        img = copy.deepcopy(ori_img)
        img = img[:, :, ::-1] # convert BGR to RGB
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        t = A.Compose(
            [
                A.Resize(640, 640),
                A.CenterCrop(self.img_size, self.img_size),
            ])

        ori_img=t(image=ori_img)['image']
        return img, ori_img


def get_cdict():
    _jet_data = {
              # 'red':   ((0.00, 0, 0),
              #          (0.35, 0, 0),
              #          (0.66, 1, 1),
              #          (0.89, 1, 1),
              #          (1.00, 0.5, 0.5)),
              'red':   ((0.00, 0, 0),
                       (0.35, 0.5, 0.5),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.8, 0.8)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0.3, 0.3),
                       (1.000, 0, 0)),
             # 'blue':  ((0.00, 0.5, 0.5),
             #           (0.11, 1, 1),
             #           (0.34, 1, 1),
             #           (0.65, 0, 0),
             #           (1.00, 0, 0))}
             'blue':  ((0.00, 0.30, 0.30),
                       (0.25, 0.8, 0.8),
                       (0.34, 0.8, 0.8),
                       (0.65, 0, 0),
                       (1.00, 0, 0))
             }
    return _jet_data