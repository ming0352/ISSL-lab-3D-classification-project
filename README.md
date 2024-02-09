# 3D model classification project
# 1. Environment setting
## 1.1 Requirement
The project enviroment has createded by Anaconda,and you can create enviroment use requirements.txt.
```
conda install --file requirements.yml
``` 

## 1.2 Dataset
We use 3d model images as oue dataset,and 
train data is place in dataset folder
## 1.3 IDE
You can open this project by using Pycharm

# 2. Train

## 2.1 How to train
If you want to train model,you can just run main.py.

## 2.2 training setting
Training setting is in ./configs/config.yaml, you can change any setting if you want.
### 2.2.1 using hands dataset as our augmentation dataset
If you want to use hands dataset, you can edit config.yaml file with following setting.
 
```
#config.yaml

add_hands: True
is_train_aug : True
combine_hands_path : ./combine_hands/ 
```

## 2.3 Wandb
If you want to record training information, you can edit ./configs/config.yaml file by setting your wandb username
 
```
#config.yaml
use_wandb: True
wandb_entity: <your wandb username>
```

## 2.4 Save Model
The model will save in ./records folder


# 3. Test

If you want to test model , you can just run multiple_image_run_evaluation.py

 ### 3.1 Change infer images numbers
You can choose how many image you  want to infer once by setting input_num .Now support one or three images.

 ### 3.2 save result
 You can enable <is_save_summary> to save test result.

# 4. Visualize result

#### 4.1 How to visualze result
If you want to visualize model result, you can run vis_swin_l.py , remember to change test model path and test data folder
#### 4.2 change save result folder
The result will save in pretrained model folder(ex:/record/project name/...),you can change folder name by modify <save_folder_name> variable.