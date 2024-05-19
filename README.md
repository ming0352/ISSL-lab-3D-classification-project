# 3D model component classification project
# 1. Environment setting
### 1.1 Requirement
The project enviroment has createded by Anaconda,and you can create enviroment use requirements.txt.
```
conda install --file requirements.yml
``` 

### 1.2 Dataset
We use 3d model images as oue dataset,and 
train data is place in dataset folder
### 1.3 IDE
You can open this project by using Pycharm
# 2 Model
You can choose different model to train.Now only support ResNet18,ConvNext and HERBS. You can edit ./configs/HERBS_config.yaml file by setting model type.
```
#HERBS_config.yaml
model_type: HERBS #resnet18 convnext HERBS
```

# 3. Train

### 3.1 How to train
If you want to train model,you can just run main.py.

### 3.2 Training setting
Training setting is in ./configs/HERBS_config.yaml, you can change any setting if you want.


### 3.3 Wandb
If you want to record training information, you can edit ./configs/HERBS_config.yaml file by setting your wandb username
 
```
#HERBS_config.yaml
use_wandb: True
wandb_entity: <your wandb username>
```
### 3.4 Cross validation
If you want to training with cross validation, you can edit ./configs/HERBS_config.yaml file.
```
#HERBS_config.yaml
is_using_cross_validation: True
cross_validation_folds: 5
```
### 3.4 Combine with background
If you want to combine background into dataset , you can edit ./configs/HERBS_config.yaml file.
```
#HERBS_config.yaml
train_combine_bg: Ture
bg_path: ./bg
```
### 3.5 filter dataset
If you want to filter dataset , you can edit ./configs/HERBS_config.yaml file.

```
#HERBS_config.yaml
is_truncate_train_dataset: False
div_num: 5
# div_num=5 equal to choose[alpha-20%,max] interval data as filtered dataset
```
### 3.6 Save Model
The model will save in /record/project name/ folder


# 4. Test

If you want to test model , you can just run multiple_image_run_evaluation.py

 ### 4.1 Change infer images numbers
 You can choose how many image you  want to infer once by setting input_num .Now support one or three images.
 ### 4.2 Save result
 You can enable <is_save_summary> to save test result.
 ### 4.3 Save confusion matrix
 You can enable <is_save_confusion_matrix> to save test confusion matrix.
 ### 4.4 inference with grayscale image
 You can enable <isgrayscale> to use grayscale iamge to infer.
 ### 4.5 inference with length detection
 You can enable <use_length_detection> to use length detection while infer.
 ### 4.6 save each class's real max length to excel file
 You can enable <is_save_length_to_excel> to save length information to excel file.
### 4.7 Kfold test
 You can run kfold_test.py to test all folds model.The result will save in /record/project name/ folder. remember to change test model path and test data folder.

 # !!!NOTICE!!! the model name shouldn't include <_fold_number>.


#### If you want to test those model.
    baseline_HERBS_fold_0
    baseline_HERBS_fold_1
    baseline_HERBS_fold_2
    baseline_HERBS_fold_3
    baseline_HERBS_fold_4
#### You only need to set model name as below.
 ```
 model_base_name_list = ['baseline_HERBS']
 ```


# 5. Visualize result

### 5.1 How to visualze result
If you want to visualize model result, you can run vis_swin_l.py , remember to change test model path and test data folder
### 5.2 change save result folder
The result will save in pretrained model folder(ex:/record/project name/...),you can change folder name by modify <save_folder_name> variable.