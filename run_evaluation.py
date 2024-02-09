import torch,random
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import tqdm

from vis_utils import ImgLoader
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from eval import get_length_dict,length_detection
os.environ['TORCH_HOME']=os.path.join('pretrained_model')

def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.pim_module.pim_module_eval import PluginMoodel

    model = \
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size,
                     isPretrained=False)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if
                           k in model.state_dict() and 'head' not in k}#('patch' in k or 'layer' in k or 'norm' in k)}
        model.load_state_dict(pretrained_dict,strict=False)
    
    model.eval()

    return model
@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
    ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 
    'comb_outs']

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error
    return sum_out/len(target_layer_names)

def save_confusion_matrix(y_true,y_pred,class2num,output_path):
    plt.clf()
    plt.rcParams['font.family'] = ['Microsoft YaHei']
    cf_matrix = confusion_matrix(y_true, y_pred)
    classes = [*class2num]
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])

    df_cm2 = pd.DataFrame(cf_matrix, index=[i for i in classes],
                          columns=[i for i in classes])
    plt.figure(figsize=(50, 20))
    sn.heatmap(df_cm, annot=True)

    plt.savefig(os.path.join(output_path , 'confusion matrix.png'))
    plt.clf()
    plt.figure(figsize=(50, 20))
    sn.heatmap(df_cm2, annot=True)
    plt.savefig(os.path.join(output_path , 'confusion matrix1.png'))
    plt.clf()
    plt.close('all')

def save_to_txt(test_acc,test_top5_acc,classification_report_save_path,y_true,y_pred):
    with open(classification_report_save_path, 'a') as f:

        f.write(f'model test acc: {test_acc:.3f}, test top5 acc: {test_top5_acc:.3f}\n')
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write('\n')




if __name__ == "__main__":
    use_three_avg=False
    use_length_detection=False
    use_three_best=False



    import time
    start=time.time()
    # ===== 0. get setting =====
    pretrained_root=os.path.join('records','FGVC-HERBS','M11-augmentation_90_50_r')#M11-augmentation_90_n
    test_image_path= os.path.join('50_classes','test')
    #os.path.join('88_classes','train_dataset','M11_real_test_image')

    parser = argparse.ArgumentParser("Visualize SwinT Large")

    args = parser.parse_args()

    model_pt_path=os.path.join(pretrained_root , "save_model","best.pth")
    pt_file = torch.load(model_pt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # ===== 1. build model =====
    model = build_model(pretrainewd_path=model_pt_path,
                             img_size=pt_file['img_size'],
                             fpn_size=pt_file['fpn_size'],
                             num_classes=pt_file['num_class'],
                             num_selects=pt_file['num_selects'])

    model.cuda()
    print(f'load model time:{time.time()-start}')
    cls_folders = [name for name in os.listdir(test_image_path) if os.path.isdir(os.path.join(test_image_path, name))]#os.listdir(test_image_path)
    cls_folders.sort()
    top1, top3, top5, top7 = 0, 0, 0, 0
    total = 0
    n_samples = 0
    length_top_5_num_correct=0
    length_top_7_num_correct=0
    length_top_1_num_correct=0
    length_top_3_num_correct=0

    for ci, cf in enumerate(cls_folders):
        n_samples += len(os.listdir(os.path.join(test_image_path,cf)))
    pbar = tqdm.tqdm(total=n_samples, ascii=True)
    wrongs = {}
    gt_list=[]
    pred_list=[]
    top1_dic,top3_dic,top5_dic,top7_dic={},{},{},{}


    if use_length_detection:
        length_dict =pt_file['real_length_dict']

    class2num = pt_file['class2num']
    num2class=dict((value,key) for key,value in class2num.items())

    for ci, cf in enumerate(cls_folders):
        class_name = cf.split('.iam')[0].split('.ipt')[0]
        if '-' in class_name:
            class_name = class_name.split('-')[1]
        #init dict
        if class2num[class_name] not in top1_dic:
            top1_dic[class2num[class_name]]=0
        if class2num[class_name] not in top5_dic:
            top5_dic[class2num[class_name]]=0
        if class2num[class_name] not in top3_dic:
            top3_dic[class2num[class_name]]=0
        if class2num[class_name] not in top7_dic:
            top7_dic[class2num[class_name]]=0
        files = os.listdir(os.path.join(test_image_path , cf))
        imgs = []
        img_paths = []
        update_n = 0
        for fi, f in enumerate(files):
            tmp=time.time()
            img_path = os.path.join(test_image_path , cf , f)
            img_paths.append(img_path)
            img_loader = ImgLoader(img_size=pt_file['img_size'])
            img, ori_img = img_loader.load(img_path)
            imgs = img.unsqueeze(0).cuda()

            with torch.no_grad():
                tmp=time.time()
                imgs = imgs.cuda()
                outs = model(imgs)
                sum_outs = sum_all_out(outs, sum_type="softmax") # softmax
                probs,preds = torch.sort(sum_outs, dim=-1, descending=True)

                update_n += 1
                if use_length_detection:
                    new_preds,new_probs = length_detection(length_dict, class_name, preds, probs,num2class)########

                    #calculate length detection top1,top5
                    if class2num[class_name] in new_preds[0][:1]:
                        length_top_1_num_correct+=1
                        top1_dic[class2num[class_name]] += 1
                    if class2num[class_name] in new_preds[0][:3]:
                        length_top_3_num_correct+=1
                        top3_dic[class2num[class_name]] += 1
                    if class2num[class_name] in new_preds[0][:5]:
                        length_top_5_num_correct+=1
                        top5_dic[class2num[class_name]] += 1
                    if class2num[class_name] in new_preds[0][:7]:
                        length_top_7_num_correct+=1
                        top7_dic[class2num[class_name]] += 1

                if class2num[class_name] in preds[0][:1]:
                    top1 += 1
                if class2num[class_name] in preds[0][:3]:
                    top3 += 1
                if class2num[class_name] in preds[0][:5]:
                    top5 += 1
                if class2num[class_name] in preds[0][:7]:
                    top7 += 1

        total += update_n

        imgs = []
        img_paths = []
        top1_acc = round(top1 / total * 100, 3)
        top3_acc = round(top3 / total * 100, 3)
        top5_acc = round(top5 / total * 100, 3)
        top7_acc = round(top7 / total * 100, 3)
        ld_top1_acc = round(length_top_1_num_correct / total * 100, 3)
        ld_top3_acc = round(length_top_3_num_correct / total * 100, 3)
        ld_top5_acc = round(length_top_5_num_correct / total * 100, 3)
        ld_top7_acc = round(length_top_7_num_correct / total * 100, 3)

        msg = f"top1: {top1_acc}%, top3: {top3_acc}%, top5: {top5_acc}%,, top7: {top7_acc}% "# ld  top1: {ld_top1_acc}%, top3: {ld_top3_acc}%, top5: {ld_top5_acc}%,, top7: {ld_top7_acc}%"
        pbar.set_description(msg)
        pbar.update(update_n)
        update_n = 0
    pbar.close()
    print(f'avg inference time:{(time.time() - start)/total}')

if use_length_detection:
    print(f'top1 length detection:{ld_top1_acc} %,top5 length detection:{ld_top5_acc} %,top7 length detection:{ld_top7_acc} %')



