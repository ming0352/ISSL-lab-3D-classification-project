import torch, random
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
from sklearn.metrics import confusion_matrix, classification_report
from eval import count_total_pick_times, avg_result, choose_random_paths, length_detection

os.environ['TORCH_HOME'] = os.path.join('pretrained_model')


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
        PluginMoodel(img_size=img_size,
                     use_fpn=use_fpn,
                     fpn_size=fpn_size,
                     proj_type="Linear",
                     upsample_type="Conv",
                     use_selection=use_selection,
                     num_classes=num_classes,
                     num_selects=num_selects,
                     use_combiner=use_combiner,
                     comb_proj_size=comb_proj_size,
                     isPretrained=True)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        pretrained_dict =  ckpt['model_state_dict']#{k: v for k, v in ckpt['model_state_dict'].items() if
                          # k in model.state_dict() and 'head' not in k}  # ('patch' in k or 'layer' in k or 'norm' in k)}
        model.load_state_dict(pretrained_dict, strict=True)

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
            sum_out = sum_out + tmp_out  # note that use '+=' would cause inplace error
    return sum_out / len(target_layer_names)


def save_confusion_matrix(y_true, y_pred, class2num, output_path):
    font = {'family': 'Microsoft YaHei',
            'weight': 'bold',
            'size': 24}
    plt.rc('font', **font)

    plt.clf()
    ax = plt.subplot()
    # plt.rcParams['font.family'] = ['Microsoft YaHei']
    y_array=y_true+y_pred
    cf_matrix = confusion_matrix(y_true, y_pred,labels=np.unique(y_array))
    classes = [*class2num]
    num2class = dict((value, key) for key, value in class2num.items())
    if len(classes)>len(np.unique(y_array)):
        target_classes=[num2class[i] for i in  np.unique(y_array)]
    else:
        target_classes=classes
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=target_classes,
    #                      columns=target_classes)

    df_cm2 = pd.DataFrame(cf_matrix, index=target_classes,
                          columns=target_classes)
    # plt.figure(figsize=(50, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.xlabel('Predicted',fontsize=48.0, fontweight='bold')
    # plt.ylabel('Ground Truth',fontsize=48.0, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, 'confusion matrix.png'))
    # plt.clf()
    plt.figure(figsize=(20, 20))
    ax=sn.heatmap(df_cm2, annot=True)
    ax.set_title('HERBS(MVImgNet)_RGB',fontsize=60)
    plt.xlabel('Predicted',fontsize=48.0, fontweight='bold')
    plt.ylabel('Ground Truth',fontsize=48.0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion matrix1.png'))
    plt.clf()
    plt.close('all')


def save_to_txt(test_acc, test_top5_acc, classification_report_save_path, y_true, y_pred):
    with open(classification_report_save_path, 'a') as f:
        f.write(f'model test acc: {test_acc:.3f}, test top5 acc: {test_top5_acc:.3f}\n')
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write('\n')





if __name__ == "__main__":
    tmp_rd=[]
    is_save_summary = False
    isgrayscale=False
    is_save_confusion_matrix=False
    input_num =1####need setup this value,ex:1,3
    use_length_detection = True
    # ===== 0. get setting =====
    pretrained_root=os.path.join('records', 'FGVC-HERBS', 'M11_aug_90_50_mvimgnet')
    test_image_path = os.path.join('dataset','50_classes', 'test_replace_10')
    parser = argparse.ArgumentParser("Visualize SwinT Large")

    args = parser.parse_args()
    model_pt_path = os.path.join(pretrained_root, "save_model", "best.pth")
    pt_file = torch.load(model_pt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # ===== 1. build model =====
    model = build_model(pretrainewd_path=model_pt_path,
                        img_size=pt_file['img_size'],
                        fpn_size=pt_file['fpn_size'],
                        num_classes=pt_file['num_class'],
                        num_selects=pt_file['num_selects'])
    model.cuda()

    cls_folders = os.listdir(test_image_path)
    cls_folders.sort()
    top1, top3, top5, top7 = 0, 0, 0, 0
    if use_length_detection:
        ld_top1, ld_top3, ld_top5, ld_top7 = 0, 0, 0, 0
    total = 0
    # n_samples = 0
    top_5_num_correct = 0
    top_7_num_correct = 0
    top_1_num_correct = 0
    # pick_times=0

    class2num = pt_file['class2num']
    num2class = dict((value, key) for key, value in class2num.items())

    # count n_samples
    n_samples = count_total_pick_times(input_num, test_image_path, class2num, cls_folders)

    pbar = tqdm.tqdm(total=n_samples, ascii=True)

    # init summary
    wrongs = {}
    gt_list = []
    pred_list = []
    top1_dic, top5_dic = {}, {}
    ld_top1_dic, ld_top5_dic = {}, {}

    pd_img_name = []
    pd_class_idx = []
    pd_class_name = []
    pd_probs_list = []
    pd_preds_list = []
    # for summary

    pd_top1_correct_list = []
    pd_top5_correct_list = []
    pd_total_class_img_num_dict = {}
    pd_class_name_list = {}

    ld_pd_top1_correct_list = []
    ld_pd_top5_correct_list = []
    ld_pd_total_class_img_num_dict = {}
    ld_pd_class_name_list = {}
    total_num = 0

    y_pred=[]
    y_true=[]

    for ci, cf in enumerate(cls_folders):
        # get class name from folder
        class_name = cf.split('.iam')[0].split('.ipt')[0]
        # if '-' in class_name:
        #     class_name = class_name.split('-')[1]
        # init dict
        if class2num[class_name] not in top1_dic:
            top1_dic[class2num[class_name]] = 0
        if class2num[class_name] not in top5_dic:
            top5_dic[class2num[class_name]] = 0
        if class2num[class_name] not in pd_total_class_img_num_dict:
            pd_total_class_img_num_dict[class2num[class_name]] = 0
        if class2num[class_name] not in pd_class_name_list:
            pd_class_name_list[class2num[class_name]] = class_name
        if class2num[class_name] not in ld_pd_total_class_img_num_dict:
            ld_pd_total_class_img_num_dict[class2num[class_name]] = 0
        if class2num[class_name] not in ld_pd_class_name_list:
            ld_pd_class_name_list[class2num[class_name]] = class_name
        if class2num[class_name] not in ld_top1_dic:
            ld_top1_dic[class2num[class_name]] = 0
        if class2num[class_name] not in ld_top5_dic:
            ld_top5_dic[class2num[class_name]] = 0

        files = os.listdir(os.path.join(test_image_path, cf))
        files.sort()
        imgs = []
        img_paths = []
        update_n = 0
        selected_paths = []
        remaining_paths = []
        if input_num >= 1:
            # set pick time
            pick_times = len(files) // input_num
            if len(files) % input_num != 0:
                pick_times += 1
            tmp = set()
            for i in range(pick_times):
                # add select path into set
                for path_list in selected_paths:
                    for j in path_list:
                        tmp.add(j)
                remaining_paths = list(set(files) - tmp)
                remaining_paths.sort()
                selected_paths.append(choose_random_paths(files, remaining_paths, input_num))

        for img_path in selected_paths:
            tmp_probs_list = []
            tmp_preds_list = []
            pd_total_class_img_num_dict[class2num[class_name]] += 1
            ld_pd_total_class_img_num_dict[class2num[class_name]] += 1
            for idx, img_name in enumerate(img_path):
                if class2num[class_name]==11:
                    print(11)
                # record predict result
                pd_img_name.append(img_name)
                pd_class_idx.append(class2num[class_name])
                pd_class_name.append(class_name)

                img_path = os.path.join(test_image_path, cf, img_name)
                img_paths.append(img_path)
                img_loader = ImgLoader(img_size=pt_file['img_size'],isgrayscale=isgrayscale)
                img, ori_img = img_loader.load(img_path)
                imgs = img.unsqueeze(0).cuda()  # add batch size dimension

                with torch.no_grad():
                    imgs = imgs.cuda()
                    outs = model(imgs)
                    sum_outs = sum_all_out(outs, sum_type="softmax")  # softmax
                    probs, preds = torch.sort(sum_outs, dim=-1, descending=True)
                    y_pred.append(preds[0][0].cpu())  # Save Prediction
                    label = class2num[class_name]#.numpy()
                    y_true.append(label)
                if input_num > 1:
                        tmp_preds_list.append(preds[0])
                        tmp_probs_list.append(probs[0])

            with torch.no_grad():
                if input_num > 1:
                    preds, probs = avg_result(tmp_probs_list, tmp_preds_list)
                if use_length_detection:
                    length_dict = pt_file['real_length_dict']
                    new_preds, new_probs = length_detection(length_dict, class_name, preds, probs, num2class)
                    # tmp_rd.append(rd_value)

                pd_probs_list.append(probs[0].cpu())
                pd_preds_list.append(preds[0].cpu())
            # length detection count correct
            if use_length_detection:
                if class2num[class_name] in new_preds[0][:1]:
                    ld_top1 += 1
                    ld_top1_dic[class2num[class_name]] += 1
                if class2num[class_name] in new_preds[0][:3]:
                    ld_top3 += 1
                if class2num[class_name] in new_preds[0][:5]:
                    ld_top5 += 1
                    ld_top5_dic[class2num[class_name]] += 1
                if class2num[class_name] in new_preds[0][:7]:
                    ld_top7 += 1

            update_n += 1

            # if class2num[cf] in preds[0][:1]:
            #     top_1_num_correct += 1
            #     top1_dic[class2num[cf]] += 1
            # if class2num[cf] in preds[0][:5]:
            #     top_5_num_correct += 1
            #     top5_dic[class2num[cf]] += 1

            if class2num[class_name] in preds[0][:1]:
                top1 += 1
                top_1_num_correct += 1
                top1_dic[class2num[class_name]] += 1
            if class2num[class_name] in preds[0][:3]:
                top3 += 1
            if class2num[class_name] in preds[0][:5]:
                top5 += 1
                top_5_num_correct += 1
                top5_dic[class2num[class_name]] += 1
            if class2num[class_name] in preds[0][:7]:
                top7 += 1
            total += update_n
            top1_acc = round(top1 / total * 100, 3)
            top3_acc = round(top3 / total * 100, 3)
            top5_acc = round(top5 / total * 100, 3)
            top7_acc = round(top7 / total * 100, 3)
            if use_length_detection:
                ld_top1_acc = round(ld_top1 / total * 100, 3)
                ld_top3_acc = round(ld_top3 / total * 100, 3)
                ld_top5_acc = round(ld_top5 / total * 100, 3)
                ld_top7_acc = round(ld_top7 / total * 100, 3)

            msg = "top1: {}%, top3: {}%, top5: {}%, top7: {}%".format(top1_acc, top3_acc, top5_acc, top7_acc)
            pbar.set_description(msg)
            pbar.update(update_n)
            update_n = 0
    pbar.close()
    if use_length_detection:
        print(f'use length detection top1: {ld_top1_acc}%,top5: {ld_top5_acc}%')
    if is_save_confusion_matrix:
        save_confusion_matrix(y_true, y_pred, class2num,pretrained_root)

    if is_save_summary and input_num==1:
        # df = pd.DataFrame(
        #     {'class id': pd_class_idx, 'class name': pd_class_name, 'img name': pd_img_name, 'preds': pd_preds_list,
        #      'probs': pd_probs_list})
        # df.to_excel(os.path.join(pretrained_root, '10_test_bbox_0119_predict_result.xlsx'))
        top1_acc_list = [str(round(a / b,3))+f'({a})' for a, b in zip(top1_dic.values(), pd_total_class_img_num_dict.values())]
        top5_acc_list = [str(round(a / b,3))+f'({a})' for a, b in zip(top5_dic.values(), pd_total_class_img_num_dict.values())]
        df = pd.DataFrame(
            {'class id': top1_dic.keys(), 'class name': pd_class_name_list.values(),
             'total num img': pd_total_class_img_num_dict.values(), 'top1_correct': top1_dic.values(),
             'top1_acc': top1_acc_list, 'top5_correct': top5_dic.values(), 'top5_acc': top5_acc_list, })
        df.to_excel(os.path.join(pretrained_root, '50_10_ori.xlsx'))
        top1_acc_list = [str(round(a / b, 3)) + f'({a})' for a, b in
                         zip(ld_top1_dic.values(), ld_pd_total_class_img_num_dict.values())]
        top5_acc_list = [str(round(a / b, 3)) + f'({a})' for a, b in
                         zip(ld_top5_dic.values(), ld_pd_total_class_img_num_dict.values())]
        df = pd.DataFrame(
            {'class id': ld_top1_dic.keys(), 'class name': pd_class_name_list.values(),
             'total num img': ld_pd_total_class_img_num_dict.values(), 'top1_correct': ld_top1_dic.values(),
             'top1_acc': top1_acc_list, 'top5_correct':ld_top5_dic.values(), 'top5_acc': top5_acc_list, })
        df.to_excel(os.path.join(pretrained_root, '50_10_ld.xlsx'))
    # print(tmp_rd)
