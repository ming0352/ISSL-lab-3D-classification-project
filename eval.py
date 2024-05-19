import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import pandas as pd
import os,random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def suppression(target: torch.Tensor, threshold: torch.Tensor, temperature: float = 2):
    """
    target size: [B, S, C]
    threshold: [B',]
    """
    B = target.size(0)
    target = torch.softmax(target / temperature, dim=-1)
    # target = 1 - target
    return target


@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int, thresholds: dict,
                      loss_function=None):
    """
    only present top-1 training accuracy
    """

    total_loss = 0.0

    if args.use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer" + str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = loss_function(outs["layer" + str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()

            gt_score_map = outs["layer" + str(i)]
            thres = torch.Tensor(thresholds["layer" + str(i)])
            gt_score_map = suppression(gt_score_map, thres)
            logit = F.log_softmax(outs["FPN1_layer" + str(i)] / args.temperature, dim=-1)
            loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
            msg["train_loss/layer{}_FPN1_loss".format(i)] = loss_b0.item()

    if args.use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B * S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, args.num_classes]) - 1
            labels_0 = labels_0.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B * S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = loss_function(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if args.use_combiner:
        acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = loss_function(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = loss_function(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss


def cal_evalute_metrics(args, msg: dict, test_loader, batch_size: int, thresholds: dict, model, loss_function=None):
    """
    only present top-1 training accuracy
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):
            datas, labels = datas.to(args.device), labels.to(args.device)

            outs = model(datas)
        if args.use_fpn:
            for i in range(1, 5):
                acc = top_k_corrects(outs["layer" + str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
                acc = round(acc * 100, 2)
                msg["val_acc/layer{}_acc".format(i)] = acc
                loss = loss_function(outs["layer" + str(i)].mean(1), labels)
                msg["val_loss/layer{}_loss".format(i)] = loss.item()
                total_loss += loss.item()

                gt_score_map = outs["layer" + str(i)]
                thres = torch.Tensor(thresholds["layer" + str(i)])
                gt_score_map = suppression(gt_score_map, thres)
                logit = F.log_softmax(outs["FPN1_layer" + str(i)] / args.temperature, dim=-1)
                loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
                msg["val_loss/layer{}_FPN1_loss".format(i)] = loss_b0.item()

        if args.use_selection:
            for name in outs:
                if "select_" not in name:
                    continue
                B, S, _ = outs[name].size()
                logit = outs[name].view(-1, args.num_classes)
                labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B * S)
                acc = round(acc * 100, 2)
                msg["val_acc/{}_acc".format(name)] = acc
                labels_0 = torch.zeros([B * S, args.num_classes]) - 1
                labels_0 = labels_0.to(args.device)
                loss = F.mse_loss(F.tanh(logit), labels_0)
                msg["val_loss/{}_loss".format(name)] = loss.item()
                total_loss += loss.item()

            for name in outs:
                if "drop_" not in name:
                    continue
                B, S, _ = outs[name].size()
                logit = outs[name].view(-1, args.num_classes)
                labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B * S)
                acc = round(acc * 100, 2)
                msg["val_acc/{}_acc".format(name)] = acc
                loss = loss_function(logit, labels_1)
                msg["val_loss/{}_loss".format(name)] = loss.item()
                total_loss += loss.item()

        if args.use_combiner:
            acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["val_acc/combiner_acc"] = acc
            loss = loss_function(outs['comb_outs'], labels)
            msg["val_loss/combiner_loss"] = loss.item()
            total_loss += loss.item()

        if "ori_out" in outs:
            acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["val_acc/ori_acc"] = acc
            loss = loss_function(outs["ori_out"], labels)
            msg["val_loss/ori_loss"] = loss.item()
            total_loss += loss.item()

    msg["val_loss/total_loss"] = total_loss


@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects


@torch.no_grad()
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    
    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1]) # return top-1, top-3, top-5 accuracy
    
    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor, 
    tops: list = [1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    
    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1


def evaluate(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under 
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~ 
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}

    total_batchs = len(test_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0
    total_loss = 0.0

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):

            score_names = []
            scores = []
            datas = datas.to(args.device)

            outs = model(datas)
            out = {}
            for k, v in outs.items():
                tmp = v.clone()
                out[k] = tmp
            if args.use_fpn:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

                    this_name = "FPN1_layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            ### for research
            if args.use_selection:

                for name in outs:
                    if "select_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_1, this_name)

                for name in outs:
                    if "drop_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_0, this_name)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name)

            _average_top_k_result(corrects, total_samples, scores, labels)

            eval_progress = (batch_id + 1) / total_batchs

            if eval_progress > show_progress[progress_i]:
                print(".."+str(int(show_progress[progress_i]*100))+"%", end='', flush=True)
                progress_i += 1

        """ calculate accuracy """
        # total_samples = len(test_loader.dataset)

        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name

    return best_top1, best_top1_name, eval_acces


def evaluate_cm(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}
    results = []

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):

            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas)

            # if args.use_fpn and (0 < args.highest < 5):
            #     this_name = "layer" + str(args.highest)
            #     _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            # _average_top_k_result(corrects, total_samples, scores, labels)

            for i in range(scores[0].shape[0]):
                results.append([test_loader.dataset.data_infos[ids[i].item()]['path'], int(labels[i].item()),
                                int(scores[0][i].argmax().item()),
                                scores[0][i][scores[0][i].argmax().item()].item()])  # 图片路径，标签，预测标签，得分

        """ wirte xlsx"""
        writer = pd.ExcelWriter(args.save_dir + 'infer_result.xlsx')
        df = pd.DataFrame(results, columns=["id", "original_label", "predict_label", "goal"])
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.save()
        writer.close()

        """ calculate accuracy """

        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc > best_top1:
                    best_top1 = acc
                    best_top1_name = name

        """ wirte xlsx"""
        results_mat = np.mat(results)
        y_actual = results_mat[:, 1].transpose().tolist()[0]
        y_actual = list(map(int, y_actual))
        y_predict = results_mat[:, 2].transpose().tolist()[0]
        y_predict = list(map(int, y_predict))

        folders = os.listdir(args.val_root)
        folders.sort()  # sort by alphabet
        print("[dataset] class:", folders)
        df_confusion = confusion_matrix(y_actual, y_predict)
        plot_confusion_matrix(df_confusion, folders, args.save_dir + "infer_cm.png", accuracy=best_top1)

    return best_top1, best_top1_name, eval_acces


@torch.no_grad()
def eval_and_save(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write(msg)


@torch.no_grad()
def eval_and_cm(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate_cm(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "infer_results.txt", "w") as ftxt:
        ftxt.write(msg)


def plot_confusion_matrix(cm, label_names, save_name, title='Confusion Matrix acc = ', accuracy=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(len(label_names) / 2, len(label_names) / 2), dpi=100)
    np.set_printoptions(precision=2)
    # print("cm:\n",cm)

    # 统计混淆矩阵中每格的概率值
    x, y = np.meshgrid(np.arange(len(cm)), np.arange(len(cm)))
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        try:
            c = (cm[y_val][x_val] / np.sum(cm, axis=1)[y_val]) * 100
        except KeyError:
            c = 0
        if c > 0.001:
            plt.text(x_val, y_val, "%0.1f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title + str('{:.3f}'.format(accuracy)))
    plt.colorbar()
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(label_names))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    # plt.show()

def get_length_dict(path):
    '''
        parse  log file to get real max length dict
        Args:
            path: log file path

        Returns: length_dict

        '''
    length_dict = {}
    with open(path, encoding='utf8') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            if 'The size of ' in lines[i]:
                a=float(lines[i].split(':')[-1].split(',')[0].split('mm')[0].split(' ')[-1])/1000
                b=float(lines[i].split(':')[-1].split(',')[1].split('mm')[0].split(' ')[-1])/1000
                c=float(lines[i].split(':')[-1].split(',')[2].split('mm')[0].split(' ')[-1])/1000
                max_length = max(a,b,c)
                class_name = lines[i].split(' :')[0].split('The size of ')[-1].split('.iam')[0].split('.ipt')[0]
                if '_MIR' in class_name:
                    class_name=class_name.split('_MIR')[0]

                if class_name not in length_dict.keys():
                    length_dict[class_name] = max_length
    return length_dict
def length_detection(length_dict, cf, preds, probs,num2class):
    ld_tolerance_scope = 0.01
    target_length = length_dict[cf]
    #holo_tolerance_scope = 0.01

    preds_tmp=preds.detach().clone()
    probs_tmp=probs.detach().clone()
    for i in range(0, len(preds[0])):
        P=length_dict[num2class[int(preds[0][i])]]
        if ld_tolerance_scope != 0:
            rd_value=np.random.choice(np.asarray([-ld_tolerance_scope, ld_tolerance_scope]))
            cmp_value = rd_value + target_length
        else:
            cmp_value = target_length

        if not ((P-ld_tolerance_scope) <= cmp_value <= (P+ld_tolerance_scope)):
            preds_tmp = torch.cat([preds_tmp[preds_tmp != preds[0][i]],preds_tmp[preds_tmp == preds[0][i]]])
            probs_tmp = torch.cat([probs_tmp[probs_tmp != probs[0][i]],probs_tmp[probs_tmp == probs[0][i]]])

        new_preds = torch.unsqueeze(torch.tensor(preds_tmp).cuda(), dim=0)
        new_probs = torch.unsqueeze(torch.tensor(probs_tmp).cuda(), dim=0)
    return new_preds, new_probs
def choose_random_paths(files, paths, input_num):
    paths.sort()
    random.seed(2)
    if len(paths) > input_num:
        return random.sample(paths, input_num)
    else:
        other_path = [item for item in files if item not in paths]
        other_path.sort()
        return random.sample(other_path, input_num - len(paths)) + paths

def avg_result(tmp_probs_list,tmp_preds_list):
    t = {}
    for i, pred_tensor in enumerate(tmp_preds_list):
        for j, g in enumerate(pred_tensor):
            if int(g) not in t.keys():
                t[int(g)] = 0
            t[int(g)] = torch.add(t[int(g)], tmp_probs_list[i][j])
    t = {key: value / len(tmp_preds_list) for key, value in t.items()}

    sorted_pairs = sorted(zip(list(t.values()), list(t.keys())), reverse=True)
    prob, pred = zip(*sorted_pairs)

    preds = torch.unsqueeze(torch.tensor(pred).cuda(), dim=0)
    probs = torch.unsqueeze(torch.tensor(prob).cuda(), dim=0)
    return preds, probs


def count_total_pick_times(input_num,test_image_path,class2num,cls_folders):
    n_samples=0
    num_dict = {}
    for ci, cf in enumerate(cls_folders):
        if input_num > 1:
            cf = cf.split('.iam')[0].split('.ipt')[0]
            # if '-' in cf:
            #     cf = cf.split('-')[1]
            if ci not in num_dict:
                num_dict[class2num[cf]]=0
            files = os.listdir(os.path.join(test_image_path, cf))
            files.sort()
            pick_times = len(files) // input_num
            if len(files) % input_num != 0:
                pick_times += 1
            n_samples += pick_times
            num_dict[class2num[cf]]=pick_times
        else:
            n_samples += len(os.listdir(os.path.join(test_image_path, cf)))
    return n_samples

def log_softmax(x):
    x=x.detach().cpu()
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp