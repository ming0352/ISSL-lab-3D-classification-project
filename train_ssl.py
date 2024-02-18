from SKD.dataloader import get_dataloaders
from SKD.dataset.transform_cfg import transforms_options, transforms_test_options, transforms_list
from data.dataset import get_train_image_list,get_class2num,get_MVImgNet_train_image_list,get_MVImgNet_train_image_list_ori
from SKD.dataset.custom_dataset import MVImgNet_Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.builder import build_swintransformer
import time,os
import torch,timm
from SKD.util import adjust_learning_rate, accuracy, AverageMeter
from tqdm import tqdm
import torch.nn.functional as F
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
import contextlib
import torch.nn as nn
import argparse
import wandb

os.environ['TORCH_HOME']=os.path.join('pretrained_model')

def train(epoch, train_loader, model, criterion, optimizer):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for idx, (_,input, target) in enumerate(pbar):
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            batch_size = input.size()[0]
            x = input
            x_90 = x.transpose(2, 3).flip(2)
            x_180 = x.flip(2).flip(3)
            x_270 = x.flip(2).transpose(2, 3)
            generated_data = torch.cat((x, x_90, x_180, x_270), 0)
            train_targets = target.repeat(4)

            rot_labels = torch.zeros(4 * batch_size).cuda().long()
            for i in range(4 * batch_size):
                if i < batch_size:
                    rot_labels[i] = 0
                elif i < 2 * batch_size:
                    rot_labels[i] = 1
                elif i < 3 * batch_size:
                    rot_labels[i] = 2
                else:
                    rot_labels[i] = 3

            # ===================forward=====================

            feat, (train_logit, rot_logits) = model(generated_data, rot=True)

            rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
            loss_ss = torch.sum(F.binary_cross_entropy_with_logits(input=rot_logits, target=rot_labels))
            loss_ce = criterion(train_logit, train_targets)
            gamma=2
            loss = gamma * loss_ss + loss_ce

            acc1, acc5 = accuracy(train_logit, train_targets, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({"train":f"{epoch+1} epochs",
                "Acc@1": '{0:.4f}'.format(top1.avg.cpu().numpy()),
                              # "Acc@5": '{0:.4f}'.format(top5.avg.cpu().numpy(), 2),
                              "Loss": '{0:.4f}'.format(losses.avg, 2),
                              })


    return top1.avg, losses.avg


def validate(epoch,val_loader, model, criterion):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    simclr=False
    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as pbar:
            end = time.time()
            for idx, (_,input, target) in enumerate(pbar):

                if simclr:
                    input = input[0].float()
                else:
                    input = input.float()

                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                batch_size = input.size()[0]
                x = input
                x_90 = x.transpose(2, 3).flip(2)
                x_180 = x.flip(2).flip(3)
                x_270 = x.flip(2).transpose(2, 3)
                generated_data = torch.cat((x, x_90, x_180, x_270), 0)
                train_targets = target.repeat(4)

                rot_labels = torch.zeros(4 * batch_size).cuda().long()
                for i in range(4 * batch_size):
                    if i < batch_size:
                        rot_labels[i] = 0
                    elif i < 2 * batch_size:
                        rot_labels[i] = 1
                    elif i < 3 * batch_size:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3

                feat, (val_logit, rot_logits) = model(generated_data, rot=True)

                rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                loss_ss = torch.sum(F.binary_cross_entropy_with_logits(input=rot_logits, target=rot_labels))
                loss_ce = criterion(val_logit, train_targets)
                gamma = 2
                loss = gamma * loss_ss + loss_ce

                acc1, acc5 = accuracy(val_logit, train_targets, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({"valid":f"{epoch+1} epochs",
                    "Acc@1": '{0:.4f}'.format(top1.avg.cpu().numpy()),
                                  # "Acc@5": '{0:.4f}'.format(top5.avg.cpu().numpy(), 2),
                                  "Loss": '{0:.4f}'.format(losses.avg, 2),
                                  })

    return top1.avg, top5.avg, losses.avg
def build_loader(args):
    if args.isgrayscale:
        print('SSL dataset is grayscale')
    else:
        print('SSL dataset is RGB')


    class2num = get_class2num(args.train_root)
    num_classes = len(class2num)
    print("[dataset] class number:", num_classes)
    #split train valid data
    print(f'ssl train root:{args.train_root}')
    original_img_path_list, original_img_classes_list = get_MVImgNet_train_image_list(args.train_root, class2num)#get_train_image_list(args.ssl_train_root, class2num)
    print(f'total_dataset_length:{len(original_img_path_list)}')
    x_train, x_valid, y_train, y_valid = train_test_split(original_img_path_list, original_img_classes_list,
                                                          test_size=0.2,
                                                          random_state=32, stratify=original_img_classes_list,
                                                          shuffle=True)
    print(f'train_length:{len(x_train)},valid length:{len(x_valid)}')
    train_trans, test_trans = transforms_options['A']
    train_set, train_loader = None, None
    if x_train is not None:
        train_set = MVImgNet_Dataset(istrain=True, img_list=x_train,label_list=y_train, data_size=args.data_size, return_index=True,transform=train_trans,isgrayscale=args.isgrayscale)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if x_valid is not None:
        val_set = MVImgNet_Dataset(istrain=False, img_list=x_valid,label_list=y_valid, data_size=args.data_size, return_index=True,transform=test_trans,isgrayscale=args.isgrayscale)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def set_environment(args, tlogger):
    print("Setting Environment...")

    args.device = torch.device(args.which_gpu if torch.cuda.is_available() else "cpu")

    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    train_loader, val_loader = build_loader(args)

    tlogger.print()
    class2num = get_class2num(args.train_root)
    args.num_classes = len(class2num)
    ### = = = =  Model = = = =
    # args.pretrained=True
    if not args.pretrained:
        print("doesn't use ImageNet pretrained weight")
    else:
        print("use ImageNet pretrained weight")
    tlogger.print("Building backbone Model....")
    backbone = timm.create_model('swin_large_patch4_window12_384_in22k',
                                     pretrained=args.pretrained, out_feature=False, ssl=True)  # swin_large_patch4_window12_384_in22k

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.classifier =nn.Linear(backbone.out_features_dim, args.num_classes)
    backbone.rot_classifier = nn.Linear(args.num_classes, 4)
    backbone.train()
    model = backbone
    tlogger.print()
    model.to('cuda')

    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None

    ### = = = =  Optimizer = = = =
    tlogger.print("Building Optimizer....")
    # args.ssl_lr_rate=0.0001
    # args.ssl_wdecay=0.0005
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9,
                                    weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext
    start_epoch = 0
    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch
def run_ssl(args,tlogger):
    start_time=time.time()
    #set gpu
    if args.which_gpu=='cuda:0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif args.which_gpu=='cuda:1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    best_acc = 0.0
    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,project=args.project_name, tags='ssl',name=args.model_name)

    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args,
                                                                                                             tlogger,
                                                                                                             )
    os.makedirs(os.path.join('pretrained_model', args.model_name), exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, args.max_epochs):

        tlogger.print("Start Training {} Epoch".format(epoch + 1))
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer)
        tlogger.print()
        time2 = time.time()
        #print('epoch {}, total train time {:.2f}'.format(epoch+1, time2 - time1))

        if args.use_wandb:
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            msg["acc/train_top1__acc"] = train_acc
            msg["loss/train_loss"] = train_loss

        tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
        time3 = time.time()
        val_top1_acc, val_acc_top5, val_loss = validate(epoch,val_loader, model, criterion)
        time4 = time.time()
        if args.use_wandb:
            msg["acc/val_top1_acc"] = val_top1_acc
            msg["acc/val_top5_acc"] = val_acc_top5
            msg["loss/val_loss"] = val_loss
            wandb.log(msg)
        if val_top1_acc >= best_acc:
            best_acc = val_top1_acc
            state = {
                'epoch': epoch,
                # 'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }

            save_file = os.path.join('pretrained_model',args.model_name, 'best.pth')  # 'model_' + str(wandb.run.name) + '.pth'
            torch.save(state, save_file)

        tlogger.print(f"{epoch+1} epochs valid....BEST_ACC: {max(val_top1_acc, best_acc)}% ({val_top1_acc}%)")
        tlogger.print()

        # regular saving
        if epoch % args.eval_freq == 0 or epoch == args.max_epochs:
            #print('==> Saving...')
            state = {
                'epoch': epoch,
                #'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            save_file = os.path.join('pretrained_model',args.model_name, 'last.pth')#'model_' + str(wandb.run.name) + '.pth'
            torch.save(state, save_file)
    end_time = time.time() - start_time
    print(f'train ssl model time:{end_time}')
    if args.use_wandb:
        wandb.finish()
    with open(os.path.join('pretrained_model',args.model_name, 'log.txt'), 'a') as f:
        f.write(f'pretrained ssl swin-l model \n')
        f.write(f'train ssl model time:{end_time}')
        f.write('\n')


if __name__ == "__main__":
    pass
