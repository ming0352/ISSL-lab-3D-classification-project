import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_record_folder(args):

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")

    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "save_model/", exist_ok=True)
    shutil.copy(args.c, args.save_dir+"config.yaml")

def get_args(with_deepspeed: bool=False,config_path="./configs/HERBS_config.yaml"):

    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    parser.add_argument("--c", default=config_path, type=str, help="config file path")
    args = parser.parse_args()

    return args

