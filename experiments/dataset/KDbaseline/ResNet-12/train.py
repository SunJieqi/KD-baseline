import os
import sys
import torch
import yaml
from functools import partial

sys.path.append('../../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
from models.KDbaseline import KD_baseline
from utils import util

args = trainer.train_parser()
fewshot_path = "/path_to_dataset"
args.resume = "/root/autodl-tmp/FRN/experiments/dataset/KDbaseline/ResNet-12/model_ResNet-12.pth"
pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                 way=train_way,
                                                 shots=shots,
                                                 transform_type=args.train_transform_type)

model = KD_baseline(way=train_way,
           shots=[args.train_shot, args.train_query_shot],
         resnet=args.resnet)

pretrained_knowledge_path = "/root/autodl-tmp/FRN/experiments/dataset/KDbaseline/ResNet-12"
model.load_state_dict(torch.load(pretrained_knowledge_path, map_location=util.get_device_map(args.gpu)), strict=False)

train_func = partial(frn_train.default_train, train_loader=train_loader)

tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)

tm.train(model)

tm.evaluate(model)
