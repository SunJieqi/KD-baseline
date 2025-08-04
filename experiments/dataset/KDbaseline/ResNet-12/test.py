import os
import torch
import yaml
import sys

sys.path.append('../../../../')
from models.KDbaseline import KD_baseline
from utils import util
from trainers.eval import meta_test


test_path = "/path_to_dataset"
model_path = 'experiments/dataset/KDbaseline/ResNet-12/model_ResNet-12.pth'


gpu = 0
torch.cuda.set_device(gpu)

model = KD_baseline(resnet=True)
model.cuda()
checkpoint = torch.load(model_path, map_location=util.get_device_map(gpu))
model.load_state_dict(checkpoint["model_state_dict"], strict=True)

model.eval()
with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        mean1, interval1,mean2, interval2= meta_test(data_path=test_path,
                                                       model=model,
                                                       way=way,
                                                       shot=shot,
                                                       pre=False,
                                                       transform_type=0,
                                                       trial=2000)

        print('%d-way-%d-shot acc1: %.3f\t%.3f' % (way, shot, mean1, interval1))
        print('%d-way-%d-shot acc1: %.3f\t%.3f' % (way, shot, mean2, interval2))

