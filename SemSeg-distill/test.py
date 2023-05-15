import os
import torch
from options import TestOptions
from torch.utils import data
from dataset.datasets import CSTestSet
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck
from utils.evaluate import evaluate_main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = TestOptions().initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    testloader = data.DataLoader(CSTestSet(args.data_dir, args.data_list, crop_size=(1024, 2048)), 
                                 batch_size=1, shuffle=False, pin_memory=True, num_workers=24)
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
    student.load_state_dict(torch.load(args.restore_from))
    mean_IU, IU_array = evaluate_main(student, testloader, '512,512', args.num_classes, True, 1, 'test')
    print('mean_IU: {:.6f}  IU_array: \n{}'.format(mean_IU, IU_array))
