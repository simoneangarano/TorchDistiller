import os
import warnings
warnings.filterwarnings("ignore")
import logging
from alive_progress import alive_it
from alive_progress import config_handler
config_handler.set_global(length=15)

from options import TrainOptions
from dataset.datasets import CSTrainValSet, VOCDataSet, PascalVOCSearchDataset
from networks.kd_model import NetModel
from utils.evaluate import evaluate_main

import random
import numpy as np
import torch
import torchvision

import optuna

class Trainer:
    
    def __init__(self, args, trial=None):

        self.args = args
        self.trial = trial
               
        self.get_data()
        
        self.model = NetModel(self.args)
        
        if args.verbose:
            for key, val in args._get_kwargs():
                logging.info(key+' : '+str(val))
       
    def get_data(self):
        self.h, self.w = map(int, self.args.input_size.split(','))
        
        if self.args.dataset == 'cityscapes':
            train_dataset = CSTrainValSet(self.args.data_dir, self.args.data_list, 
                                          max_iters=self.args.num_steps*self.args.batch_size, crop_size=(self.h, self.w),
                                          scale=self.args.random_scale, mirror=self.args.random_mirror)
            val_dataset = CSTrainValSet(self.args.data_dir, self.args.data_listval, crop_size=(1024, 2048), 
                                        scale=False, mirror=False)
#         elif self.args.dataset == 'pascalvoc':
# #             train_dataset = VOCDataSet(self.args.data_dir, self.args.data_list, 
# #                                        max_iters=self.args.num_steps*self.args.batch_size, crop_size=(self.h, self.w),
# #                                        scale=self.args.random_scale, mirror=self.args.random_mirror)
# #             val_dataset = VOCDataSet(self.args.data_dir, self.args.data_listval, crop_size=(1024, 2048), 
# #                                          scale=False, mirror=False)
#             train_dataset = torchvision.datasets.VOCSegmentation(self.args.data_dir, image_set='train',
#                                                                  transform=torchvision.transforms.ToTensor())
#             val_dataset = torchvision.datasets.VOCSegmentation(self.args.data_dir, image_set='val',
#                                                                  transform=torchvision.transforms.ToTensor())
            
        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, 
                                                       shuffle=True, num_workers=24, pin_memory=True)
        self.valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=24, pin_memory=True)

    def train(self):

        for step, data in alive_it(enumerate(self.trainloader, self.args.last_step), total=self.args.num_steps):
            self.model.adjust_learning_rate(self.args.lr_g, self.model.G_solver, step)
            if self.args.adv:
                self.model.adjust_learning_rate(self.args.lr_d, self.model.D_solver, step)
            self.model.set_input(data)
            self.model.optimize_parameters(step)
            self.model.print_info(step)
            if (((step + 1) >= self.args.save_ckpt_start) \
            and ((step + 1 - self.args.save_ckpt_start) % self.args.save_ckpt_every == 0)) \
            or (step + 1 == self.args.num_steps):
                self.model.save_ckpt(step)
                mean_IU, IU_array = evaluate_main(self.model.student, self.valloader, '512,512', self.args.num_classes,
                                                  whole=True, recurrence=1, split='val', save_out=self.args.save_out)
                logging.info('mean_IU: {:.4f}  IU_array: \n{}'.format(mean_IU, IU_array))
                
                if self.trial is not None:
                    self.trial.report(mean_IU, step)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
        return mean_IU
           
        
def main():
    args = TrainOptions().initialize()

    if args.reproduce:   
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    trainer = Trainer(args)
    trainer.train()
        
if __name__ == "__main__":
    main()