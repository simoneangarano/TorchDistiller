import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.backends.cudnn as cudnn
from inplace_abn import InPlaceABN, InPlaceABNSync
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck, TransferConv
from networks.sagan_models import Discriminator
from utils.criterion import *

def load_S_model(args, model):
    logging.info("------------")
    if args.is_student_load_imgnet:
        if os.path.isfile(args.student_pretrain_model_imgnet):
            saved_state_dict = torch.load(args.student_pretrain_model_imgnet)
            new_params=model.state_dict()
            saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_params}
            new_params.update(saved_state_dict)
            model.load_state_dict(new_params)
            logging.info("=> load" + str(args.student_pretrain_model_imgnet))
        else:
            logging.info("=> the pretrain model on imgnet '{}' does not exit".format(args.student_pretrain_model_imgnet))
    if args.S_resume:
        if os.path.isfile(args.S_ckpt_path):
            checkpoint = torch.load(args.S_ckpt_path)
            args.last_step = checkpoint['step'] if 'step' in checkpoint else None
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' \n (step:{} \n )".format(args.S_ckpt_path, args.last_step))
        else:
            logging.info("=> student checkpoint '{}' does not exit".format(args.S_ckpt_path))
    logging.info("------------")
    return model

def load_T_model(args, model):
    logging.info("------------")
    if os.path.isfile(args.T_ckpt_path):
        try:
            model.load_state_dict(torch.load(args.T_ckpt_path))
        except:
            model.load_state_dict(torch.load(args.T_ckpt_path)['state_dict'], strict=False)
        logging.info("=> load" + str(args.T_ckpt_path))
    else:
        logging.info("=> teacher checkpoint '{}' does not exit".format(args.T_ckpt_path))
    logging.info("------------")

def load_D_model(args, model):
    logging.info("------------")
    if args.D_resume:
        if os.path.isfile(args.D_ckpt_path):
            checkpoint = torch.load(args.D_ckpt_path)
            args.last_step = checkpoint['step'] if 'step' in checkpoint else None
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' \n (step:{} \n )".format(args.D_ckpt_path, args.last_step))
        else:
            logging.info("=> checkpoint '{}' does not exit".format(args.D_ckpt_path))
    else:
        logging.info("=> train d from scratch")
    logging.info("------------")

def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)

def to_tuple_str(str_first, gpu_num, str_ind):
    if gpu_num > 1:
        tmp = '(' 
        for cpu_ind in range(gpu_num):
            tmp += '(' + str_first + '[' + str(cpu_ind) + ']' + str_ind +',)'  
            if cpu_ind != gpu_num-1: tmp +=  ', '
        tmp += ')'
    else:
        tmp = str_first + str_ind  
    return tmp

class NetModel():
    def name(self):
        return 'kd_seg'

    def __init__(self, args):
        self.args = args
        trainable_list = torch.nn.ModuleList([])

        student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
        load_S_model(args, student)
        print_model_parm_nums(student, 'student_model')
        student.cuda()
        self.student = student
        trainable_list.append(self.student)

        teacher = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes = args.num_classes)
        load_T_model(args, teacher)
        print_model_parm_nums(teacher, 'teacher_model')
        teacher.cuda()
        self.teacher = teacher
        
        if args.adv:
            D_model = Discriminator(args.preprocess_GAN_mode, args.num_classes, args.batch_size, 
                                    args.imsize_for_adv, args.adv_conv_dim)
            load_D_model(args, D_model)
            print_model_parm_nums(D_model, 'D_model')
            logging.info("------------")
            D_model.cuda()
            self.D_model = D_model
            
            self.D_solver = optim.Adam(filter(lambda p: p.requires_grad, D_model.parameters()), args.lr_d, [0.9, 0.99])
            
            self.criterion_adv = CriterionAdv(args.adv_loss_type).cuda()
            if args.adv_loss_type == 'wgan-gp': 
                self.criterion_AdditionalGP = CriterionAdditionalGP(D_model, args.lambda_gp).cuda()
            self.criterion_adv_for_G = CriterionAdvForG(args.adv_loss_type).cuda()

        if args.srrl:
            s_chans = 512 if self.args.srrl_layer == 'back' else 128
            t_chans = 2048 if self.args.srrl_layer == 'back' else 512
            
            self.connector = torch.nn.DataParallel(TransferConv(s_chans, t_chans)).cuda()
            self.criterion_srrl = CriterionSRRL(reg_loss=args.reg_loss)
            trainable_list.append(self.connector)
            self.srrl_index = -2 if self.args.srrl_layer == 'back' else -1
            
        if args.mgd:
            self.mgd_index = -2 if self.args.mgd_layer == 'back' else -1
            self.t_chans = 2048 if self.args.mgd_layer == 'back' else 512
            self.s_chans = 512 if self.args.mgd_layer == 'back' else 128
            self.criterion_mgd = MGDLoss(self.s_chans, self.t_chans, args.alpha_mgd, args.mgd_mask)
            trainable_list.append(self.criterion_mgd.align)
            trainable_list.append(self.criterion_mgd.generation)

        if args.ce: 
            self.criterion_dsn = CriterionDSN().cuda()
 
        if args.kd:
            self.criterion_kd = CriterionKD().cuda()
        if args.ekd:
            self.criterion_ekd = CriterionEdgeKD('spatial',args.divergence,args.temperature).cuda()
        if args.cwd:
            self.criterion_cwd = CriterionCWD(args.norm_type,args.divergence,args.temperature).cuda()
            if args.cwd_feat:
                self.linear = torch.nn.Sequential(torch.nn.Conv2d(128, 512, kernel_size=1, stride=1,
                                                                  padding=0, dilation=1, bias=False),
                InPlaceABNSync(512),
                torch.nn.ReLU(inplace=False)).cuda()
                # self.linear = torch.nn.Conv2d(128,512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
                # self.linear = torch.nn.Conv2d(128,512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False).cuda()
                # self.G_solver.add_param_group({'params':self.linear.parameters(), 'initial_lr': args.lr_g})
                
        if args.akd:
            self.criterion_akd = AdaptiveKD(args.norm_type,args.divergence,args.temperature, args.k).cuda()
            
        if args.dkd:
            self.criterion_dkd = CriterionDKD(norm=self.args.norm_dkd, alpha=self.args.alpha_dkd, 
                                              beta=self.args.beta_dkd, temperature=self.args.temp_dkd,
                                              eps=1e-9)
        
        
        self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, trainable_list.parameters()),
                                    'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, 
                                      weight_decay=args.weight_decay)
            
        self.G_loss, self.D_loss = 0.0, 0.0
        self.mc_G_loss, self.kd_G_loss, self.adv_G_loss, self.cwd_G_loss = 0.0, 0.0, 0.0, 0.0
        self.akd_G_loss, self.ekd_G_loss, self.mgd_G_loss, self.dkd_G_loss = 0.0, 0.0, 0.0, 0.0
        
        cudnn.deterministic = True if args.reproduce else False
        cudnn.benchmark = True

    def set_input(self, data):
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def segmentation_forward(self):
        self.preds_S = self.student.train()(self.images)
        if self.args.srrl:
            self.feat_S_adapt = self.connector.train()(self.preds_S[self.srrl_index])
              
        with torch.no_grad():
            self.preds_T = self.teacher.eval()(self.images)
            if self.args.srrl:
                self.preds_ST = self.teacher.eval()(x=None, feats=self.feat_S_adapt, feat_layer=self.args.srrl_layer)[0]

    def segmentation_backward(self):
        args = self.args
        g_loss = 0
        if args.ce:
            temp = self.criterion_dsn(self.preds_S, self.labels)
            self.mc_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.kd:
            temp = args.lambda_kd*self.criterion_kd(self.preds_S, self.preds_T)
            self.kd_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.adv:
            temp = args.lambda_adv*self.criterion_adv_for_G(self.D_model(self.preds_S[0]))
            self.adv_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.cwd:
            if args.cwd_feat:
                temp = args.lambda_cwd*self.criterion_cwd(self.linear(self.preds_S[-1]), self.preds_T[-1])
            else:
                temp = args.lambda_cwd*self.criterion_cwd(self.preds_S[0], self.preds_T[0])
            self.cwd_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.akd:
            temp = args.lambda_akd*self.criterion_akd(self.preds_S[0], self.preds_T[0], self.labels)
            self.akd_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.ekd:
            temp = args.lambda_ekd*self.criterion_ekd(self.preds_S[0], self.preds_T[0])
            self.ekd_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.srrl:
            temp_reg, temp_feat = self.criterion_srrl(self.feat_S_adapt, self.preds_T[self.srrl_index], 
                                                      self.preds_ST, self.preds_T[0])
            temp_feat = temp_feat * args.lambda_srrl_feat
            temp_reg = temp_reg * args.lambda_srrl_reg
            self.srrl_G_loss_feat = temp_feat.item()
            self.srrl_G_loss_reg = temp_reg.item()
            g_loss = g_loss + temp_reg + temp_feat
        if args.mgd:
            temp = self.criterion_mgd(self.preds_S[self.mgd_index], self.preds_T[self.mgd_index])
            temp = temp * args.lambda_mgd
            self.mgd_G_loss = temp.item()
            g_loss = g_loss + temp
            
        if args.dkd:
            temp = min(self.step/self.args.warmup_dkd, 1.0) * self.criterion_dkd(self.preds_S[0], self.preds_T[0], self.labels)
            temp = temp * self.args.lambda_dkd
            self.dkd_G_loss = temp.item()
            g_loss = g_loss + temp
            
        g_loss.backward()
        self.G_loss = g_loss.item()

    def discriminator_forward_backward(self):
        args = self.args
        d_loss = args.lambda_d*self.criterion_adv(self.D_model(self.preds_S[0].detach()), self.D_model(self.preds_T[0].detach()))
        if args.adv_loss_type == 'wgan-gp': d_loss += args.lambda_d*self.criterion_AdditionalGP(self.preds_S, self.preds_T)
        d_loss.backward()
        self.D_loss = d_loss.item()

    def optimize_parameters(self, step=0):
        self.step = step
        self.segmentation_forward()
        self.G_solver.zero_grad()
        self.segmentation_backward()
        self.G_solver.step()
        if self.args.adv:
            self.D_solver.zero_grad()
            self.discriminator_forward_backward()
            self.D_solver.step()

    def print_info(self, step):
        if not step % self.args.log_freq:
            out = f'lr:{self.G_solver.param_groups[-1]["lr"]:.4f} loss:{self.G_loss:.4f} |'
            if self.args.ce:
                out += f' ce:{self.mc_G_loss:.4f}'
            if self.args.kd:
                out += f' kd:{self.kd_G_loss:.4f}'
            if self.args.adv:
                out += f' adv:{self.adv_G_loss:.4f}'
            if self.args.cwd:
                out += f' cwd:{self.cwd_G_loss:.4f}'
            if self.args.ekd:
                out += f' ekd:{self.ekd_G_loss:.4f}'
            if self.args.akd:
                out += f' akd:{self.akd_G_loss:.4f}'            
            if self.args.srrl:
                out += f' srrl_reg:{self.srrl_G_loss_reg:.4f}'
                out += f' srrl_feat:{self.srrl_G_loss_feat:.4f}'
            if self.args.mgd:
                out += f' mgd:{self.mgd_G_loss:.4f}' 
            if self.args.dkd:
                out += f' dkd:{self.dkd_G_loss:.4f}' 
            print(out)

    def save_ckpt(self, step):
        args = self.args
        logging.info('saving ckpt: '+args.save_path+'/'+args.dataset+'_'+str(step)+'_G.pth')
        torch.save(self.student.state_dict(), args.save_path+'/'+args.dataset+'_'+str(step)+'_G.pth')
        if self.args.adv:
            logging.info('saving ckpt: '+args.save_path+'/'+args.dataset+'_'+str(step)+'_D.pth')
            torch.save(self.D_model.state_dict(), args.save_path+'/'+args.dataset+'_'+str(step)+'_D.pth')

    def __del__(self):
        pass
