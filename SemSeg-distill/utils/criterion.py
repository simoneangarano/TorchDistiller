import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import lovely_tensors as lt
lt.monkey_patch()

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4
    
    
class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, upsample=False, temperature=1):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft):
        soft[0].detach()
        h, w = soft[0].size(2), soft[0].size(3)
        if self.upsample:
            scale_pred = F.upsample(input=pred[0], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
            scale_soft = F.upsample(input=soft[0], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = pred[0]
            scale_soft = soft[0]
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.temperature, dim=1), F.softmax(scale_soft / self.temperature, dim=1))
        return loss

class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S):
        g_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake

class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S[0].shape == d_out_T[0].shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake

class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S[0].shape == d_in_T[0].shape,'the output dim of D with teacher and student as input differ'

        real_images = d_in_T[0]
        fake_images = d_in_S[0]
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out[0].size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss

class CriterionIFV(nn.Module):
    def __init__(self, classes):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, preds_S, preds_T, target):
        feat_S = preds_S # features
        feat_T = preds_T # features
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3]) # H, W
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size()) # rescale label
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size()) # rescale label
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float() # binary mask
            mask_feat_T = (tar_feat_T == i).float() # binary mask
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1) # compute center
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1) # compute center

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss

class SpatialNorm(nn.Module):
    def __init__(self,divergence='kl'):
        if divergence =='kl':
            self.criterion = nn.KLDivLoss()
        else:
            self.criterion = nn.MSELoss()

        self.norm = nn.Softmax(dim=-1)
    
    def forward(self,pred_S,pred_T):
        norm_S = self.norm(pred_S)
        norm_T = self.norm(pred_T)

        loss = self.criterion(pred_S,pred_T)
        return loss

# class ChannelNorm(nn.Module):
#     def __init__(self,divergence='kl'):
#         if divergence =='kl':
#             self.criterion = nn.KLDivLoss()
#         else:
#             self.criterion = nn.MSELoss()


#         self.norm = nn.Sequential(
#                 nn.Flatten(start_dim=2),
#                 nn.Softmax(dim=-1),
#             )
    
#     def forward(self,pred_S,pred_T):
#         norm_S = self.norm(pred_S,dim=1)
#         norm_T = self.norm(pred_T,dim=1)

#         loss = self.criterion(pred_S,pred_T)
#         return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap
    
    
class EdgeNorm(torch.nn.Module):
    def __init__(self):
        super(EdgeNorm, self).__init__()
        
        self.kernel = np.ones((10, 10), np.uint8)
        self.t1 = 10
        self.t2 = 50

    def forward(self,t_log,s_log):
        n,c,h,w = t_log.shape
        
        t_map = (np.argmax(t_log.cpu(), axis=1) / 13 * 255).type(torch.uint8)

        edges = []
        for (tl, sl, tm) in zip(t_log, s_log, t_map):
            e = cv2.Canny(image=tm.cpu().numpy(), threshold1=self.t1, threshold2=self.t2)

            e = cv2.dilate(e, self.kernel, iterations = 1)
            e = cv2.blur(e, (10,10))
            edges.append(e / 255)
            
        edges = torch.tensor(np.stack(edges)).cuda()
        t_log_masked = edges[:,None,...] * t_log
        s_log_masked = edges[:,None,...] * s_log
        
#         t_log_masked = t_log_masked.reshape((n,c,-1))
#         s_log_masked = s_log_masked.reshape((n,c,-1))
        
#         t_soft_masked = t_log_masked.softmax(dim=-1)
#         s_soft_masked = s_log_masked.softmax(dim=-1)
        
        return t_log_masked, s_log_masked

    
class AdaptiveNorm(torch.nn.Module):
    
    def __init__(self, k=0.5):
        super(AdaptiveNorm, self).__init__()
        self.k = k

    def forward(self,t_log,s_log,lab):
        n,c,h,w = t_log.shape
        
        t_lab = np.argmax(t_log.cpu(), axis=1) 
        mask = torch.eq(t_lab, lab)

        gt_log = torch.nn.functional.one_hot(lab, num_classes=c)
        gt_log = torch.permute(gt_log, (0, 3, 1, 2))
        t_log_masked = (self.k * t_log + (1 - self.k) * gt_log) * mask[:,None,...] + gt_log * ~mask[:,None,...]

        t_log_masked = t_log_masked.reshape((n,c,-1))
        s_log = s_log.reshape((n,c,-1))
        
        t_soft_masked = t_log_masked.softmax(dim=-1)
        s_soft = s_log.softmax(dim=-1)
        
        return t_soft_masked, s_soft

    
class CriterionCWD(nn.Module):

    def __init__(self,norm_type='none',divergence='mse',temperature=1.0, k=0.5):
        super(CriterionCWD, self).__init__()
       
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        elif norm_type == 'edge':
            self.normalize = EdgeNorm()
        elif norm_type == 'adaptive':
            self.normalize = AdaptiveNorm(k=k)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = temperature

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence
 

    def forward(self,preds_S, preds_T):
        n,c,h,w = preds_S.shape
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            if self.norm_type == 'edge':
                norm_t, norm_s = self.normalize(preds_T.detach()/self.temperature, preds_S/self.temperature)
            else:
                norm_s = self.normalize(preds_S/self.temperature)
                norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type in ['channel','channel_mean','edge']:
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)
        
        
class AdaptiveKD(torch.nn.Module):

    def __init__(self,norm_type='none',divergence='mse',temperature=1.0, k=0.5):
        super(AdaptiveKD, self).__init__()
       
        self.temperature = temperature
        self.k = k

        # define loss function
        if divergence == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = torch.nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self,t_log,s_log,lab):
        
        n, c, h, w = t_log.shape
        
        lab = F.interpolate(lab[:,None,...].float(), (h,w), mode='bilinear', align_corners=True).long()
        t_lab = torch.argmax(t_log, axis=1, keepdims=True) 
        
        mask = torch.eq(t_lab, lab)
        gt_log = torch.nn.functional.one_hot(lab[:,0,...])[...,:c]
        gt_log = torch.permute(gt_log, (0, 3, 1, 2))
        
        t_log_masked = (self.k * t_log + (1 - self.k) * gt_log) * mask + gt_log * ~mask
        
        t_log_masked = t_log_masked.reshape((n,c,-1))
        s_log = s_log.reshape((n,c,-1))
        
        t_soft_masked = t_log_masked.softmax(dim=-1)
        s_soft = s_log.softmax(dim=-1)   
        
        if self.divergence == 'kl':
            s_soft = s_soft.log()
        loss = self.criterion(s_soft,t_soft_masked)
        loss /= n * c
        
        return loss * (self.temperature**2)
    
    
class CriterionEdgeKD(torch.nn.Module):

    def __init__(self,norm_type='none',divergence='mse',temperature=1.0):
        super(CriterionEdgeKD, self).__init__()
        
        self.attention = EdgeNorm()
        self.norm_type = norm_type
        self.temperature = temperature
        self.divergence = divergence

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        else:
            self.normalize = None
        
        # define loss function
        if divergence == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = torch.nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
    
    
    def forward(self, preds_S, preds_T):
        n,c,h,w = preds_S.shape
        
        pred_T, pred_S = self.attention(preds_T, preds_S)
        
        if self.normalize is not None:
            if self.norm_type == 'channel':
                pred_T = pred_T.reshape((n,c,-1))
                pred_S = pred_S.reshape((n,c,-1))
                norm_t = self.normalize(pred_T)
                norm_s = self.normalize(pred_S)
                
            if self.norm_type == 'spatial':
                norm_s = self.normalize(preds_S/self.temperature)
                norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)

        if self.norm_type in ['channel','channel_mean']:
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)
    
    
class StatmLoss(nn.Module):
    def __init__(self):
        super(StatmLoss, self).__init__()

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()
    
    
class CriterionSRRL(nn.Module):

    def __init__(self, reg_loss='mse'):
        super(CriterionSRRL, self).__init__()
       
        # define normalize function
        if reg_loss == 'mse':
            self.regression_loss = F.mse_loss
        elif reg_loss == 'channel':
            self.regression_loss = CriterionCWD('channel','kl').cuda()
            
        self.feature_loss = StatmLoss()
      
    def forward(self, feat_S, feat_T, pred_ST, pred_T):
        preds_T = pred_T
        preds_S = pred_ST
        
        n,c,h,w = preds_S.shape
        
        loss_reg = self.regression_loss(preds_S,preds_T)
        loss_feat = self.feature_loss(feat_S, feat_T.detach())

        return loss_reg, loss_feat
    
    
class MGDLoss(nn.Module):
    """
    PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): masked ratio. Defaults to 0.5
    """
    def __init__(self, student_channels, teacher_channels, alpha_mgd=0.15, mask='channel', name='MGD'):
        super(MGDLoss, self).__init__()
        
        self.alpha_mgd = alpha_mgd
        self.mask = mask
        
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0).cuda()
        else:
            self.align = None

        self.generation = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True), 
                                        nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)).cuda()


    def forward(self, preds_S, preds_T):
        """
        Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)
            
        return loss

    
    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        if self.mask == 'channel': 
            mat = torch.rand((N,C,1,1)).to(device)
        elif self.mask == 'space':
            mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < self.alpha_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss

    
    
class CriterionDKD(nn.Module):

    def __init__(self, norm='space', alpha=1.0, beta=8.0, temperature=4.0, eps=1e-5):
        super(CriterionDKD, self).__init__()
       
        # define normalize function
        self.norm = norm
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # define loss function
        self.criterion = nn.KLDivLoss(reduction='sum')
 

    def forward(self, preds_S, preds_T, target):

        h, w = target.size(1), target.size(2)
        preds_S = F.interpolate(preds_S, size=(h, w), mode='bilinear', align_corners=True)
        preds_T = F.interpolate(preds_T, size=(h, w), mode='bilinear', align_corners=True)
        
        n,c,h,w = preds_S.shape
        
        target = self._ignore_index(target)
        
        loss_dkd = self.dkd_loss(preds_S.reshape((n,c,-1)), preds_T.reshape((n,c,-1)), target.reshape((n,-1)),
                                 self.norm, self.alpha, self.beta, self.temperature)
        return loss_dkd
    
    
    def dkd_loss(self, logits_student, logits_teacher, target, norm, alpha, beta, temperature):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        
        pred_student = F.softmax(logits_student / temperature, dim=1 if norm == 'space' else 1) # good
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1 if norm == 'space' else 1) # good
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask, ax=1 if norm == 'space' else 1)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask, ax=1 if norm == 'space' else 1)
        
        log_pred_student = torch.log(pred_student + self.eps)
        
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        tckd_loss *= (temperature**2) / pred_teacher.shape[0]
        tckd_loss /= pred_teacher.shape[-1] if norm == 'space' else pred_teacher.shape[-1]            
            
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1 if norm == 'space' else 2)
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1 if norm == 'space' else 2)
               
        nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        nckd_loss *= (temperature**2) / pred_teacher_part2.shape[0]
        nckd_loss /= pred_teacher_part2.shape[-1] if norm == 'space' else pred_teacher_part2.shape[-2]
        
        return alpha * tckd_loss + beta * nckd_loss

    
    def _ignore_index(self, target, index=255, n_classes=19):
        return torch.clamp(target, 0, n_classes)
        
        
    def _get_gt_mask(self, logits, target): # sequential target coding (B)
        n, c, hw = logits.shape
        mask = torch.zeros((n, c+1, hw)).cuda().scatter_(1, target.unsqueeze(1), 1).bool()
        return mask[:,:-1,...]


    def _get_other_mask(self, logits, target):
        n, c, hw = logits.shape
        mask = torch.ones((n, c+1, hw)).cuda().scatter_(1, target.unsqueeze(1), 0).bool()
        return mask[:,:-1,...]


    def cat_mask(self, t, mask1, mask2, ax):
        t1 = (t * mask1).sum(ax, keepdims=True)
        t2 = (t * mask2).sum(ax, keepdims=True)
        rt = torch.cat([t1, t2], dim=ax)
        return rt