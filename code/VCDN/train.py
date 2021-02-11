import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from config import gen_args
from data import PhysicsDataset, load_data
from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, to_np, set_seed

from torch.utils.tensorboard import SummaryWriter

from azureml.core.run import Run

#import pdb


args = gen_args()
set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

os.system('mkdir -p ' + args.outf_dy)
os.system('mkdir -p ' + args.dataf)
writer=SummaryWriter(args.outf_dy)

if args.stage == 'dy':
        os.system('mkdir -p ' + args.outf_dy)
        tee = Tee(os.path.join(args.outf_dy, 'train.log'), 'w')
else:
    raise AssertionError("Unsupported env %s" % args.stage)

print(args)

# Get AzureML context for logging
amlc = Run.get_context()

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase, trans_to_tensor=trans_to_tensor)

    if args.gen_data:
        datasets[phase].gen_data()
 #   else:
 #       datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])

#args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


'''
define model for keypoint detection
'''
if args.env == 'Cloth':
    args.nf_hidden = 32


'''
define model for dynamics prediction
'''
if args.stage == 'dy':

    args.nf_hidden = 16

    if args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)
    else:
        raise AssertionError("Unknown dy_model %s" % args.dy_model)

    print("model_dy #params: %d" % count_parameters(model_dy))

    if args.dy_epoch >= 0:
        # if resume from a pretrained checkpoint
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.dy_epoch, args.dy_iter))
        print("Loading saved ckp for dynamics net from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))

else:
    raise AssertionError("Unknown stage %s" % args.stage)


# criterion
criterionMSE = nn.MSELoss()
criterionH = HLoss()

# optimizer
if args.stage == 'dy':
    params = model_dy.parameters()
else:
    raise AssertionError('Unknown stage %s' % args.stage)

optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if use_gpu:
    criterionMSE = criterionMSE.cuda()

    if args.stage == 'dy':
        model_dy = model_dy.cuda()
    else:
        raise AssertionError("Unknown stage %s" % args.stage)


if args.stage == 'dy':
    st_epoch = args.dy_epoch if args.dy_epoch > 0 else 0
    log_fout = open(os.path.join(args.outf_dy, 'log_st_epoch_%d.txt' % st_epoch), 'w')
else:
    raise AssertionError("Unknown stage %s" % args.stage)


best_valid_loss = np.inf

for epoch in range(st_epoch, args.n_epoch):
    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:
        # model_kp.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_contras = AverageMeter()

        if args.stage == 'dy':
            model_dy.train(phase == 'train')
            meter_loss_rmse = AverageMeter()
            meter_loss_kp = AverageMeter()
            meter_loss_H = AverageMeter()
            meter_acc = AverageMeter()
            meter_f1 = AverageMeter()
            meter_cor = AverageMeter()
            meter_num_edge_per_type = np.zeros(args.edge_type_num)

        loader = dataloaders[phase]

        # ProgressBar does not bind nicely with philly
        for i, data in enumerate(loader):

            #if i>100: break

            if use_gpu:
                if isinstance(data, list):
                    # nested transform
                    data = [[d.cuda() for d in dd] if isinstance(dd, list) else dd.cuda() for dd in data]
                else:
                    data = data.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                if args.stage == 'dy':
                    '''
                    hyperparameter on the length of data
                    '''
                    n_his, n_kp = args.n_his, args.n_kp
                    n_samples = args.n_identify + args.n_his + args.n_roll
                    n_identify = args.n_identify

                    '''
                    load data
                    '''
                    if args.env in ['Ball']:
                        if args.gt_kp == 1:
                            # if using ground truth keypoints
                            kps_gt, graph_gt = data[:2]
                        else:
                            # if using detected keypoints
                            if args.preload_kp == 1:
                                # if using preloaded keypoints
                                kps_preload, kps_gt, graph_gt = data[:3]
                            else:
                                # if detect keypoints during runtime
                                imgs, kps_gt, graph_gt = data[:3]
                                B, _, H, W = imgs.size()
                                imgs = imgs.view(B, n_samples, 3, H, W)
                                imgs_id, imgs_dy = imgs[:, :n_identify], imgs[:, n_identify:]

                        actions = data[-1]
                        B = kps_gt.size(0)

                    elif args.env in ['Cloth']:
                        if args.preload_kp == 1:
                            # if using preloaded keypoints
                            kps_preload, actions = data
                        else:
                            imgs, actions = data
                        kps_gt = None
                        B = actions.size(0)

                    '''
                    get detected keypoints -- kps
                    '''
                    # kps: (B * (n_identify + n_his + n_roll)) x n_kp x 2
                    if args.gt_kp == 1:
                        kps = kps_gt.view(-1, n_kp, 2)
                        if args.gt_kp_noise > 0:
                            noise = torch.FloatTensor(np.random.normal(size=(kps.size(0), n_kp, 2))).cuda()
                            noise = noise * args.gt_kp_noise
                            kps = kps + noise

                    else:
                        if args.preload_kp == 1:
                            kps = kps_preload
                        else:
                            kps = model_kp.predict_keypoint(imgs.view(-1, 3, H, W))

                        '''
                        print(kps[0, 0])
                        print(kps_gt[0, 0])
                        print()
                        '''

                        # permute the keypoints to make the edge accuracy correct
                        permu_node_idx = np.arange(args.n_kp)
                        if args.env in ['Ball']:
                            permu_node_idx = np.array([2, 1, 0, 4, 3])

                        kps = kps[:, :, permu_node_idx]

                        '''
                        print(kps[0, 0])
                        print(kps_gt[0, 0])
                        print()
                        '''

                    # kps = kps.view(B, n_samples, n_kp, 2)
                    kps = kps.view(B, n_samples, n_kp, 2)
                    kps_id, kps_dy = kps[:, :n_identify], kps[:, n_identify:]

                    # only train dynamics module
                    kps = kps.detach()

                    if actions is not None:
                        actions_id, actions_dy = actions[:, :n_identify], actions[:, n_identify:]
                    else:
                        actions_id, actions_dy = None, None


                    '''
                    step #1: identify the dynamics graph
                    '''
                    if args.env in ['Ball']:
                        # randomize the observation length
                        observe_length = rand_int(args.min_res, n_identify + 1)

                        if args.baseline == 1:
                            graph = model_dy.init_graph(
                                kps_id[:, :observe_length], use_gpu=True, hard=True)
                        else:
                            graph = model_dy.graph_inference(
                                kps_id[:, :observe_length], actions_id[:, :observe_length],
                                None if args.gt_graph == 0 else graph_gt, env=args.env)

                        # calculate edge calculation accuracy
                        # edge_attr: B x n_kp x n_kp x edge_attr_dim
                        # graph_gt:
                        #   edge_type_gt: B x n_kp x n_kp x edge_type_num
                        #   edge_attr_gt: B x n_kp x n_kp x edge_attr_dim
                        # edge_type_logits: B x n_kp x n_kp x edge_type_num
                        edge_attr, edge_type_logits = graph[1], graph[3]

                        edge_type_gt, edge_attr_gt = graph_gt
                        idx_gt = torch.argmax(edge_type_gt, dim=3)
                        idx_pred = torch.argmax(edge_type_logits, dim=3)
                        assert idx_gt.size() == torch.Size([B, n_kp, n_kp])

                        idx_gt = idx_gt.data.cpu().numpy()
                        idx_pred = idx_pred.data.cpu().numpy()

                        permu_edge_idx = np.array([0, 1])
                        permu_edge_acc = 0.
                        permu_edge_cor = 0.
                        permu_edge_f1 = 0.

                        if permu_edge_idx is None:
                            permu = list(itertools.permutations(np.arange(args.edge_type_num)))

                            edge_attr_np = to_np(edge_attr)
                            edge_attr_gt_np = to_np(edge_attr_gt)

                            for ii in permu:
                                p = np.array(ii)
                                idx_mapped = p[idx_gt]
                                acc = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                                acc = np.sum(acc) / (B * n_kp * (n_kp - 1))

                                f1 = np.logical_and(idx_mapped * idx_pred == 1, np.logical_not(np.eye(n_kp)))
                                recall = np.sum(f1) / \
                                        np.sum(np.logical_and(idx_mapped==1, np.logical_not(np.eye(n_kp))))
                                precision = np.sum(f1) / \
                                        np.sum(np.logical_and(idx_pred, np.logical_not(np.eye(n_kp))))
                                f1 = (2*recall*precision) / (recall+precision)
                                
                                if acc > permu_edge_acc:
                                    permu_edge_f1 = 0 if np.isnan(f1) else f1
                                    permu_edge_acc = acc
                                    permu_edge_idx = p

                        else:

                            edge_attr_np = to_np(edge_attr)
                            edge_attr_gt_np = to_np(edge_attr_gt)

                            idx_mapped = permu_edge_idx[idx_gt]
                            permu_edge_acc = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                            permu_edge_acc = np.sum(permu_edge_acc) / (B * n_kp * (n_kp - 1))


                            tmpsum = np.logical_and(idx_mapped * idx_pred==1, np.logical_not(np.eye(n_kp)))
                            recall = np.sum(tmpsum) / \
                                    np.sum(np.logical_and(idx_mapped==1, np.logical_not(np.eye(n_kp))))
                            precision = np.sum(tmpsum) / \
                                    np.sum(np.logical_and(idx_pred==1, np.logical_not(np.eye(n_kp))))
                            f1 = (2*recall*precision) / (recall+precision)
                            permu_edge_f1 = 0 if np.isnan(f1) else f1
                        '''
                        print()
                        print(permu_edge_idx[idx_gt].reshape(-1))
                        print(idx_pred.reshape(-1))
                        print()
                        '''

                        if args.env in ['Ball']:
                            tmp_cor = np.corrcoef(
                                edge_attr_np.reshape(-1),
                                edge_attr_gt_np.reshape(-1))[0, 1]
                            permu_edge_cor = 0 if np.isnan(tmp_cor) else tmp_cor
                        else:
                            permu_edge_cor = 0.

                    elif args.env in ['Cloth']:
                        # randomize the observation length
                        observe_length = rand_int(args.min_res, n_identify + 1)

                        if args.baseline == 1:
                            graph = model_dy.init_graph(
                                kps_id[:, :observe_length], use_gpu=True, hard=True)
                        else:
                            graph = model_dy.graph_inference(
                                kps_id[:, :observe_length], actions_id[:, :observe_length], env=args.env)

                        # edge_attr: B x n_kp x n_kp x edge_attr_dim
                        # graph_gt:
                        #   edge_type_gt: B x n_kp x n_kp x edge_type_num
                        #   edge_attr_gt: B x n_kp x n_kp x edge_attr_dim
                        # edge_type_logits: B x n_kp x n_kp x edge_type_num
                        edge_attr, edge_type_logits = graph[1], graph[3]

                        idx_pred = torch.argmax(edge_type_logits, dim=3)
                        idx_pred = idx_pred.data.cpu().numpy()

                    # record the number of edges that belongs to a specific type
                    num_edge_per_type = np.zeros(args.edge_type_num)
                    for tt in range(args.edge_type_num):
                        num_edge_per_type[tt] = np.sum(idx_pred == tt)
                    meter_num_edge_per_type += num_edge_per_type


                    # step #2: dynamics prediction
                    eps = args.gauss_std # 0.05
                    kp_cur = kps_dy[:, :n_his].view(B, n_his, n_kp, 2)
                    covar_gt = torch.FloatTensor(np.array([eps, 0., 0., eps])).cuda()
                    covar_gt = covar_gt.view(1, 1, 1, 4).repeat(B, n_his, n_kp, 1)
                    kp_cur = torch.cat([kp_cur, covar_gt], 3)

                    loss_kp = 0.
                    loss_mse = 0.

                    edge_type_logits = graph[3].view(-1, args.edge_type_num)
                    loss_H = -criterionH(edge_type_logits, args.prior)

                    for j in range(args.n_roll):

                        kp_des = kps_dy[:, n_his + j]
                        # predict the feat and hmap at the next time step
                        action_cur = actions_dy[:, j : j + n_his] if actions is not None else None

                        if args.dy_model == 'gnn':
                            # kp_pred: B x n_kp x 2
                            kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur, env=args.env)
                            mean_cur, covar_cur = kp_pred[:, :, :2], kp_pred[:, :, 2:].view(B, n_kp, 2, 2)

                            mean_des, covar_des = kp_des, covar_gt[:, 0].view(B, n_kp, 2, 2)

                            m_cur = MultivariateNormal(mean_cur, scale_tril=covar_cur)
                            m_des = MultivariateNormal(mean_des, scale_tril=covar_des)
                            
                            log_prob = (m_cur.log_prob(kp_des) - m_des.log_prob(kp_des)).mean()

                            loss_kp_cur = -log_prob * args.lam_kp
                            loss_kp += loss_kp_cur / args.n_roll

                            loss_mse_cur = criterionMSE(mean_cur, mean_des)
                            loss_mse += loss_mse_cur / args.n_roll

                        # update feat_cur and hmap_cur
                        kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                    # summarize the losses
                    loss = loss_kp + loss_H

                    # update meter
                    meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
                    meter_loss_kp.update(loss_kp.item(), B)
                    meter_loss_H.update(loss_H.item(), B)
                    meter_loss.update(loss.item(), B)

                    if args.env in ['Ball']:
                        meter_acc.update(permu_edge_acc, B)
                        meter_cor.update(permu_edge_cor, B)
                        meter_f1.update(permu_edge_f1, B)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % args.log_per_iter == 0:
                log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                    phase, epoch, args.n_epoch, i, data_n_batches[phase],
                    get_lr(optimizer))

                if args.stage == 'dy':
                    log += ', kp: %.6f (%.6f), H: %.6f (%.6f)' % (
                        loss_kp.item(), meter_loss_kp.avg,
                        loss_H.item(), meter_loss_H.avg)

                    log += ' [%d' % num_edge_per_type[0]
                    for tt in range(1, args.edge_type_num):
                        log += ', %d' % num_edge_per_type[tt]
                    log += ']'

                    log += ', rmse: %.6f (%.6f)' % (
                        np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                    if args.env in ['Ball']:
                        log += ', acc: %.4f (%.4f)' % (
                            permu_edge_acc, meter_acc.avg)
                        log += ', f1: %.4f (%.4f)' % (permu_edge_f1, meter_f1.avg)
                        log += ' [%d' % permu_edge_idx[0]
                        for ii in permu_edge_idx[1:]:
                            log += ' %d' % ii
                        log += '], cor: %.4f (%.4f)' % (permu_edge_cor, meter_cor.avg)

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()
                
                amlc.log('loss',  np.float(loss.item()))
                amlc.log('loss_kp', np.float(loss_kp.item()))
                amlc.log('loss_H', np.float(loss_H.item()))
                amlc.log('rmse', np.sqrt(loss_mse.item()))
                amlc.log('acc', np.float(permu_edge_acc))
                amlc.log('f1', np.float(permu_edge_f1))
                amlc.log('corr', np.float(permu_edge_cor))
                
                writer.add_scalar("Loss/rmse_train", meter_loss_rmse.avg, i+(epoch*data_n_batches[phase]))
                writer.add_scalar("Loss/hloss_train", meter_loss_H.avg, i+(epoch*data_n_batches[phase]))
                writer.add_scalar("Loss/f1score_train", meter_f1.avg, i+(epoch*data_n_batches[phase]))

            if phase == 'train' and i % args.ckp_per_iter == 0:
                if args.stage == 'dy':
                        torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d_iter_%d.pth' % (args.outf_dy, epoch, i))

        log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
        log += ', [%d' % meter_num_edge_per_type[0]
        for tt in range(1, args.edge_type_num):
            log += ', %d' % meter_num_edge_per_type[tt]
        log += ']'
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg

                if args.stage == 'dy':
                    torch.save(model_dy.state_dict(), '%s/net_best_dy.pth' % (args.outf_dy))

writer.close()
log_fout.close()
tee.flush()
tee.close()
