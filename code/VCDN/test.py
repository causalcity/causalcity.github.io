import os
import random
import itertools
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 12

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import gen_args
from data import PhysicsDataset, load_data, store_data, resize_and_crop, pil_loader
#from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import count_parameters, Tee, AverageMeter, to_np, to_var, confidence_ellipse, norm, set_seed
from draw_graph import draw_graph
from data import normalize, denormalize
import pdb
import csv
import sklearn.metrics as metrics

args = gen_args()

use_gpu = torch.cuda.is_available()

set_seed(args.random_seed)


# used for cnn encoder, minimum input observation length
min_res = args.min_res


'''
model
'''
if args.env == 'Cloth':
    args.nf_hidden = 32


if args.stage == 'dy':

    args.nf_hidden = 16

    if args.dy_model == 'mlp':
        model_dy = DynaNetMLP(args, use_gpu=use_gpu)
    elif args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)

    # print model #params
    print("model #params: %d" % count_parameters(model_dy))

    if args.eval_dy_epoch == -1:
        model_dy_path = os.path.join(args.outf_dy, 'net_best_dy.pth')
    else:
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.eval_dy_epoch, args.eval_dy_iter))

    print("Loading saved ckp from %s" % model_dy_path)
    model_dy.load_state_dict(torch.load(model_dy_path))
    model_dy.eval()

if use_gpu:
    model_dy.cuda()


criterionMSE = nn.MSELoss()
criterionH = HLoss()


'''
data
'''
data_dir = os.path.join(args.dataf, args.eval_set)

if args.env in ['Ball']:
    data_names = ['attrs', 'states', 'actions', 'rels']
elif args.env in ['Cloth']:
    data_names = ['states', 'actions', 'scene_params']

loader = pil_loader

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


'''
store results
'''
os.system('mkdir -p ' + args.logf)

log_path = os.path.join(args.logf, 'log.txt')
tee = Tee(log_path, 'w')



def evaluate(roll_idx, video=False, image=False):
    fwd_loss_mse_cur = []

    eval_path = os.path.join(args.evalf, str(roll_idx))

    n_split = 4
    split = 4
    n_kp = args.n_kp

    if False:
        os.system('mkdir -p ' + eval_path)
        print('Save images to %s' % eval_path)

    if video:
        video_path = eval_path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % video_path)
        out = cv2.VideoWriter(video_path, fourcc, 10, (
            120, 120))


    # load images
    fig_suffix = '.png' if args.env == 'Ball' else '.jpg'

    # load action
    if args.env in ['Ball']:
        data_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        data = load_data(data_names, data_path)

        actions = data[data_names.index('actions')] / 10.
        actions = torch.FloatTensor(actions).cuda()
        actions_id = actions[args.identify_st_idx:args.identify_ed_idx]
 
    elif args.env in ['Cloth']:
        data_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        data = load_data(data_names, data_path)

        states = data[data_names.index('states')][::args.frame_offset]
        actions_raw = data[data_names.index('actions')][::args.frame_offset]
        scene_params = data[data_names.index('scene_params')]
        stiffness = scene_params[15]
        ctrl_idx = scene_params[7:15].astype(np.int)

        actions = np.zeros((states.shape[0], 6))
        actions[:, :3] = states[
            np.arange(actions.shape[0]),
            ctrl_idx[actions_raw[:, 0, 0].astype(np.int)],
            :3] / 0.5   # normalize
        actions[:, 3:] = actions_raw[:, 0, 1:] / 0.03   # normalize

        actions = torch.FloatTensor(actions)[:, None, :].repeat(1, args.n_kp, 1)
        actions = actions.cuda()
        actions_id = actions[args.identify_st_idx:args.identify_ed_idx]


    '''
    model prediction
    '''

    if args.stage == 'dy':

        '''
        metadata
        '''
        metadata_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        metadata = load_data(data_names, metadata_path)
        if args.env in ['Ball']:
            # graph_gt
            edge_type = metadata[data_names.index('rels')][0, :, 0].astype(np.int)
            edge_attr = metadata[data_names.index('rels')][0, :, 1:]
            edge_type_gt = np.zeros((args.n_kp, args.n_kp, args.edge_type_num))
            edge_attr_gt = np.zeros((args.n_kp, args.n_kp, edge_attr.shape[1]))
            cnt = 0
            for x in range(args.n_kp):
                for y in range(x+1, args.n_kp):
                    edge_type_gt[x, y, edge_type[cnt]] = 1.
                    edge_type_gt[y, x, edge_type[cnt]] = 1.
                    edge_attr_gt[x, y] = edge_attr[cnt]
                    edge_attr_gt[y, x] = edge_attr[cnt]
                    cnt += 1
            graph_gt_ret = edge_type_gt, edge_attr_gt
            edge_type_gt = torch.FloatTensor(edge_type_gt).cuda()
            edge_attr_gt = torch.FloatTensor(edge_attr_gt).cuda()

            graph_gt = edge_type_gt, edge_attr_gt

            # kps
            kps = metadata[1][args.eval_st_idx:args.eval_ed_idx, :, :2]
            kps[:, :, 0] = (kps[:,:,0]) * 2 / (1024.0) - 1.0 
            kps[:, :, 1] = (kps[:,:,1]) * 2 / (640.0) - 1.0 
            kps[:, :, 1] *= -1
            kps = torch.FloatTensor(kps).cuda()
            kps_id = metadata[1][args.identify_st_idx:args.identify_ed_idx, :, :2]
            kps_id[:, :, 0] = (kps_id[:,:,0]) * 2 / (1024.0) - 1.0 
            kps_id[:, :, 1] = (kps_id[:,:,1]) * 2 / (640.0) - 1.0
            kps_id = torch.FloatTensor(kps_id).cuda()
            kps_id[:, :, 1] *= -1


            kps_gt = kps
            kps_gt_id = kps_id

            if args.gt_kp == 0:
                # if using detected keypoints
                kps = None
                kps_id = None

        elif args.env in ['Cloth']:
            kps = None
            kps_id = None


        with torch.set_grad_enabled(False):
            # extract features for prediction

            # permute the keypoints to make the edge accuracy correct
            if args.gt_kp == 0:
                if args.env in ['Ball']:
                    if args.n_kp == 3:
                        permu_node_idx = np.array([1, 0, 2])
                    elif args.n_kp == 4:
                        permu_node_idx = np.array([0, 1, 2, 3])
                    elif args.n_kp == 5:
                        permu_node_idx = np.array([2, 1, 0, 4, 3])
                    elif args.n_kp == 6:
                        permu_node_idx = np.array([2, 5, 3, 0, 4, 1])
                    elif args.n_kp == 7:
                        permu_node_idx = np.array([2, 0, 1, 6, 3, 4, 5])

                    print(permu_node_idx)

                    kps = kps[:, permu_node_idx]
                    kps_id = kps_id[:, permu_node_idx]


            graphs = []
            for i in range(min_res, kps_id.size(0) + 1):
                edge_type_distribution = 0
                edge_attr_distribution = []

                if args.baseline == 1:
                    graph = model_dy.init_graph(kps_id[:i].unsqueeze(0), use_gpu=True, hard=True)
                else:
                    graph = model_dy.graph_inference(
                        kps_id[:i].unsqueeze(0),
                        actions_id[:i].unsqueeze(0) if actions_id is not None else None,
                        None if args.gt_graph == 0 else graph_gt.unsqueeze(0),
                        hard=True, env=args.env)
                graphs.append(graph)    # append the inferred graph

            edge_type_logits = graphs[-1][3].view(-1, args.edge_type_num)
            loss_H = -criterionH(edge_type_logits, args.prior)

            edge_attr, edge_type_logits = graphs[-1][1], graphs[-1][3]
            graph_pred_ret = to_np(edge_attr[0]), to_np(edge_type_logits[0])


            if args.env in ['Ball']:
                # record the inferred graph over time
                idx_gt = torch.argmax(edge_type_gt, dim=2)
                idx_pred = torch.argmax(edge_type_logits[0], dim=2)

                assert idx_gt.size() == torch.Size([n_kp, n_kp])
                assert idx_pred.size() == torch.Size([n_kp, n_kp])

                idx_gt = to_np(idx_gt)
                idx_pred = to_np(idx_pred)

                permu_edge_list = list(itertools.permutations(np.arange(args.edge_type_num)))
                permu_edge_acc = 0.
                permu_edge_idx = None
                for ii in permu_edge_list:
                    p = np.array(ii)
                    idx_mapped = p[idx_gt]
                    acc = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                    acc = np.sum(acc) / (n_kp * (n_kp - 1))

                    if acc > permu_edge_acc:
                        permu_edge_acc = acc
                        permu_edge_idx = p

                if args.env in ['Ball']:
                    permu_edge_idx = np.array([0, 1])

                print('selected premu', permu_edge_idx)

                # record the edge type accuracy over time
                acc_over_time = np.zeros(len(graphs))
                ent_over_time = np.zeros(len(graphs))
                recall_over_time = np.zeros(len(graphs))
                pre_over_time = np.zeros(len(graphs))
                f1_over_time = np.zeros(len(graphs))
                idx_mapped = idx_gt
                for i in range(len(graphs)):
                    edge_type_logits_cur = graphs[i][3][0]

                    # accuracy
                    idx_pred = torch.argmax(edge_type_logits_cur, dim=2)
                    assert idx_pred.size() == torch.Size([n_kp, n_kp])

                    idx_pred = to_np(idx_pred)
                    idx_pred = permu_edge_idx[idx_pred]

                    tmp = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                    acc_over_time[i] = np.sum(tmp) / (n_kp * (n_kp - 1))

                    tmp2 = np.logical_and(idx_mapped * idx_pred == 1, np.logical_not(np.eye(n_kp)))
                    recall_over_time[i] = np.sum(tmp2) / \
                        np.sum(np.logical_and(idx_mapped==1, np.logical_not(np.eye(n_kp))))
                    pre_over_time[i] = np.sum(tmp2) / \
                            np.sum(np.logical_and(idx_pred==1, np.logical_not(np.eye(n_kp))))
                    f1_over_time[i] = (2 * pre_over_time[i] * recall_over_time[i]) / \
                            (pre_over_time[i] + recall_over_time[i])
                    if np.isnan(pre_over_time[i]):
                        pre_over_time[i] = 0
                    if np.isnan(recall_over_time[i]):
                        recall_over_time[i] = 0
                    if np.isnan(f1_over_time[i]):
                        f1_over_time[i] = 0
                    # entropy
                    ent = F.softmax(edge_type_logits_cur, dim=2) * F.log_softmax(edge_type_logits_cur, dim=2)
                    ent = -ent.sum(2)
                    ent = ent.mean().item()
                    ent_over_time[i] = ent

                # print out causal graph
#                print('Pr: ', idx_pred)
#                print('GT: ', idx_mapped)
                # record the edge param correlation over time
                cor_over_time_raw = []
                for i in range(len(graphs)):
                    edge_attr_np = to_np(graphs[i][1][0])
                    edge_attr_gt_np = graph_gt_ret[1]

                    if args.env in ['Ball']:
                        idx_rel = np.argmax(graph_gt_ret[0], axis=2)
                        idx_empty = np.logical_and(idx_rel == 0, np.logical_not(np.eye(n_kp)))
                        idx_spring = np.logical_and(idx_rel == 1, np.logical_not(np.eye(n_kp)))
                        idx_rod = np.logical_and(idx_rel == 2, np.logical_not(np.eye(n_kp)))

                        cor_over_time_raw.append([
                            [edge_attr_np[idx_empty], edge_attr_gt_np[idx_empty]],
                            [edge_attr_np[idx_spring], edge_attr_gt_np[idx_spring]],
                            [edge_attr_np[idx_rod], edge_attr_gt_np[idx_rod]]])

                over_time_results = acc_over_time, ent_over_time, cor_over_time_raw, \
                        recall_over_time, pre_over_time, f1_over_time

            else:
                # record the entropy over edge type over time
                ent_over_time = np.zeros(len(graphs))
                for i in range(len(graphs)):
                    edge_type_logits_cur = graphs[i][3][0]

                    # entropy
                    ent = F.softmax(edge_type_logits_cur, dim=2) * F.log_softmax(edge_type_logits_cur, dim=2)
                    ent = -ent.sum(2)
                    ent = ent.mean().item()
                    ent_over_time[i] = ent


                over_time_results = ent_over_time

        # the current keypoints
        eps = 5e-2
        kp_cur = kps[:args.n_his].view(1, args.n_his, args.n_kp, 2)
        covar_gt = torch.FloatTensor(np.array([eps, 0., 0., eps])).cuda()
        covar_gt = covar_gt.view(1, 1, 1, 4).repeat(1, args.n_his, args.n_kp, 1)
        kp_cur = torch.cat([kp_cur, covar_gt], 3)

    loss_kp_acc = 0.
    n_roll = args.eval_ed_idx - args.eval_st_idx - args.n_his


    eval_txt = open(os.path.join(args.evalf, str(roll_idx)+'.txt'), 'w')

    for i in range(args.eval_ed_idx - args.eval_st_idx):

        if args.stage == 'dy':

            if i >= args.n_his:

                with torch.set_grad_enabled(False):
                    # predict the feat and hmap at the next time step
                    if actions is not None:
                        action_cur = actions[i-args.n_his+args.eval_st_idx:i+args.eval_st_idx].unsqueeze(0)
                    else:
                        action_cur = None
                    kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur, env=args.env)

                    mean_pred, covar_pred = kp_pred[:, :, :2], kp_pred[:, :, 2:].view(1, n_kp, 2, 2)

                # compare with the ground truth
                kp_des = kps[i : i + 1]

                loss_kp = criterionMSE(mean_pred, kp_des) * args.lam_kp

                fwd_loss_mse_cur.append(loss_kp.item())


                loss_kp_acc += loss_kp.item()


                # update feat_cur and hmap_cur
                kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                # img_pred & heatmap
                keypoint = mean_pred
                keypoint_covar = covar_pred
                keypoint_gt = kp_des
            else:
                kp_cur_t = kps[i:i+1]

                keypoint = kp_cur_t
                keypoint_covar = covar_gt[:, -1].view(1, n_kp, 2, 2)
                keypoint_gt = kp_cur_t

            if i >= args.n_his:
                eval_txt.write('GT: {}\n'.format(to_np(keypoint_gt[0])))
                eval_txt.write('PR: {}\n'.format(to_np(keypoint[0])))
                img = np.zeros((120, 120, 3), np.uint8) + 255
                # (10, 10) is free space on the border 

                c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
                    (128, 128, 128), (30, 105, 210), (147, 20, 255), (205, 90, 106), (0, 215, 255),
                    (65, 65, 225), (225, 185, 65)]

                lim = args.lim
                keypoint = to_np(keypoint)[0] - [lim[0], lim[2]]
                keypoint = (keypoint-keypoint.min()) / (keypoint.max()-keypoint.min()) 
                keypoint *= args.height_raw / 2.
                keypoint = np.round(keypoint).astype(np.int)
                keypoint_covar = to_np(keypoint_covar[0])


                keypoint_gt = to_np(keypoint_gt)[0] - [lim[0], lim[2]]
                keypoint_gt = (keypoint_gt-keypoint_gt.min()) / (keypoint_gt.max()-keypoint_gt.min()) 
                keypoint_gt *= args.height_raw
                keypoint_gt = np.round(keypoint_gt).astype(np.int)
                if args.env in ['Ball']:
                    for j in range(keypoint.shape[0]):
                        cv2.circle(img, (keypoint[j, 0]+5, keypoint[j, 1]+5), 4, c[j], 2, cv2.LINE_AA)
                        cv2.circle(img, (keypoint_gt[j, 0]+5, keypoint_gt[j, 1]+5), 4, (0,0,0), 1)

                merge = img

                c = ['orangered', 'royalblue', 'limegreen', 'orange', 'violet',
                        'gray', 'chocolate', 'deeppink', 'slateblue', 'gold', 'darkcyan', 'darkorchid']

                if False:
                    draw_graph(
                        keypoint,
                        edge_type=np.argmax(to_np(
                            edge_type_logits.view(args.n_kp, args.n_kp, args.edge_type_num)), -1),
                        lim=lim,
                        c=c,
                        file_name = os.path.join(eval_path, 'graph_pred_%d.png' % i))

                    draw_graph(
                        keypoint,
                        edge_type=np.argmax(to_np(edge_type_gt), -1),
                        lim=lim,
                        c=c,
                        file_name = os.path.join(eval_path, 'graph_gt_%d.png' % i))
                if video:
                    out.write(merge)

    if video:
        out.release()
        eval_txt.close()


    if args.env in ['Ball']:
        return graph_gt_ret, graph_pred_ret, over_time_results, np.array(fwd_loss_mse_cur)
    elif args.env in ['Cloth']:
        return graph_pred_ret, over_time_results, np.array(fwd_loss_mse_cur)


ls_rollout_idx = np.arange(len(os.listdir(data_dir)))



edge_acc_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
edge_ent_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
edge_cor_over_time_raw_record = []

f1_acc_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
recall_acc_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
pre_acc_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))

fwd_loss_mse = []

for roll_idx in ls_rollout_idx:

    if args.env in ['Ball']:
        graph_gt, graph_pred, over_time_results, fwd_loss_mse_cur = evaluate(
            roll_idx, video=args.store_demo, image=args.store_demo)
    elif args.env in ['Cloth']:
        gt_pred, over_time_results, fwd_loss_mse_cur = evaluate(
            roll_idx, video=args.store_demo, image=args.store_demo)

    fwd_loss_mse.append(fwd_loss_mse_cur)

    if args.env in ['Ball']:
        edge_acc_over_time_record[roll_idx] = over_time_results[0]
        edge_ent_over_time_record[roll_idx] = over_time_results[1]
        f1_acc_over_time_record[roll_idx] = over_time_results[5]
        recall_acc_over_time_record[roll_idx] = over_time_results[3]
        pre_acc_over_time_record[roll_idx] = over_time_results[4]
        edge_cor_over_time_raw_record.append(over_time_results[2])
    elif args.env in ['Cloth']:
        edge_ent_over_time_record[roll_idx] = over_time_results

fwd_loss_mse = np.array(fwd_loss_mse)
print('fwd_loss_mse', fwd_loss_mse.shape)
for i in range(fwd_loss_mse.shape[1]):
    print(i, np.mean(fwd_loss_mse[:, i]), np.std(fwd_loss_mse[:, i]))

print('Print all F1 score')
for i in range(f1_acc_over_time_record.shape[0]):
    print('%2.6f' % np.mean(f1_acc_over_time_record[i,:]))

edge_acc_over_time_record = np.array(edge_acc_over_time_record)
print('edge accuracy',
      edge_acc_over_time_record.shape,
      np.mean(edge_acc_over_time_record[:, -1]),
      np.std(edge_acc_over_time_record[:, -1]))

print('Recall: %2.6f Precision: %2.6f F1Score: %2.6f' % (\
    np.mean(recall_acc_over_time_record), \
    np.mean(pre_acc_over_time_record), \
    np.mean(f1_acc_over_time_record))
)

tee.flush()
tee.close()
