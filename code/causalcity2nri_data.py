import h5py
import pandas as pd
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import pdb
import random

def read(fname, args):
    data = pd.read_csv(fname, header=None)
    ncar = args.ncar

    states, actions = [], []
    for i in range(2*args.nconnect):
        pixel_x = data[8*i+5][2:]
        pixel_y = data[8*i+6][2:]
        pixel_x = np.array(pixel_x, np.float)
        pixel_y = np.array(pixel_y, np.float)

        pixel_x[pixel_x < 0] = 0
        pixel_y[pixel_y < 0] = 0

        pixel_x[pixel_x > 1024.0] = 1024.0
        pixel_y[pixel_y > 640.0] = 640.0

        tmp_states = np.concatenate([[pixel_x], [pixel_y]], axis=0)
        states.append(np.transpose(tmp_states))
   
    for i in range(ncar-2*args.nconnect):
        pixel_x = data[48+8*i+5][2:]
        pixel_y = data[48+8*i+6][2:]
        pixel_x = np.array(pixel_x, np.float)
        pixel_y = np.array(pixel_y, np.float)

        pixel_x[pixel_x < 0] = 0
        pixel_y[pixel_y < 0] = 0

        pixel_x[pixel_x > 1024.0] = 1024.0
        pixel_y[pixel_y > 640.0] = 640.0

        tmp_states = np.concatenate([[pixel_x], [pixel_y]], axis=0)
        states.append(np.transpose(tmp_states))

    # Random shuffle of the order of cars
    caridx = [i for i in range(ncar)]
    random.shuffle(caridx)

    # States: pixel_x, pixel_y
    states = np.array(states, dtype=np.float)
    states = states[caridx, :, :]
    states = np.transpose(states, (1, 0, 2))

    # Make relation graph
    diag_cls = np.eye(ncar)
    for k in range(args.nconnect):
        diag_cls[caridx[2*k], caridx[2*k+1]] = 1.
        diag_cls[caridx[2*k+1], caridx[2*k]] = 1.

    # Velocity: dx, dy
    actions = np.zeros([states.shape[0], ncar, 2])
    actions[:-1, :, :] = np.diff(states, n=1, axis=0)[:,:,:2]
    actions[-1, :, :] = actions[-2,:,:]

    outdata = [states, actions, diag_cls]

    return outdata

def write(npyname, data):
    np.save(npyname, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncar', default=12, type=int)
    parser.add_argument('--nconnect', default=3, type=int)
    parser.add_argument('--infolder', default="Dataset_ICML_Final")
    parser.add_argument('--outfolder', default="NRI_Final001")
    parser.add_argument('--data_names', default=['attrs', 'states', 'actions', 'rels'])
    args = parser.parse_args()


    args.outfolder = args.outfolder + '_car' + str(args.ncar) + '_edge' + str(args.nconnect)
    os.makedirs(args.outfolder)

    k_train = 0
    k_val = 0
    k_test = 0
    loc = {'train':[], 'valid':[], 'test':[]}
    vel = {'train':[], 'valid':[], 'test':[]}
    edge = {'train':[], 'valid':[], 'test':[]}

    flist = os.listdir(args.infolder)
    for i in flist:
        vlist = os.listdir(os.path.join(args.infolder,i))
        vlist.sort()
        for j, v in tqdm(enumerate(vlist)):
            fname = os.path.join(args.infolder, i, v)
            data = read(fname, args)
            if not np.isnan(data[1]).any():
                try:
                    if i == 'Train':
                        k_train += 1
                        loc['train'].append(data[0])
                        vel['train'].append(data[1])
                        edge['train'].append(data[2])
                    elif i == 'Valid':
                        k_val += 1
                        loc['valid'].append(data[0])
                        vel['valid'].append(data[1])
                        edge['valid'].append(data[2])
                    else:
                        k_test += 1
                        loc['test'].append(data[0])
                        vel['test'].append(data[1])
                        edge['test'].append(data[2])
                except:
                    continue

    # Write Files
    for typename in ['train', 'valid', 'test']:
        locname = os.path.join(args.outfolder, 'loc_'+typename+'_airsim_final.npy')
        velname = os.path.join(args.outfolder, 'vel_'+typename+'_airsim_final.npy')
        edgename = os.path.join(args.outfolder, 'edges_'+typename+'_airsim_final.npy')
        write(locname, np.stack(loc[typename], 0))
        write(velname, np.stack(vel[typename], 0))
        write(edgename, np.stack(edge[typename], 0))
