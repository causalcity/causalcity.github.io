import h5py
import pandas as pd
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
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

    # Velocity: dx, dy
    vel = np.zeros([states.shape[0], ncar, 2])
    vel[1:-1,:,:] = states[2:,:,:]-states[1:-1,:,:]
    vel[-1,:,:] = vel[-2,:,:]
  
    states = np.concatenate([states, vel], axis=-1)

    # Make relation graph
    diag_cls = np.eye(ncar)
    diag_attr = np.eye(ncar)
    for k in range(args.nconnect):
        diag_cls[caridx[2*k], caridx[2*k+1]] = 1.
        diag_cls[caridx[2*k+1], caridx[2*k]] = 1.
        diag_attr[caridx[2*k], caridx[2*k+1]] = np.mean(np.linalg.norm(states[:, caridx[2*k], :2] - states[:, caridx[2*k+1], :2], axis=1))
        diag_attr[caridx[2*k+1], caridx[2*k]] = np.mean(np.linalg.norm(states[:, caridx[2*k], :2] - states[:, caridx[2*k+1], :2], axis=1))

    rels = np.zeros([states.shape[0], int(ncar*(ncar-1)/2), 2])
    rels[:, :, 0] = diag_cls[np.triu_indices(ncar,k=1)]
    rels[:, :, 1] = diag_attr[np.triu_indices(ncar,k=1)]

    # Zero attribute in CausalCity
    attrs = np.zeros([states.shape[0], ncar, 1]) + 1.

    # Actions (acceleration): ddx, ddy
    actions = np.zeros([states.shape[0], ncar, 2])
    actions[:-2, :, :] = np.diff(states, n=2, axis=0)[:, :, :2]
    actions[-1, :, :] = actions[-3,:,:]
    actions[-2, :, :] = actions[-3,:,:]
    outdata = [attrs, states, actions, rels]

    return outdata

def write(h5name, args, data):
    hf = h5py.File(h5name, 'w')
    for i in range(len(args.data_names)):
        hf.create_dataset(args.data_names[i], data=data[i])
    hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncar', default=12, type=int)
    parser.add_argument('--nconnect', default=3, type=int)
    parser.add_argument('--infolder', default="Dataset_ICML_Final_Toy_V2", help='Input directory name')
    parser.add_argument('--outfolder', default="Final001", help='Output directory name')
    parser.add_argument('--data_names', default=['attrs', 'states', 'actions', 'rels'])
    args = parser.parse_args()


    args.outfolder = args.outfolder + '_car' + str(args.ncar) + '_edge' + str(args.nconnect)
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder+'/train')
        os.makedirs(args.outfolder+'/test')
        os.makedirs(args.outfolder+'/valid')

    k_train = 0
    k_val = 0
    k_test = 0
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
                        outfolder = args.outfolder+'/train'
                        h5name = os.path.join(outfolder, '%d.h5' % k_train)
                        write(h5name, args, data)
                        k_train += 1
                    elif i == 'Valid':
                        outfolder = args.outfolder+'/valid'
                        h5name = os.path.join(outfolder, '%d.h5' % k_val)
                        write(h5name, args, data)
                        k_val += 1
                    else:
                        outfolder = args.outfolder+'/test'
                        h5name = os.path.join(outfolder, '%d.h5' % k_test)
                        write(h5name, args, data)
                        k_test += 1
                except:
                    continue
