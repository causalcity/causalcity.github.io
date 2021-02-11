## Requirements
    Python 3.7

## Usage
    python causalcity2nri_data.py --ncar 12 --nconnect 3
    python causalcity2vcdn_data.py --ncar 12 --nconnect 3
    
    python NRI/train.py --suffix ${dataname} --num-atoms 12 --exp-name ${outfoldername}
    python NRI/test.py --suffix ${dataname} --num-atoms 12 --exp-name ${outfoldername}
  
    python VCDN/train.py --dataf ${dataname} --n_kp 12 
    python VCDN/test.py --dataf ${dataname} --n_kp 12

## Reference
    Causal Discovery in Physical Systems from Videos
    Neural Relational Inference for Interacting Systems
