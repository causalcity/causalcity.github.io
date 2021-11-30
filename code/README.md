## Requirements
    Python 3.7

## Generating scenarios
    secnarioGenerator.py can be used to create a JSON configuration file programatically. 

## Evaluation
    python causalcity2nri_data.py --ncar 12 --nconnect 3
    python causalcity2vcdn_data.py --ncar 12 --nconnect 3
    
    python NRI/train.py --suffix ${dataname} --num-atoms 12 --exp-name ${outfoldername}
    python NRI/test.py --suffix ${dataname} --num-atoms 12 --exp-name ${outfoldername}
  
    python VCDN/train.py --dataf ${dataname} --n_kp 12 
    python VCDN/test.py --dataf ${dataname} --n_kp 12

## Reference
    Causal Discovery in Physical Systems from Videos
    Li, Y., Torralba, A., Anandkumar, A., Fox, D., & Garg, A. (2020). Causal discovery in physical systems from videos. arXiv preprint arXiv:2007.00631.
    
    Neural Relational Inference for Interacting Systems
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). Neural relational inference for interacting systems. In International Conference on Machine Learning (pp. 2688-2697). PMLR.
