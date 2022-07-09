# RA-Depth

This repo is for **[RA-Depth: Resolution Adaptive Self-Supervised Monocular Depth Estimation], ECCV2022**


If you think it is a useful work, please consider citing it.
```
@inproceedings{ra_depth,
    title={RA-Depth: Resolution Adaptive Self-Supervised Monocular Depth Estimation},
    author={Mu, He and Le, Hui and Beibei, Zhou and Kun, Wang and Yikai, Bian and Jian, Ren and Jin, Xie and Jian, Yang},
    booktitle={ECCV},
    year={2022}
    }

```

## Overview of RA-Depth
![](assets/pipeline.png)


## Basic results on KITTI dataset
![](assets/results1.png)


## Resolution Adaptation Results
![](assets/results2.png)


##  Ablation Study
![](assets/results3.png)


## Visualization Results of Resolution Adaptation
![](assets/visuals.png)


## Training:

```
CUDA_VISIBLE_DEVICES=0 python train.py --model_name RA-Depth --scales 0 --png --log_dir models --data_path /datasets/Kitti/Kitti_raw_data
```


## Testing:

```
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /models/RA-Depth/ --eval_mono --height 192 --width 640 --scales 0 --data_path /datasets/Kitti/Kitti_raw_data --png
```


#### Acknowledgement
 Thanks the authors for their works:
 - [monodepth2](https://github.com/nianticlabs/monodepth2)
 - [DIFFNet](https://github.com/brandleyzhou/DIFFNet)
 
 
