from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from ptflops import get_model_complexity_info
from thop import profile

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/EMDepth00_abs1_1_1024/models/weights_19/ --eval_mono --height 320 --width 1024 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MRDepth00/models/weights_0/ --eval_mono --height 192 --width 640 --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth/models/weights_19/ --eval_mono --height 192 --width 640 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/models/mono_640x192/ --height 128 --width 416 --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_416x128/models/weights_19/ --height 128 --width 416 --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_hrdepth_416x128/models/weights_19/ --height 128 --width 416 --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_baseline_nopt/models/weights_19/ --eval_mono --height 192 --width 640 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_abs02/models/weights_19/ --eval_mono --height 192 --width 640 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_bian/models/weights_19/ --eval_mono --height 192 --width 640 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_baseline_nopt/models/weights_19/ --eval_mono --height 128 --width 416 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_nopt/models/weights_19/ --eval_mono --height 128 --width 416 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/RA-Depth_sf1/models/weights_19/ --eval_mono --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/models/mono_640x192/ --eval_split improved_eigen --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MRDepth10/models/weights_19/ --eval_split improved_eigen --eval_mono --height 192 --width 640 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    # # compute the parameters and flops
    # unet = networks.UNet()
    # macs, params = get_model_complexity_info(unet, (3,192,640), as_strings=True,
    #                                    print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(hhh)

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files_right.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0], 4, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        encoder = networks.hrnet18(False)
        
        depth_decoder = networks.DepthDecoder_MSF(encoder.num_ch_enc, opt.scales, num_output_channels=1)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            opt.width, opt.height))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                # input_color = data[("color_MiS", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp, _ = disp_to_depth(output[("disp", 0)][:,0,:,:].unsqueeze(1), opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths_right.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        if gt_depth is None:
            continue
        gt_height, gt_width = gt_depth.shape[:2] #[375, 1242]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if gt_depth.shape[0] == 0:
            continue

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        ratio_max = ratios.max()
        ratio_min = ratios.min()
        ratio_mean = ratios.mean()
        print("Scaling ratios | mean: {:0.3f} | min: {:0.3f} | max: {:0.3f}".format(ratio_mean, ratio_min, ratio_max))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())



#0511
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet07/models/weights_19/ --eval_mono --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_resnet50/models/weights_17/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MS21_freezepose44_1/models/weights_19/ --scales 0 --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_depth50_pose18/models/weights_17/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet10/models/weights_19/ --scales 0 --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_depth50_pose50/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet13/models/weights_2/ --scales 0 1 2 3 4 --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/models/mono+stereo_1024x320/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png --width 1024  --height 320
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_densefusion/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png



#python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_640x192/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_stereo_640x192_1/models/weights_8/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_stereo_1024x320/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_stereo_disp/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_640x192/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/models/stereo_640x192/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MS_lr04/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png

#0912
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_06/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_09/models/weights_16/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_00/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_640x192/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png

#0928
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono01/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_convex01/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_convex00_1/models/weights_9/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_convex03/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_1024x320/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_09/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_occlu01/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_occlu02_1/models/weights_19/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_occlu02_2/models/weights_17/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_occlu03_2/models/weights_18/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_geo00/models/weights_1/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_occlu03_1_1/models/weights_4/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/stereo_resnet50/models/weights_0/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png --num_layers 50
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono_cycleconsis01/models/weights_0/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MS_lr17_/models/weights_0/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png --scales 0
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/mono21_01/models/weights_0/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png --scales 0
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MS21_freezepose01/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MS_lr_in_right_continue/models/weights_15/ --eval_stereo --data_path /test/datasets/Kitti/Kitti_raw_data --png

#0313
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono01/models/weights_19/ --eval_mono --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet14/models/weights_12/ --eval_stereo  --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet11_rep1/models/weights_16/ --eval_mono  --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug/models/weights_19/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug05_1024/models/weights_6/ --eval_mono --height 320 --width 1024 --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug05_1/models/weights_0/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis/models/weights_0/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis13/models/weights_16/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis05_2/models/weights_18/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis19/models/weights_27/ --eval_mono --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis25/models/weights_29/ --eval_mono --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M3Depth_0/models/weights_0/ --eval_stereo --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug05_1024/models/weights_29/ --eval_mono --height 320 --width 1024 --scales 0 1 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis28_1024/models/weights_8/ --eval_mono --height 320 --width 1024 --scales 0 --data_path /test/datasets/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/MMB_Depth05/models/weights_5/ --scales 0 --eval_stereo --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/M21_wnet47_aug_consis28_1024/models/weights_19/ --eval_mono --height 320 --width 1024 --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/EMDepth00_abs2/models/weights_19/ --eval_mono --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/EMDepth00_baseline_scale4_continue/models/weights_7/ --eval_mono --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/EMDepth00_abs1_1/models/weights_14/ --eval_mono --scales 0 --data_path /opt/data/common/Kitti/Kitti_raw_data --png
#CUDA_VISIBLE_DEVICES=0 python3 evaluate_depth.py --load_weights_folder /test/monodepth2-master/hm_train/hm_mono_1024x320/models/weights_19/ --eval_mono --height 320 --width 1024 --data_path /test/datasets/Kitti/Kitti_raw_data --png
