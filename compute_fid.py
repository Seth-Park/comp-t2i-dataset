import os
import pickle

from utils import gather_by_pair, balance_candidates
from fid import calculate_fid_given_files

if __name__ == "__main__":
    """
    python compute_fid.py --dataset C-CUB --comp_type color --split test_swapped \
    --gt_img_root /home/dhpseth/scratch/control_gan_data/birds/CUB_200_2011/images \
    --pred_path /shared/dhpseth/comp-DMGAN/output/comp_birds_a_b_color_DMGAN_2020_11_01_19_29_59/Model/netG_epoch_800/test_seen_swapped++/results.pkl \
    --gpu 0
    """
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="", type=str,
                        help="Dataset to run [C_CUB, C_Flowers].")
    parser.add_argument("--comp_type", default="", type=str,
                        help="Type of composition [color, shape].")
    parser.add_argument("--split", default="", type=str,
                        help="Test split to use [test_seen, test_unseen, test_swapped].")
    parser.add_argument("--gt_img_root", default="", type=str,
                        help="Root directory to the groundtruth images.")
    parser.add_argument("--pred_path", default="", type=str,
                        help="Path to the generated image results or their .npz statistics file.")
    parser.add_argument("--gt_path", default="", type=str,
                        help="Path to the groundtruth image .npz statistics file (will be computed if not provided).")
    parser.add_argument("--gpu", default="", type=str,
                        help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--inception", type=str, default=None,
                        help="Path to Inception model (will be downloaded if not provided)")
    parser.add_argument("--lowprofile", action="store_true",
                        help="Keep only one batch of images in memory at a time. This reduces memory footprint, but may decrease speed slightly.")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    anno_data_path = f"./data/{args.dataset}/{args.comp_type}/data.pkl"
    split_path = f"./data/{args.dataset}/{args.comp_type}/split.pkl"

    # separate result files for each split
    with open(args.pred_path, "rb") as f:
        result = pickle.load(f)

    with open(anno_data_path, "rb") as f:
        anno_data = pickle.load(f)

    with open(split_path, "rb") as f:
        split_data = pickle.load(f)

    if args.split == "test_swapped":
        split_ids = split_data["test_seen"]
    else:
        split_ids = split_data[args.split]

    gt_image_paths = []
    for image_id in split_ids:
        gt_image_paths.append(os.path.join(args.gt_img_root, image_id + ".jpg"))

    if args.split == "test_swapped":
        gathered_results = gather_by_pair(result, anno_data)
        candidates = balance_candidates(args.dataset, args.comp_type, gathered_results)
    else:
        candidates = []
        for entry in result:
            img_id, cap_id, gen_img_path, r_precision_prediction = entry
            candidates.append({
                "prediction": r_precision_prediction,
                "img_path": gen_img_path,
                "text": anno_data[img_id][cap_id]["text"]
            })

    gen_image_paths = []
    for cand in candidates:
        gen_image_paths.append(cand["img_path"])

    fid_value = calculate_fid_given_files(gen_image_paths, gt_image_paths,
                                          args.inception, low_profile=args.lowprofile)

    print("FID score: ")
    print(f"\t {fid_value:.2f}")