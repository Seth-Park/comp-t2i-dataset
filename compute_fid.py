import os
import pickle
import torch
import random

from utils import gather_by_pair, balance_candidates
from fid.fid_score import calculate_fid_given_files
from fid.inception import InceptionV3

if __name__ == "__main__":
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
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

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

    fid_value = calculate_fid_given_files(
        random.sample(gen_image_paths, 10000),
        gt_image_paths,
        args.batch_size,
        device,
        args.dims,
        num_workers
    )

    print("FID score: ")
    print(f"\t {fid_value:.2f}")
