import pickle

from utils import gather_by_pair, balance_candidates


if __name__ == "__main__":
    dataset = "C_CUB"
    comp_type = "color"
    split = "test_swapped"

    anno_data_path = f"./data/{dataset}/data.pkl"
    split_path = f"./data/{dataset}/split.pkl"

    result_path = ""

    with open(result_path, "rb") as f:
        result = pickle.load(f)

    with open(anno_data_path, "rb") as f:
        anno_data = pickle.load(f)

    if split == "test_swapped":
        gathered_results = gather_by_pair(result, anno_data)
        candidates = balance_candidates(dataset, comp_type, gathered_results)

    else:








