import random

from collections import Counter, defaultdict


def gather_by_pair(result, anno_data):

    gathered_result = defaultdict(list)

    for entry in result:
        img_id, cap_id, gen_img_path, r_precision_prediction = entry
        data = anno_data[img_id]
        if cap_id in data:
            caption_data = data[cap_id]
        else:
            continue
        text = caption_data["swapped_text"]
        changes = caption_data["changes_made"]
        new_adj = changes["new_adj"]
        noun = changes["noun"]
        pair = f"{new_adj}_{noun}"
        gathered_result[pair].append({
            "prediction": r_precision_prediction,
            "img_path": gen_img_path,
            "text": text
        })

    return gathered_result


def balance_candidates(dataset, comp_type, gather_by_pair):
    change_type_counts = Counter()
    for change_type, data in gather_by_pair.items():
        change_type_counts[change_type] = len(data)
    top3 = change_type_counts.most_common(3)

    """
    for CUB color split
    """
    if dataset == 'C_CUB' and comp_type == 'color':
        max_num_dominant = int(min(top3[0][1], 1.25 * top3[-1][1]))
        dominant_1 = top3[0][0]
        dominant_2 = top3[1][0]

        dominant_1_cands = random.sample(gather_by_pair[dominant_1], max_num_dominant)
        dominant_2_cands = random.sample(gather_by_pair[dominant_2], max_num_dominant)

        all_cands = []
        for pair, entries in gather_by_pair.items():
            if pair == dominant_1 or pair == dominant_2:
                continue
            all_cands += entries
        all_cands += dominant_1_cands + dominant_2_cands
    ####################################################################
    else:
        max_num_dominant = int(min(top3[0][1], 1.25 * top3[1][1]))
        dominant = top3[0][0]

        dominant_cands = random.sample(gather_by_pair[dominant], max_num_dominant)
        all_cands = []
        for pair, entries in gather_by_pair.items():
            if pair == dominant:
                continue
            all_cands += entries
        all_cands += dominant_cands
    #####################################################################
    return all_cands