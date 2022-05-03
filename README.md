# Benchmark for Compositional Text-to-Image Synthesis [NeurIPS 2021 Track Datasets and Benchmarks]

This repository provides the benchmark dataset and evaluation code described in [this paper](https://openreview.net/pdf?id=bKBhQhPeKaF).

## Installation
Assuming a conda environment:
```
# NOTE: if you are not using CUDA 10.2, you need to change the 10.2 in this command appropriately. 
# Code tested with torch 1.10
# (check CUDA version with e.g. `cat /usr/local/cuda/version.txt`)
conda create --name comp-t2i-benchmark python=3.7
conda activate comp-t2i-benchmark

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pandas ftfy regex
conda install -c conda-forge pytorch-lightning
```

## Compositional Splits
The compositional splits `C-CUB` and `C-Flowers` can be downloaded from [here](https://drive.google.com/drive/folders/1jOnNp0RkUN_QFa-GvfitbKh8YTthLtgl?usp=sharing). Download the splits and save them under the directory `data`. 

For each split, there are two files, namely `split.pkl` and `data.pkl`. 

`split.pkl` contains image ids for `train`, `test_seen`, and `test_unseen`. Also, it contains information about the heldout pairs. 

`data.pkl` is structured as the following:
```
{
    image_id: {
        caption_id: {
            text: caption
            swapped_text: swapped caption  # only for image_ids in test_seen split and caption_ids that have swappable adjectives
            changes_made: {  # only for image_ids in test_seen split and caption_ids that have swappable adjectives
                noun: noun,
                original_adj: original adjective before the swap,
                new_adj: adjective that was swapped in
                
            }
        }
    }
}
```

Additionally, download the original datasets [Caltech-UCSD Birds-200-2011](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images) and [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and create a symlink to the images:

```
ln -s path/to/CUB_200_2011/images ./data/C-CUB/images
ln -s path/to/oxford-flowers/images ./data/C-Flowers/images
```

## R-Precision Evaluation
The benchmark relies on [DMGAN repo](https://github.com/MinfengZhu/DM-GAN) for R-precision evaluation. To compute R-precision, follow the instructions in the DMGAN repo. We provide the DAMSM encoder weights [here](https://drive.google.com/drive/folders/1rE-tflZ84EDp1ckYiq_NXllRls6x-ohO?usp=sharing).

## Preparing CLIP predictions
Download the CLIP encoders [here](https://drive.google.com/drive/folders/1A2I6uSnrFEgu-7QIpcxjDXE7m5799UCy?usp=sharing) and place them under `clip_weights`.

We provide a script `make_clip_prediction.py` to prepare CLIP retrieval predictions. Note that the script expects a python pickle file with DAMSM predictions (e.g. `damsm_predictions.pkl`) which you can generate while running the R-precision evaluation. The file should contain the following contents:

```
[(image_id, caption_id, generated_image_path, DAMSM_prediction), ...]
```
`DAMSM_prediction` indicates the index of the retrieved text as computed by the DAMSM encoders. During R-precision evaluation, 100 captions are sampled given the generated image in which the 0th caption is the groundtruth (the actual caption used to generate the image). Therefore, the correct prediction will be `0`. We provide an example file in `predictions/C_CUB_color_test_swapped_damsm_predictions.pkl`.

If you do not wish to generate DAMSM predictions, you can prepare the pickle file by simply filling `DAMSM_prediction` with random values. The script simply takes `image_id` and `caption_id` to sample 100 captions, computes similarity scores using the pretrained CLIP encoders, and replace `DAMSM_prediction` with CLIP retrieval results.

We provide an example command below:
```
python make_clip_prediction.py \
    --dataset C-CUB \
    --comp_type color \
    --split test_swapped \
    --ckpt clip_weights/C-CUB/C_CUB_color.pt \
    --gpu 1 \
    --pred_path predictions/C_CUB_color_test_swapped_damsm_predictions.pkl \
    --out_path predictions/C_CUB_color_test_swapped_clip_predictions.pkl
```

## Computing CLIP-R-Precision
Once the CLIP predictions are ready, run the following command to compute CLIP-R-precision:
```
python compute_r_precision.py \ 
    --dataset C-CUB \
    --comp_type color \
    --split test_swapped \ 
    --pred_path predictions/C_CUB_color_test_swapped_clip_predictions.pkl
```

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](./LICENSE).