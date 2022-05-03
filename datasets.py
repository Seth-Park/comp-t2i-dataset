from collections import defaultdict

import torch
import torch.utils.data as data

import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import pickle

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, comp_type, split,
                 image_transform=None, tokenizer=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.comp_type = comp_type
        self.split = split

        anno_data_path = os.path.join(data_dir, dataset_name, comp_type, "data.pkl")
        with open(anno_data_path, "rb") as f:
            self.anno_data = pickle.load(f)

        split_path = os.path.join(data_dir, dataset_name, comp_type, "split.pkl")
        with open(split_path, "rb") as f:
            self.split_data = pickle.load(f)

        self.image_dir = os.path.join(data_dir, dataset_name, "images")
        if split == "test_swapped":
            self.image_ids = self.split_data["test_seen"]
        else:
            self.image_ids = self.split_data[split]

        self.image_id_to_valid_caption_ids = defaultdict(list)
        for image_id in self.image_ids:
            for caption_id, caption_data in self.anno_data[image_id].items():
                if self.split == "test_swapped":
                    if "swapped_text" in caption_data:
                        self.image_id_to_valid_caption_ids[image_id].append(caption_id)
                else:
                    self.image_id_to_valid_caption_ids[image_id].append(caption_id)

        self.image_id_to_class_id = self.load_class_ids()

        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def preprocess_image(self, image, image_id):
        return image

    def preprocess_text(self, text):
        return text

    def load_class_ids(self):
        image_id_to_class_id = {}
        for i, image_id in enumerate(self.image_ids):
            image_id_to_class_id[image_id] = i
        return image_id_to_class_id

    def get_mismatched_caption(self, image_id):
        class_id = self.image_id_to_class_id[image_id]
        mismatch_captions = []
        num_imgs = len(self.image_ids)

        while len(mismatch_captions) < 99:
            sampled_idx = random.randint(0, num_imgs-1)
            sampled_image_id = self.image_ids[sampled_idx]
            sampled_class_id = self.image_id_to_class_id[sampled_image_id]
            if class_id == sampled_class_id:
                continue
            sampled_data = self.anno_data[sampled_image_id]
            sampled_caption_ids = self.image_id_to_valid_caption_ids[sampled_image_id]
            sampled_caption_id = random.choice(sampled_caption_ids)
            sampled_caption_data = sampled_data[sampled_caption_id]

            sampled_text = sampled_caption_data["text"]
            sampled_text = self.preprocess_text(sampled_text)

            if self.tokenizer is not None:
                sampled_text = self.tokenizer(sampled_text)

            mismatch_captions.append(sampled_text)

        mismatch_captions = torch.cat(mismatch_captions, 0).long()

        return mismatch_captions

    def get_text(self, image_id, caption_id, raw=False, swapped=False):
        data = self.anno_data[image_id]
        caption_data = data[caption_id]
        if swapped:
            text = caption_data["swapped_text"]
        else:
            text = caption_data["text"]

        if raw:
            return text
        else:
            text = self.preprocess_text(text)
            if self.tokenizer is not None:
                text = self.tokenizer(text)
            return text

    def get_image(self, image_id, raw=False, preprocess=True):
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        if raw:
            return image
        else:
            if preprocess:
                image = self.preprocess_image(image, image_id)
            if self.image_transform is not None:
                image = self.image_transform(image)
            return image

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess_image(image, image_id)

        data = self.anno_data[image_id]
        caption_ids = self.image_id_to_valid_caption_ids[image_id]
        caption_id = random.choice(caption_ids)
        caption_data = data[caption_id]
        if self.split == "test_swapped":
            text = caption_data["swapped_text"]
        else:
            text = caption_data["text"]
        text = self.preprocess_text(text)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return image, text, image_id, caption_id

    def __len__(self):
        return len(self.image_ids)


class CCUBDataset(BaseDataset):
    def __init__(self,
                 data_dir,
                 dataset_name,
                 comp_type,
                 split,
                 image_transform=None,
                 tokenizer=None,
                 images_txt_path=None,
                 bbox_txt_path=None):
        super().__init__(
            data_dir,
            dataset_name,
            comp_type,
            split,
            image_transform,
            tokenizer
        )

        if images_txt_path is not None and bbox_txt_path is not None:
            self.image_id_to_bbox = self.load_bbox(images_txt_path, bbox_txt_path)
        else:
            self.image_id_to_bbox = None

    def load_bbox(self, images_txt_path, bbox_txt_path):

        df_bounding_boxes = pd.read_csv(bbox_txt_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        df_filenames = pd.read_csv(images_txt_path, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        num_imgs = len(filenames)
        for i in range(num_imgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_class_ids(self):
        image_id_to_class_id = {}
        for image_id in self.image_ids:
            class_id = int(image_id.split('.')[0])
            image_id_to_class_id[image_id] = class_id
        return image_id_to_class_id

    def preprocess_image(self, image, image_id):
        width, height = image.size
        if self.image_id_to_bbox is not None:
            bbox = self.image_id_to_bbox[image_id]
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            image = image.crop([x1, y1, x2, y2])
        return image

    def preprocess_text(self, text):
        return text.strip().lower()


class CFlowersDataset(BaseDataset):
    def __init__(self,
                 data_dir,
                 dataset_name,
                 comp_type,
                 split,
                 image_transform=None,
                 tokenizer=None,
                 class_id_txt_path=None):
        self.class_id_txt_path = class_id_txt_path
        super().__init__(
            data_dir,
            dataset_name,
            comp_type,
            split,
            image_transform,
            tokenizer
        )

    def load_class_ids(self):
        if self.class_id_txt_path is not None:
            image_id_to_class_id = {}
            with open(self.class_id_txt_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_id, class_id = line.split("\t")
                image_id_to_class_id[image_id] = class_id
            return image_id_to_class_id
        else:
            return super().load_class_ids()

    def preprocess_image(self, image, image_id):
        width, height = image.size
        if self.image_id_to_bbox is not None:
            bbox = self.image_id_to_bbox[image_id]
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            image = image.crop([x1, y1, x2, y2])
        return image

    def preprocess_text(self, text):
        return text.strip().lower()