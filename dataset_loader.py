import os
import json
from datasets import load_dataset


# =========================
# COCO LOADER
# =========================

def load_coco_captions(image_root, annotation_file):

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    id_to_filename = {
        img["id"]: img["file_name"]
        for img in data["images"]
    }

    image_paths = []
    captions = []

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]

        image_path = os.path.join(image_root, id_to_filename[image_id])

        image_paths.append(image_path)
        captions.append(caption)

    return image_paths, captions


# =========================
# FLICKR30K LOADER
# =========================

def load_flickr30k():
    images_dir = "datasets/flickr_images"
    cache_file = "datasets/flickr30k_local_cache.json"

    # Optimization: If we have already processed the dataset, load from local JSON
    # This prevents re-downloading/re-processing from Hugging Face on every run (critical for Colab)
    if os.path.exists(cache_file) and os.path.exists(images_dir):
        print(f"Loading Flickr30k from local cache: {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return data["image_paths"], data["captions"]

    # Using lmms-lab/flickr30k which is compatible with current datasets library
    print("Downloading/Loading Flickr30k from Hugging Face...")
    dataset = load_dataset("lmms-lab/flickr30k", split="test")

    image_paths = []
    captions = []

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    print("Extracting images and captions (this may take a while first time)...")
    for sample in dataset:
        img = sample["image"]
        caps = sample["caption"]
        filename = sample["filename"]

        # Save image locally (required for our pipeline)
        image_path = f"{images_dir}/{filename}"

        if not os.path.exists(image_path):
             img.save(image_path)

        for caption in caps:
            image_paths.append(image_path)
            captions.append(caption)

    # Save cache for next time
    print(f"Saving local cache to {cache_file}...")
    with open(cache_file, 'w') as f:
        json.dump({"image_paths": image_paths, "captions": captions}, f)

    return image_paths, captions