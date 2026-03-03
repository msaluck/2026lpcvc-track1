import json
import os

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