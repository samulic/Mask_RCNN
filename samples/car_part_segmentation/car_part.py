import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import random
import numpy as np
from pathlib import Path
import scipy.io as sio
import tensorflow as tf
import maskrcnn.model as modellib
from maskrcnn import utils
from maskrcnn.config import Config
import imgaug.augmenters as iaa
from tqdm import tqdm
import json


np.random.seed = 42
random.seet = 42
tf.set_random_seed(42)

def extract_annotations(path):
    # print(annotation_path)
    annotations = sio.loadmat(path)['anno']
    objects = annotations[0, 0]['objects']

    # list containing all the objects in the image
    objects_list = []

    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        classname = obj['class'][0]
        mask = obj['mask']

        parts_list = []
        parts = obj['parts']

        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            part_name = part['part_name'][0]
            part_mask = part['mask']

            parts_list.append({'part_name': part_name, 'mask': part_mask})

        objects_list.append(
            {'class_name': classname, 'mask': mask, "parts": parts_list})

    return objects_list


def preprocess_dataset(images_path, annotations_path, filter={'car'}):
    """Process the dataset returning a list of tuple with
        (file_name, image_path, mask_list, class_list)

        Args:
        images_path -- the folder containing the images of the dataset
        annotations_path -- the folder containing the annotations for the dataset
        classes -- a set with the classes to process

        Returns:
            a tuple with:
                a list of ennuples (file_name, image_path, mask_list, class_list)
    """
    images_path = Path(images_path)

    class_names = set()
    results = list()

    for path in tqdm(annotations_path):
        # get the annotations
        image_objs = extract_annotations(path)

        # get the immage path
        file_name = path.name.replace('mat', 'jpg')
        image_path = images_path / file_name

        mask_list = []
        class_list = []

        for obj in image_objs:
            if obj['class_name'] in filter:
                if 'parts' in obj:
                    for part in obj['parts']:
                        # handle the mask
                        mask_list.append(part['mask'].astype(bool))

                        # handle the class name
                        part_name = part['part_name']
                        class_list.append(part_name)
                        class_names.add(part_name)

        if len(mask_list):
            # reshape the mask list
            mask_list = np.array(mask_list)
            mask_list = np.moveaxis(mask_list, 0, -1)

            results.append(
                (file_name, image_path, mask_list, class_list)
            )

    class_list = sorted(list(class_names))
    idx_class = dict(enumerate(class_list, 1))
    class_idx = {v: k for k, v in idx_class.items()}

    results_class_idx = []
    for file_name, image_path, mask_list, class_list in results:
        class_idx_list = [class_idx[x] for x in class_list]
        results_class_idx.append(
            (file_name, image_path, mask_list, class_idx_list)
        )

    return results_class_idx, class_idx


def prepare_datasets(images_path, images_annotations_files,
                     train_perc=0.9, val_perc=1.0, filter={'car'}):

    results, parts_idx_dict = preprocess_dataset(
        images_path, images_annotations_files, filter)

    train_split = int(len(results) * train_perc)
    val_split = int(len(results) * val_perc)
    print(
        f'train size {train_split}, val size {val_split - train_split} test size { len(results) - val_split}')

    dataset_train = CarPartDataset()
    dataset_train.load_dataset(parts_idx_dict, results[:train_split])
    dataset_train.prepare()
    dataset_val = CarPartDataset()
    dataset_val.load_dataset(
        parts_idx_dict, results[train_split:val_split])
    dataset_val.prepare()

    dataset_test = CarPartDataset()
    dataset_test.load_dataset(parts_idx_dict, results[val_split:])
    dataset_test.prepare()

    return dataset_train, dataset_val, dataset_test, parts_idx_dict


class CarPartConfig(Config):
    NAME = 'car_parts'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 31  # 26 parts

    # STEPS_PER_EPOCH = 100
    # VALIDATION_STEPS = 10

    # BACKBONE = "resnet50"

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 30

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


class CarPartDataset(utils.Dataset):

    def load_dataset(self, parts_idx_dict, preprocessed_images):
        """

        classes: in case of None it loads all the classes otherwise
            it filter on a particular class
        """
        for part_name, i in parts_idx_dict.items():
            self.add_class('car_parts', i, part_name)

        for file_name, image_path, masks, classes in preprocessed_images:
            # add all the classes classes
            self.add_image(
                "car_parts",
                image_id=file_name,
                path=image_path,
                masks=masks,
                classes=np.array(classes)
            )

    def load_mask(self, image_id):
        # load all the masks from the image id
        info = self.image_info[image_id]
        return info['masks'], info['classes']

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


if __name__ == '__main__':
    from keras import backend as K
    print(K.tensorflow_backend._get_available_gpus())

    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect car parts')
    parser.add_argument('--images_path', required=True,
                        metavar="/path/to/balloon/images/",
                        help='The directory to load the images')

    parser.add_argument('--annotations_path', required=True,
                        metavar="/path/to/balloon/annotations/",
                        help='The directory to load the annotations')

    parser.add_argument('--weights', required=False,
                        help='the weights that can be used, values: imagenet or last')

    parser.add_argument('--checkpoint', required=True,
                        help='the folder where the checkpoints are saved')
    # parser.

    args = parser.parse_args()

    model_checkpoints = args.checkpoint
    print('checkpointing models in folder {}'.format(model_checkpoints))

    print('load the dataset ...')
    images_path = Path(args.images_path)
    annotations_path = list(Path(args.annotations_path).glob('*.mat'))

    dataset_train, dataset_val, dataset_test, parts_idx_dict = prepare_datasets(
        images_path, annotations_path
    )
    print('finished loading the dataset')

    print(parts_idx_dict)
    with open('parts_idx_dict.json', 'w') as f:
        json.dump(parts_idx_dict, f)

    config = CarPartConfig()
    # print(config.display())

    augmentation = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Affine(scale=(1., 2.5), rotate=(-90, 90), shear=(-16, 16), 
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.ContrastNormalization((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0)),
        iaa.LogContrast(gain=(0.6, 1.4)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Clouds(),
        iaa.Noop(),
    ])

    with tf.device('/gpu:0'):
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=model_checkpoints)

        if args.weights == 'imagenet':
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        else:
            model.load_weights(model.find_last(), by_name=True)

        print("Training network heads")
        model.train(dataset_val, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_val, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_val, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
