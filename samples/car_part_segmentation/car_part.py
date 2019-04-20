from mrcnn import utils
from mrcnn.config import Config
import os
import sys
import random
import numpy as np
from pathlib import Path
import scipy.io as sio
import tensorflow as tf
import mrcnn.model as modellib
import imgaug.augmenters as iaa
from tqdm import tqdm
import json


np.random.seed = 42
random.seet = 42
tf.set_random_seed(42)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


def process_annotation(annotation_path):
    # print(annotation_path)
    annotations = sio.loadmat(annotation_path)['anno']
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


def preprocess_dataset(images_path, images_annotations_files):
    images_path = Path(images_path)

    parts_set = set()

    # iterate to create the parts set
    for ann_path in tqdm(images_annotations_files):
        # process the annotations
        img_objects = process_annotation(ann_path)

        # get the image path
        file_name = ann_path.name.replace('mat', 'jpg')
        image_path = images_path / file_name

        for obj in img_objects:
            if obj['class_name'] == 'car':
                # get the car parts
                if 'parts' in obj:
                    for part in obj['parts']:
                        # add the part name
                        part_name = part['part_name']
                        parts_set.add(part_name)

    # transform to a dictionary of sorted part names
    parts_list = sorted(list(parts_set))
    idx_parts = dict(enumerate(parts_list, 1))
    parts_idx = {v: k for k, v in idx_parts.items()}

    results = []

    for ann_path in tqdm(images_annotations_files):
        # process the annotations
        img_objects = process_annotation(ann_path)

        # get the image path
        file_name = ann_path.name.replace('mat', 'jpg')
        image_path = images_path / file_name

        to_predict_classes = []
        to_predict_masks = []

        for obj in img_objects:
            if obj['class_name'] == 'car':
                # get the car parts
                if 'parts' in obj:
                    for part in obj['parts']:
                        # add the part name
                        part_name = part['part_name']
                        to_predict_classes.append(parts_idx[part_name])
                        # add the mask
                        mask = part['mask'].astype(bool)
                        to_predict_masks.append(mask)

        if len(to_predict_classes):
            # reshape the masks as an unique array
            to_predict_masks = np.array(to_predict_masks)
            to_predict_masks = np.moveaxis(to_predict_masks, 0, -1)

            results.append(
                (file_name, image_path, to_predict_masks, to_predict_classes))

    return results, parts_idx


def prepare_datasets(images_path, images_annotations_files, train_perc=0.9, val_perc=1.0):

    inputs_outputs, parts_idx_dict = preprocess_dataset(
        images_path, images_annotations_files)

    train_split = int(len(inputs_outputs) * train_perc)
    val_split = int(len(inputs_outputs) * val_perc)
    print(f'train size {train_split}, test size {val_split}')

    dataset_train = CarPartDataset()
    dataset_train.load_dataset(parts_idx_dict, inputs_outputs[:train_split])
    dataset_train.prepare()
    dataset_val = CarPartDataset()
    dataset_val.load_dataset(
        parts_idx_dict, inputs_outputs[train_split:val_split])
    dataset_val.prepare()

    dataset_test = CarPartDataset()
    dataset_test.load_dataset(parts_idx_dict, inputs_outputs[val_split:])
    dataset_test.prepare()

    return dataset_train, dataset_val, dataset_test, parts_idx_dict


class CarPartConfig(Config):
    NAME = 'car_parts'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

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
        iaa.Noop(),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.Affine(scale=(.5, 2.)),
        iaa.Affine(scale={"x": (.5, 2.), "y": (.5, 2.)}),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(shear=(-16, 16)),
        iaa.CropAndPad(percent=(-0.25, 0.25)),
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
