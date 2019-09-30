#/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

data_path=$DIR/dataset
images_path=$data_path/VOCdevkit/VOC2010/JPEGImages
annotations_path=$data_path/car_part_annotations/Annotations_Part

python car_part.py\
 --images_path $images_path\
 --annotations $annotations_path\
 --checkpoint $DIR/logs/ --weights imagenet
