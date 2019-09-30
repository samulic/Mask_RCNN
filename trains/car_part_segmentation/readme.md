# Train the segmentation model

1. Download and unzip the dataset from this [link](https://drive.google.com/open?id=1T81ZTQhD7SKv2KYfVdzIvl6cyL9qBHQZ) into the dataset folder
2. Download and unzip di car part annotations from this [link](https://drive.google.com/open?id=15pW8EjFxiuItbojS-3OUFINrs-sjpzpl)
3. run the script `train.sh`

## Data Augmentations

The file `car_part.py` contains a list of augmentations used for the dataset, modify it in case.

```
augmentation = iaa.OneOf([
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.Affine(scale=(1., 2.5), rotate=(-90, 90), shear=(-16, 16), 
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    iaa.LinearContrast((0.5, 1.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0)),
    iaa.LogContrast(gain=(0.6, 1.4)),
    iaa.PerspectiveTransform(scale=(0.01, 0.15)),
    iaa.Clouds(),
    iaa.Noop(),
])
```