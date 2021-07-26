"""
    1. Information regarding various Augmentation Pipelines used.
    2. Add all augmentation piplines used for reference.
"""

import imgaug
from imgaug import augmenters

# ******************************************************************************************************************** #

def get_augments_1(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 5), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 7), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.15), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.75, 1.25), translate_percent=(-0.15, 0.15), seed=augment_seed),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.075), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.075), seed=augment_seed),
            augmenters.color.MultiplyBrightness(mul=(0.5, 1.5)),
            augmenters.color.MultiplySaturation(mul=(0, 5), seed=augment_seed),
            augmenters.iaa_convolutional.Sharpen(alpha=(0.75, 1), lightness=(0.75, 1.25), seed=augment_seed),
            augmenters.iaa_convolutional.Emboss(alpha=(0.75, 1), strength=(0.75, 1.25), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    imgaug.seed(entropy=augment_seed)
    roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.GlassBlur(severity=(3, 5), seed=augment_seed)] * 3)

    return dataset_augment, roi_augment

# ******************************************************************************************************************** #

def get_augments_2(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 5), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 7), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.15), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), seed=augment_seed),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.05), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.05), seed=augment_seed),
            augmenters.color.MultiplyBrightness(mul=(0.5, 1.5)),
            augmenters.color.MultiplySaturation(mul=(0, 5), seed=augment_seed),
            augmenters.iaa_convolutional.Sharpen(alpha=(0.75, 1), lightness=(0.75, 1.25), seed=augment_seed),
            augmenters.iaa_convolutional.Emboss(alpha=(0.75, 1), strength=(0.75, 1.25), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    imgaug.seed(entropy=augment_seed)
    roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.GlassBlur(severity=(3, 5), seed=augment_seed)] * 3)

    return dataset_augment, roi_augment

# ******************************************************************************************************************** #

def get_augments_3(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 5), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 7), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.25), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.5, 1.5), translate_percent=(-0.25, 0.25), seed=augment_seed),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.05), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.05), seed=augment_seed),
            augmenters.color.MultiplyBrightness(mul=(0.5, 1.5)),
            augmenters.color.MultiplySaturation(mul=(0, 5), seed=augment_seed),
            augmenters.iaa_convolutional.Sharpen(alpha=(0.75, 1), lightness=(0.5, 1.5), seed=augment_seed),
            augmenters.iaa_convolutional.Emboss(alpha=(0.75, 1), strength=(0.5, 1.5), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    imgaug.seed(entropy=augment_seed)
    roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.GlassBlur(severity=(3, 5), seed=augment_seed)] * 5)

    return dataset_augment, roi_augment

# ******************************************************************************************************************** #

def get_augments_4(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 5), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 7), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.15), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.75, 1.25), translate_percent=(-0.15, 0.15), seed=augment_seed, cval=(0, 255), mode="symmetric"),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.075), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.075), seed=augment_seed),
            augmenters.color.MultiplyBrightness(mul=(0.5, 1.5)),
            augmenters.color.MultiplySaturation(mul=(0, 5), seed=augment_seed),
            augmenters.iaa_convolutional.Sharpen(alpha=(0.75, 1), lightness=(0.75, 1.25), seed=augment_seed),
            augmenters.iaa_convolutional.Emboss(alpha=(0.75, 1), strength=(0.75, 1.25), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    imgaug.seed(entropy=augment_seed)
    # roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.GlassBlur(severity=(3, 5), seed=augment_seed)] * 3)
    roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.Pixelate(severity=5, seed=augment_seed)])

    return dataset_augment, roi_augment

# ******************************************************************************************************************** #

def get_augments(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 5), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 7), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.15), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.75, 1.25), translate_percent=(-0.15, 0.15), seed=augment_seed, cval=(0, 255), mode="symmetric"),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.075), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.075), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    return dataset_augment

# ******************************************************************************************************************** #

