import argparse

from experiments.Image_Restoration_deraing_raindrop_noise1.datasets.base import Dataset
from experiments.Image_Restoration_deraing_raindrop_noise1.datasets.generation import get_dataset



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False, 
            generation=False):
    if generation:
        dataset_import = "generation"
        dataset = "generation_CT_datasets"
        args = {"exp": "/generation_CT_datasets"}
    else:
        dataset_import = "base"

    if dataset_import == "base":
        return Dataset(folder,
                       image_size,
                       exts=exts,
                       augment_flip=augment_flip,
                       convert_image_to=convert_image_to,
                       condition=condition,
                       equalizeHist=equalizeHist,
                       crop_patch=crop_patch,
                       sample=sample)
    elif dataset_import == "generation":
        if dataset == "generation_CT_datasets":
            config = {
                "data": {
                    "dataset": "generation_CT_datasets",
                    "image_size": 64,  # 64
                    "channels": 1,
                    "logit_transform": False,
                    "uniform_dequantization": False,
                    "gaussian_dequantization": False,
                    "random_flip": True,
                    "rescaled": True,
                }}
        args = dict2namespace(args)
        config = dict2namespace(config)
        return get_dataset(args, config)[0]
