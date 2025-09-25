import timm
from timm.data import resolve_data_config, resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import transforms
from pprintpp import pprint
import os
import json


def get_model_list(model_type):
    vit_models = [
        # ViT (supervised)
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
        'vit_base_patch32_224',
        'vit_large_patch16_224',
        'vit_large_patch32_384',
        # BEiT (self-supervised with supervised fine-tuning)
        'beit_base_patch16_224',
        'beit_large_patch16_224',
        'beitv2_base_patch16_224',
        'beitv2_large_patch16_224',
        'beit3_base_patch16_224',
        'beit3_large_patch16_224',
        # deit
        'deit_small_patch16_224',
        'deit_base_patch16_224',
        'deit3_small_patch16_224',
        'deit3_medium_patch16_224',
        'deit3_base_patch16_224',
        'deit3_large_patch16_224'

    ]
    cnn_models = [
        # ResNet (supervised)
        'resnet34',
        'resnet50',
        'resnet101',
        'ssl_resnet50',
        # Inception (supervised)
        'inception_v3',
        'inception_v4',
        # EfficientNet (supervised)
        'efficientnetv2_rw_s',
        'efficientnetv2_rw_m',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4'
    ]
    hybrid_models = [
        'convnext_small',
        'convnext_base',
        'convnext_large',
        'convnext_xlarge',
        'convnextv2_base',
        'convnextv2_large'
    ]
    model_type = model_type.lower()
    if model_type == 'cnn':
        return cnn_models
    elif model_type == 'vit':
        return vit_models
    elif model_type == 'hybrid':
        return hybrid_models
    elif model_type == '':
        return vit_models + cnn_models + hybrid_models
    else:
        return None


def create_model(name):
    print('Loading model {}...'.format(name))
    model = timm.create_model(name, pretrained=True)
    config = resolve_model_data_config(model)
    transform = create_transform(**config)
    return {'model': model, 'transform': transform}

if __name__ == '__main__':
    model_list = get_model_list('hybrid')


