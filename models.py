import timm
from timm.data import resolve_data_config, resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import transforms
from pprintpp import pprint
import os
import json


def model_list(model_type):
    vit_models = [
        # ViT (supervised)
        'vit_small_patch16_224',
        'vit_small_patch32_224',
        'vit_base_patch16_224',
        'vit_base_patch32_224',
        'vit_large_patch16_224',
        'vit_large_patch32_384',
        # 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        # 'vit_small_patch32_224.augreg_in21k_ft_in1k',
        # 'vit_base_patch16_224.orig_in21k_ft_in1k',
        # 'vit_base_patch32_224.augreg_in1k',
        # 'vit_large_patch16_224.augreg_in21k_ft_in1k',
        # 'vit_large_patch32_384.orig_in21k_ft_in1k',

        # DINO-ViT (self-supervised) - num classes = 0
        # 'vit_small_patch14_reg4_dinov2.lvd142m',
        # 'vit_base_patch14_reg4_dinov2.lvd142m',
        # 'vit_large_patch14_reg4_dinov2.lvd142m',

        # BEiT (self-supervised with supervised fine-tuning)
        'beit_base_patch16_224',
        'beit_large_patch16_224',
        'beitv2_base_patch16_224',
        'beitv2_large_patch16_224',
        'beit3_base_patch16_224',
        'beit3_large_patch16_224'
        # 'beit_base_patch16_224.in22k_ft_in22k_in1k',
        # 'beit_base_patch16_384.in22k_ft_in22k_in1k',
        # 'beit_large_patch16_224.in22k_ft_in22k_in1k',
        # 'beit_large_patch16_384.in22k_ft_in22k_in1k',
        # 'beitv2_base_patch16_224.in1k_ft_in22k_in1k',
        # 'beitv2_large_patch16_224.in1k_ft_in1k',
        # 'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        # 'beit3_base_patch16_224.indomain_in22k_ft_in1k',
        # 'beit3_large_patch16_224.indomain_in22k_ft_in1k',

        # deit

    ]
    cnn_models = [
        # ResNet (supervised)
        'resnet34',
        'resnet50',
        'resnet101',
        # 'resnet34.ra4_e3600_r224_in1k',
        # 'resnet50.ram_in1k',
        # 'resnet101d.ra2_in1k',

        # Inception (supervised)
        'inception_v3',
        'inception_v4',
        # 'inception_v3.tf_adv_in1k', # Top-1 accuracy: ~79.0%
        # 'inception_v4.tf_in1k', # Top-1 accuracy: ~80.2%
        # 'inception_resnet_v2.tf_in1k', # Top-1 accuracy: ~80.4%
        # 'inception_resnet_v2.tf_ens_adv_in1k' # Top-1 accuracy: ~80.6%
        # 'inception_next_small.sail_in1k', # Top-1 accuracy: ~81.5%
        # 'inception_next_base.sail_in1k' # Top-1 accuracy: ~83.8%
        # 'inception_next_base.sail_in1k_384', # Top-1 accuracy: ~85.5%

        # EfficientNet (supervised)
        'efficientnetv2_rw_s',
        'efficientnetv2_rw_m',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4'
        # 'efficientnetv2_rw_s.ra2_in1k',
        # 'efficientnet_b4.ra2_in1k',
        # 'efficientnet_b3.ra2_in1k',
        # 'efficientnet_b2.ra_in1k'
    ]
    hybrid_models = [
        'ssl_resnet50',
        'convnext_small',
        'convnext_base',
        'convnext_large',
        'convnext_xlarge',
        'convnextv2_base',
        'convnextv2_large'
        # 'resnet50.fb_ssl_yfcc100m_ft_in1k'
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
    model_list = model_list('vit')
