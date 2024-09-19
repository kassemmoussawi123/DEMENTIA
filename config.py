#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/26
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python 3.9; PyCharm;Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0
# @License  : None
# @Brief    : 配置文件

import os
import datetime
import matplotlib
import platform
import pandas as pd
import random
import numpy as np
import py3nvml
import torch
from torch.backends import cudnn
import tensorflow as tf
import fwr13y.d9m.tensorflow as tf_determinism
import warnings
from absl import logging


logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置tensorflow输出控制台信息：1等级，屏蔽INFO，只显示WARNING + ERROR + FATAL
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
free_gpus = [i for i, value in enumerate(py3nvml.get_free_gpus()) if value]
py3nvml.grab_gpus(num_gpus=len(free_gpus), gpu_select=free_gpus, gpu_fraction=0.5, env_set_ok=True)  # 自动获取全部的可用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第0/1/2个GPU,-1不使用GPU，使用CPU
os.environ["OUTDATED_IGNORE"] = "1"  # 忽略OutdatedPackageWarning
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpu_l = tf.config.list_physical_devices('GPU')
# for i_gpu in gpu_l:
#     if int(i_gpu.name.split(':')[-1]) in free_gpus:
#         tf.config.set_visible_devices(i_gpu, 'GPU')
#         tf.config.experimental.set_memory_growth(i_gpu, True)  # 防止GPU显存爆掉,限制显存23G
#         tf.config.set_logical_device_configuration(i_gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=23552)])
if gpu_l and len(gpu_l) > 1:
    dev = [f"/gpu:{i}" for i in free_gpus]
    distribute_strategy = tf.distribute.MirroredStrategy(devices=dev,  # devices=["/gpu:0", "/gpu:1", "/gpu:2"] 多GPU训练
                                                         cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
else:
    distribute_strategy = tf.distribute.get_strategy()  # 单一实例的默认策略
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)
# np.set_printoptions(threshold=np.inf)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # 保存矢量图中的文本在AI中可编辑
warnings.filterwarnings("ignore", message='Custom mask layers require a config and must override get_config.*')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


if platform.system() == 'Windows':
    font_family = 'Arial'
    DATA_PATH = r'F:\Graduate\Datasets\audio\DementiaBank\Media\dementia\English\ADReSS-IS2020-data'
    DATA_PATH_PITT = r'F:\Graduate\Datasets\audio\DementiaBank/'
    BERT_MODEL_PATH = r'D:\pretrained_models\huggingface_models\transformers'
    NLTK_DATA_PATH = r'D:\pretrained_models\NLTK'
    StanfordCoreNLP_PATH = r'D:\pretrained_models\CoreNLP\models'
    MODEL_PATH_HanLP = r'D:\pretrained_models\hanlp'
else:
    font_family = 'DejaVu Sans'
    DATA_PATH = r'/home/medicaldata/ZZLData/Datasets/audio/DementiaBank/DementiaBank/Media/dementia/English/ADReSS-IS2020-data'
    DATA_PATH_PITT = r'/home/medicaldata/ZZLData/Datasets/audio/DementiaBank/DementiaBank/'
    BERT_MODEL_PATH = r'/home/zlzhang/pretrained_models/huggingface_models/transformers'
    NLTK_DATA_PATH = r'/home/zlzhang/pretrained_models/NLTK'
    StanfordCoreNLP_PATH = r'/home/zlzhang/pretrained_models/CoreNLP/models'
    MODEL_PATH_HanLP = r'/home/zlzhang/pretrained_models/hanlp'
matplotlib.rcParams["font.family"] = font_family
os.environ['HANLP_HOME'] = MODEL_PATH_HanLP


def setup_seed(seed: int):
    """
    全局固定随机种子
    :param seed: 随机种子值
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)  # tensorflow 2.8+
    tf_determinism.enable_determinism()
    tf.config.experimental.enable_op_determinism()  # tensorflow 2.8+
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False


rs = 323
setup_seed(rs)
