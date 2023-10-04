import sys
import os
sys.path.append(os.path.dirname(__file__))

from datasets.image_dataset import ImageDataset, WFDD
from datasets.hdf5_dataset import WMCA_H5, ThreeDMAD, CSMAD
import pandas as pd
import torch
from functools import partial
# ImageDataset does not support WFDD yet
ImageDataset_List = [
    'CASIA-FASD', 'REPLAY-ATTACK', 'ROSE-YOUTU', 'OULU-NPU', 'MSU-MFSD', 'SIW', 'CASIA-SURF-3DMASK' 
    'CASIA-SURF', 'CASIA-HIFI-MASK', 'CeFA', 'CelabA-Spoof'
]

HDF5_dataset = ['ThreeDMAD', 'WMCA', 'CSMAD']

def parse_data_list_csv(data_list_path):
    """
    :param data_list_path:
    :return:
        image_list: the image's path
        label_list: the label of the image: 0-genuine, 1-photo, 2-replay, 3-mask
    """
    csv = pd.read_csv(data_list_path, header=None)
    image_list = csv.get(0)
    label_list = csv.get(1)

    return image_list, label_list

def label_transform(label, if_binary=True):
    """
    This label transform function transforms labels ('0':genuine, '1':photo, '2':replay, '3':mask) parsed from csv files
    to binary labels (0-genuine/real, 1-spoofing/fake).
    You can define your own label transform function
    :param label: '1'
    :return:
    """
    if if_binary:
        new_label = int(bool(int(label)))
    else:
        # 0 for genuine, 1 for 2d attack, 3 for mask attack
        label = int(label)
        if label == 0:
            return 0
        if label in [1,2]:
            return 1
        if label in [3]:
            return 2
    return new_label


def get_image_dataset_from_list(csv_path, torchvision_transform=None, if_binary_label=True, filter=''):
    """
    :param file_path_list:
    :param torchvision_transform:
    :return:
    """

    image_path_list, label_list = parse_data_list_csv(csv_path)
    label_transform_func = partial(label_transform, if_binary=if_binary_label)
    transformed_label_list = list(map(label_transform_func,label_list))


    image_dataset = ImageDataset(
        file_list=image_path_list,
        label_list=transformed_label_list,
        torchvision_transform=torchvision_transform,
        use_original_frame=False,
        bbox_suffix='_bbox_mtccnn.txt',

    )

    return image_dataset


if __name__ == '__main__':

    data_list_csv_path = 'examples/example.csv'
    image_dataset = get_image_dataset_from_list(data_list_csv_path)
    x = image_dataset.__getitem__(0)
    import IPython;  IPython.embed()
