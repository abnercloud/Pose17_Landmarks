# -----------------------------------------------------
# Copyright (c) Datatang.com. All rights reserved.
# Written by wduo(wangduo@datatang.com)
# -----------------------------------------------------
import os
import glob
import json
import h5py
import numpy as np
from collections import defaultdict


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def imgname_parse(one_img_name, subdir):
    # one_img_name_id = one_img_name.split('-')[0]
    # one_img_name_ext = one_img_name.split('.')[-1]
    # one_img_name = one_img_name_id + '.' + one_img_name_ext
    # one_img_name = subdir + one_img_name

    name_char_list = [ord(char) for char in one_img_name]

    fixed_len = 37
    if len(name_char_list) < fixed_len:
        for _ in range(len(name_char_list) + 1, fixed_len + 1):
            name_char_list.append(ord(' '))

    return name_char_list


def keypoints_parser(keypoints_json):
    points_num = 18
    points = keypoints_json["points"]

    points_for_bndbox = [point["coordinate"] for point in points]
    coordinate = np.array(points_for_bndbox)
    upleft_x = min(coordinate[:, 0])
    upleft_y = min(coordinate[:, 1])
    bottomright_x = max(coordinate[:, 0])
    bottomright_y = max(coordinate[:, 1])

    exist_points = [int(point["id"].split('-')[0]) for point in points]
    points_list = []
    for ii in range(1, points_num + 1):
        if ii in exist_points:
            idx = exist_points.index(ii)
            points_list.append(points[idx]["coordinate"])
        else:
            points_list.append([0, 0])

    return points_list, [[upleft_x, upleft_y, bottomright_x, bottomright_y]]


def segmentation_parser(segmentation_json):
    points = segmentation_json["points"]
    label_list = [point["label"] for point in points]
    if 'Body' in label_list:
        idx = label_list.index('Body')
        points = points[idx]
        coordinate = np.array(points["coordinate"][0])
        upleft_x = min(coordinate[:, 0])
        upleft_y = min(coordinate[:, 1])
        bottomright_x = max(coordinate[:, 0])
        bottomright_y = max(coordinate[:, 1])

        return [[upleft_x, upleft_y, bottomright_x, bottomright_y]]
    else:
        return 0


def write_h5(annot_coco_h5_dict):
    f = h5py.File('annot_coco.h5', 'w')
    f['imgname'] = annot_coco_h5_dict['imgname']
    f['bndbox'] = annot_coco_h5_dict['bndbox']
    f['part'] = annot_coco_h5_dict['part']
    f.close()


def show_h5():
    annot = h5py.File('annot_coco.h5')
    for k in annot.keys():
        print(k)

    bndboxes = annot['bndbox'][:]
    print(bndboxes.shape)
    imgnames = annot['imgname'][:]
    print(imgnames.shape)
    parts = annot['part'][:]
    print(parts.shape)


def read_annot(data_dir):
    """

    :param data_dir:
    :param subdirs:
    :return:
    """
    annot_coco_h5_dict = defaultdict(list)
    pwd = os.getcwd()

    annot_dir = data_dir
    original_img_path = glob.glob(os.path.join(annot_dir, '*.jpg'))
    os.chdir(annot_dir)
    for one_img_path in original_img_path:
        one_img_name = os.path.basename(one_img_path)

        # Read keypoints and segmentation json file.
        one_img_name_no_ext = os.path.splitext(one_img_name)[0]
        keypoints_json_file = one_img_name_no_ext + '-landmark.json'
        # segmentation_json_file = one_img_name_no_ext + '-segmentation.json'
        keypoints_json = load_json(keypoints_json_file)
        # segmentation_json = load_json(segmentation_json_file)

        # Add imgname field.
        name_char_list = imgname_parse(one_img_name, subdir='')
        annot_coco_h5_dict['imgname'].append(name_char_list)

        # Add part field.
        keypoints, alternative_segmentation = keypoints_parser(keypoints_json)
        annot_coco_h5_dict['part'].append(keypoints)

        # Add bndbox field.
        # segmentation = segmentation_parser(segmentation_json)
        # if segmentation:
        #     annot_coco_h5_dict['bndbox'].append(segmentation)
        # else:
        annot_coco_h5_dict['bndbox'].append(alternative_segmentation)

    for k in annot_coco_h5_dict.keys():
        annot_coco_h5_dict[k] = np.array(annot_coco_h5_dict[k])
    print('All done.')

    os.chdir(pwd)
    # Write annot_coco_h5_dict to h5 file.
    write_h5(annot_coco_h5_dict)
    # Show annot_coco.h5
    show_h5()

    pass


if __name__ == '__main__':
    data_dir = r'D:\Users\wduo\Desktop\Facial_106_Landmarks\data_pose17'

    read_annot(data_dir)
