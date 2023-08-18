

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from src.utils.posemap import get_coco_body25_mapping
from src.utils.posemap import kpoint_to_heatmap

import requests

from annotator.openpose import openpose_poseonly


def get_parse_image(image_path: str, output_path:str):
    # 向 127.0.0.1:5000/get_parse 发送 post 请求， json 格式为 {"image_path": image_path, "output_path": output_path}
    # 返回值为{"status": "success"}, 200
    # 保存的图片为output_path
    response = requests.post("http://127.0.0.1:6666/get_parse", json={"image_path": image_path, "output_path": output_path})
    if response.status_code != 200:
        print(response['error'])
        raise Exception("[ImagePreprocessor.py] get_parse_image() failed. Please check the server.")

def get_openpose_json(image: str):
    return openpose_poseonly(image)
    

def vitonhd_preprocesser(
    image_path: str,
    cloth_path: str,
    output_path: str,
    size: Tuple[int, int] = (512, 384),
    radius=5,
):
    height, width = size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform2D = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # "cloth"
    cloth = Image.open(os.path.join(cloth_path, "cloth.jpg"))
    cloth = cloth.resize((width, height))
    cloth = transform(cloth)  # [-1,1]

    # "image"
    image = Image.open(os.path.join(image_path, "person.jpg"))
    image = image.resize((width, height))
    image = transform(image)  # [-1,1]
    
    try:
        if os.path.exists(os.path.join(output_path, "parse/person_vis.png")):
            print("parse image already exists")
        else:
            get_parse_image(image_path, output_path)
    except Exception as e:
        print(e)
        raise Exception("[ImagePreprocessor.py] get_parse_image() failed. Please check the server.")
    else:
        im_parse = Image.open(os.path.join(output_path, "parse/person_vis.png"))
    try:
        if os.path.exists(os.path.join(output_path, "openpose_json/keypoint.json")):
            print("openpose_json already exists")
            openpose_json = json.load(open(os.path.join(output_path, "openpose_json/keypoint.json")))
        else:
            openpose_json = get_openpose_json(os.path.join(image_path, "person.jpg")) 
            # save json
            if not os.path.exists(os.path.join(output_path, "openpose_json")):
                os.makedirs(os.path.join(output_path, "openpose_json"))
            with open(os.path.join(output_path, "openpose_json/keypoint.json"), 'w') as f:
                json.dump(openpose_json, f)
            print("openpose_json saved")    
    except Exception as e:
        print(e)
        raise Exception("[ImagePreprocessor.py] get_openpose_json() failed. Please check the code.")
   
    
    # "im_mask" "pose_map"
    # Label Map
    
    im_parse = im_parse.resize((width, height), Image.NEAREST).convert('P')
    im_parse_final = transforms.ToTensor()(im_parse) * 255
    parse_array = np.array(im_parse)

    parse_shape = (parse_array > 0).astype(np.float32)

    parse_head = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 2).astype(np.float32) + \
                    (parse_array == 4).astype(np.float32) + \
                    (parse_array == 13).astype(np.float32)

    parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 18).astype(np.float32) + \
                        (parse_array == 19).astype(np.float32)

    parser_mask_changeable = (parse_array == 0).astype(np.float32)

    arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

    parse_cloth = (parse_array == 5).astype(np.float32) + \
                    (parse_array == 6).astype(np.float32) + \
                    (parse_array == 7).astype(np.float32)
    parse_mask = (parse_array == 5).astype(np.float32) + \
                    (parse_array == 6).astype(np.float32) + \
                    (parse_array == 7).astype(np.float32)

    parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32)  # the lower body is fixed

    parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    parse_head = torch.from_numpy(parse_head)  # [0,1]
    parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
    parse_mask = torch.from_numpy(parse_mask)  # [0,1]
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    # dilation
    parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
    parse_mask = parse_mask.cpu().numpy()

    # Shape
    parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
    parse_shape = parse_shape.resize((width // 16, height // 16), Image.BILINEAR)
    parse_shape = parse_shape.resize((width, height), Image.BILINEAR)
    shape = transform2D(parse_shape)  # [-1,1]

    # Load pose points
    pose_label = openpose_json
    pose_data = pose_label['people'][0]['pose_keypoints_2d']
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 3))[:, :2]

    # rescale keypoints on the base of height and width
    pose_data[:, 0] = pose_data[:, 0] * (width / 768)
    pose_data[:, 1] = pose_data[:, 1] * (height / 1024)

    pose_mapping = get_coco_body25_mapping()

    # point_num = pose_data.shape[0]
    point_num = len(pose_mapping)

    pose_map = torch.zeros(point_num, height, width)
    r = radius * (height / 512.0)
    im_pose = Image.new('L', (width, height))
    pose_draw = ImageDraw.Draw(im_pose)
    neck = Image.new('L', (width, height))
    neck_draw = ImageDraw.Draw(neck)
    for i in range(point_num):
        one_map = Image.new('L', (width, height))
        draw = ImageDraw.Draw(one_map)

        point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
        point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)
        if point_x > 1 and point_y > 1:
            draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
            pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
            if i == 2 or i == 5:
                neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                    'white')
        one_map = transform2D(one_map)
        pose_map[i] = one_map[0]
    d = []

    for idx in range(point_num):
        ux = pose_data[pose_mapping[idx], 0]  # / (192)
        uy = (pose_data[pose_mapping[idx], 1])  # / (256)

        # scale posemap points
        px = ux  # * width
        py = uy  # * height

        d.append(kpoint_to_heatmap(np.array([px, py]), (height, width), 9))

    pose_map = torch.stack(d)

    # just for visualization
    im_pose = transform2D(im_pose)

    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)


    # do in any case because i have only upperbody
    data = openpose_json
    data = data['people'][0]['pose_keypoints_2d']
    data = np.array(data)
    data = data.reshape((-1, 3))[:, :2]

    # rescale keypoints on the base of height and width
    data[:, 0] = data[:, 0] * (width / 768)
    data[:, 1] = data[:, 1] * (height / 1024)

    shoulder_right = tuple(data[pose_mapping[2]])
    shoulder_left = tuple(data[pose_mapping[5]])
    elbow_right = tuple(data[pose_mapping[3]])
    elbow_left = tuple(data[pose_mapping[6]])
    wrist_right = tuple(data[pose_mapping[4]])
    wrist_left = tuple(data[pose_mapping[7]])

    ARM_LINE_WIDTH = int(90 / 512 * height)
    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
            arms_draw.line(
                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
            arms_draw.line(
                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
    else:
        arms_draw.line(np.concatenate(
            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

    hands = np.logical_and(np.logical_not(im_arms), arms)

    parse_mask += im_arms
    parser_mask_fixed += hands
    # delete neck
    parse_head_2 = torch.clone(parse_head)

    parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                            np.logical_not(
                                                                np.array(parse_head_2, dtype=np.uint16))))

    # tune the amount of dilation here
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    im_mask = image * parse_mask_total
    inpaint_mask = 1 - parse_mask_total
    inpaint_mask = inpaint_mask.unsqueeze(0)
    parse_mask_total = parse_mask_total.numpy()
    parse_mask_total = parse_array * parse_mask_total
    parse_mask_total = torch.from_numpy(parse_mask_total)


    result = {
        'image': image,
        'cloth': cloth,
        'im_name': image_path,
        'pose_map': pose_map,
        'inpaint_mask': inpaint_mask,
        'im_mask': im_mask,
        'category': 'upper_body',
    }
    
    print("[ImagePreprocessor.py] vitonhd_preprocesser() finished.")

    return result
