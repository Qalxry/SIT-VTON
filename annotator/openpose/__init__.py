import cv2
import numpy as np
from . import util
from .body import Body
from .hand import Hand
import json
import os
from torch.cuda import empty_cache
from typing import Union


def openpose_poseonly(oriImg: Union[np.ndarray, str]):
    # load model
    model_type = 'body25'
    body_model_path = os.path.join(os.path.dirname(__file__), './model/pose_iter_584000.caffemodel.pt')
    body_estimation = Body(body_model_path, model_type)
    # load image
    if isinstance(oriImg, str):
        oriImg = cv2.imread(oriImg)        # B,G,R order
    # detect body
    print('detecting body...')
    candidate, subset = body_estimation(oriImg)
    del body_estimation
    empty_cache()
    return util.get_bodypose_json_body25(candidate, subset)


def openpose_prepocesser(
    oriImg: Union[np.ndarray, str],
    return_img = False, 
    return_json = False, 
    save_img = True, 
    save_json = True, 
    save_img_path: str = './',
    save_img_name: str = None, 
    save_img_format: str = 'png',
    save_json_path: str = './', 
    save_json_name: str = None,
):
    # load model
    model_type = 'body25'
    body_model_path = os.path.join(os.path.dirname(__file__), './model/pose_iter_584000.caffemodel.pt')
    hand_model_path = os.path.join(os.path.dirname(__file__), 'model/hand_pose_model.pth')
    body_estimation = Body(body_model_path, model_type)
    hand_estimation = Hand(hand_model_path)
    oriImg_name = None
    
    # load image
    if isinstance(oriImg, str):
        oriImg_name = oriImg
        oriImg = cv2.imread(oriImg)        # B,G,R order
    
    # detect body
    print('detecting body...')
    candidate, subset = body_estimation(oriImg)
    canvas = np.zeros_like(oriImg)
    print('drawing bodypose...')
    canvas, pose_keypoints_2d = util.draw_bodypose(canvas, candidate, subset, model_type, return_keypoints=True)
    
    del body_estimation
    empty_cache()
    
    # detect hand
    print('detecting hand...')
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)
        
    del hand_estimation
    empty_cache()
    
    print('drawing handpose...')
    canvas = util.draw_handpose(canvas, all_hand_peaks)

    # save result
    if save_json or return_json:
        pose_keypoints_2d = np.ravel(pose_keypoints_2d)
        pose_keypoints_2d = [round(x, 6) if x != 0 else 0 for x in pose_keypoints_2d]
        keypoint_json = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": pose_keypoints_2d,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }
            ]
        }
        if save_json:
            if save_json_name is None:
                if oriImg_name is not None:
                    # 获取除去后缀与路径的文件名
                    save_json_name = os.path.basename(oriImg_name).split('.')[0] + '_keypoints.json'
                else:
                    save_json_name = 'keypoints.json'
            else:
                # check if the postfix is json
                if save_json_name.split('.')[-1] != 'json':
                    save_json_name = save_json_name + '.json'
            save_json_name = os.path.join(save_json_path, save_json_name)
            # check if the path exists    
            if not os.path.exists(save_json_path):
                os.makedirs(save_json_path)
            with open(save_json_name, 'w') as f:
                json.dump(keypoint_json, f)  
    
    if save_img:
        if save_img_name is None:
            if oriImg_name is not None:
                save_img_name = os.path.basename(oriImg_name).split('.')[0] + '_rendered.' + save_img_format
            else:
                save_img_name = 'rendered.' + save_img_format
        save_img_name = os.path.join(save_img_path, save_img_name)
        # check if the path exists    
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        cv2.imwrite(save_img_name, canvas)
        
    if return_img and return_json:
        return canvas, all_hand_peaks
    elif return_img:
        return canvas
    elif return_json:
        return all_hand_peaks
    else:
        return None
    