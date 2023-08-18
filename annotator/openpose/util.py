import numpy as np
import math
import cv2
import numpy as np


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        if len(weights_name.split(".")) > 4:  # body25
            transfered_model_weights[weights_name] = model_weights[
                ".".join(weights_name.split(".")[3:])
            ]
        else:
            transfered_model_weights[weights_name] = model_weights[
                ".".join(weights_name.split(".")[1:])
            ]
    return transfered_model_weights


def get_bodypose_json_body25(candidate, subset):
    njoint = 25
    pose_keypoints_2d = []
    
    for i in range(njoint):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                pose_keypoints_2d.append([0, 0, 0])
                continue
            pose_keypoints_2d.append(candidate[index][0:3])
            
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
    
    return keypoint_json

    
# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset, model_type="coco", radius=7, return_keypoints=False):
    if model_type == "body25":
        limbSeq = [[1, 0], [1, 2], [2, 3], [3, 4], [1, 5], 
                   [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
                   [10, 11], [8, 12], [12, 13], [13, 14], [0, 15], 
                   [0, 16], [15, 17], [16, 18], [11, 24], [11, 22], 
                   [14, 21], [14, 19], [22, 23], [19, 20]]
        njoint = 25
    else:
        limbSeq = [[1, 2],[1, 5],[2, 3],[3, 4],[5, 6],
                   [6, 7],[1, 8],[8, 9],[9, 10],[1, 11],
                   [11, 12],[12, 13],[1, 0],[0, 14],[14, 16],
                   [0, 15],[15, 17],[2, 16],[5, 17]]
        njoint = 18

    colors = [[85, 0, 255], [0, 0, 255], [0, 85, 255], [0, 170, 255], [0, 255, 255], 
              [0, 255, 170], [0, 255, 85], [0, 255, 0], [0, 0, 255], [85, 255, 0], 
              [170, 255, 0], [255, 255, 0], [255, 170, 0], [255, 85, 0], [255, 0, 0], 
              [170, 0, 255], [255, 0, 170], [255, 0, 255], [255, 0, 85], [255, 0, 0], 
              [255, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 0], [255, 255, 0]]

    pose_keypoints_2d = []
    
    for i in range(njoint - 1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                center=(round(mY), round(mX)),
                axes=(round(length / 2), radius),
                angle=round(angle),
                arcStart=0,
                arcEnd=360,
                delta=1,
            )
            if i == 0:
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            else:
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i+1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    for i in range(njoint):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                pose_keypoints_2d.append([0, 0, 0])
                continue
            cur_canvas = canvas.copy()
            x, y = candidate[index][0:2]
            pose_keypoints_2d.append(candidate[index][0:3])
            cv2.circle(cur_canvas, (int(x), int(y)), radius, colors[i], thickness=-1)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    if return_keypoints:
        return canvas, pose_keypoints_2d
    else:
        return canvas


# image drawed by opencv is not good.
def draw_handpose(canvas, peaks, show_number=True, radius=4):
    edges = [[0, 1],[1, 2],[2, 3],[3, 4],[0, 5],[5, 6],[6, 7],
             [7, 8],[0, 9],[9, 10],[10, 11],[11, 12],[0, 13],[13, 14],
             [14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]

    colors = [[100, 100, 100],
              [0, 0, 100],[0, 0, 150],[0, 0, 200],[0, 0, 255],
              [0, 100, 100],[0, 150, 150],[0, 200, 200],[0, 255, 255],
              [50, 100, 0],[75, 150, 0],[100, 200, 0],[125, 255, 0],
              [100, 50, 0],[150, 75, 0],[200, 100, 0],[255, 125, 0],
              [100, 0, 100],[150, 0, 150],[200, 0, 200],[255, 0, 255]]
    
    if peaks.__len__() == 21:
        peaks = [peaks]

    # draw edges
    cur_canvas = canvas.copy()
    for hand_peaks in peaks:
        for idx, [e1, e2] in enumerate(edges):
            x1, y1 = hand_peaks[e1][0], hand_peaks[e1][1]
            x2, y2 = hand_peaks[e2][0], hand_peaks[e2][1]
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            cur_canvas = canvas.copy()
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly(
                center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                axes=(int(length / 2), radius),
                angle=int(angle),
                arcStart=0,
                arcEnd=360,
                delta=1,
            )
            cv2.fillConvexPoly(cur_canvas, polygon, colors[idx + 1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    # draw points
    for hand_peaks in peaks:
        for idx, peak in enumerate(hand_peaks):
            cur_canvas = canvas.copy()
            x, y = peak[0], peak[1]
            if x == 0 and y == 0:
                continue
            cv2.circle(
                img=cur_canvas,
                center=(int(x), int(y)),
                radius=radius,
                color=colors[idx],
                thickness=-1,  # -1 means filled circle
            )
            if show_number:
                cv2.putText(
                    canvas,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    lineType=cv2.LINE_AA,
                )
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas



# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]
            ]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    """
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    """
    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
