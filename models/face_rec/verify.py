# -*- coding: utf-8 -*-

import sys
import numpy as np
import face_recognition
from facelib.utils import load_image_b64, extract_face_b64, _HorizontalEyes
from PIL import Image


# 返回图片中所有人脸的特征
def get_features(filename, angle=None):
    # load image
    image = face_recognition.load_image_file(filename)
    # 调整人脸角度
    if angle!=None:
        # 先修改角度
        face_landmarks_list = face_recognition.face_landmarks(image)
        if len(face_landmarks_list)>0:
            pil_image = Image.fromarray(image)        
            angle_0 = _HorizontalEyes([face_landmarks_list[0]['left_eye'][0]] + [face_landmarks_list[0]['right_eye'][0]])
            # 旋转
            pil_image = pil_image.rotate(angle+angle_0)
            #pil_image.show()
            image = np.array(pil_image)
            # extract faces
    face_bounding_boxes = face_recognition.face_locations(image)
    if len(face_bounding_boxes)==0:
        return [], []

    features = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, num_jitters=1)

    return features, face_bounding_boxes


# 返回图片中所有人脸的特征
def get_features_b64(base64_data):
    pixels = load_image_b64(base64_data)
    # extract faces
    face_bounding_boxes = face_recognition.face_locations(pixels)
    if len(face_bounding_boxes)==0:
        return [], []

    features = face_recognition.face_encodings(pixels, known_face_locations=face_bounding_boxes, num_jitters=1)

    return features, face_bounding_boxes


# 特征值距离
def face_distance(face_encodings, face_to_compare):
    return face_recognition.face_distance(np.array(face_encodings), np.array(face_to_compare))
