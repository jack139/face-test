# -*- coding: utf-8 -*-`
import time, os, base64
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from config.settings import ALGORITHM, TRAINED_MODEL_PATH
from importlib import import_module
import face_recognition
#from datetime import datetime


# 时间函数
ISOTIMEFORMAT=['%Y-%m-%d %X', '%Y-%m-%d', '%Y%m%d%H%M', '%Y-%m-%d %H:%M']
def time_str(t=None, format=0):
    return time.strftime(ISOTIMEFORMAT[format], time.localtime(t))


# 动态引入
def import_verify(face_algorithm):
    module = import_module(ALGORITHM[face_algorithm]['module'])
    #print('imported: ', module)
    return module


# 将 base64 编码的图片转为Image 数组
def load_image_b64(b64_data, mode='RGB'):
    data = base64.b64decode(b64_data) # Bytes
    tmp_buff = BytesIO()
    tmp_buff.write(data)
    tmp_buff.seek(0)
    img = Image.open(tmp_buff)
    if mode:
        img = img.convert(mode)
    if img.size>(500,500): # 处理的尺寸不超过500
        img.thumbnail((500, 500)) 
    pixels =  np.array(img)
    tmp_buff.close()
    #print('pixels size: ', pixels.nbytes)
    return pixels

# 人脸定位
def face_locations_b64(b64_data):
    # load image from file
    pixels = load_image_b64(b64_data)
    # extract the bounding box from the first face 
    face_bounding_boxes = face_recognition.face_locations(pixels)

    # 可能返回 >0, 多个人脸
    if len(face_bounding_boxes) == 0:
        return []

    # 按人脸面积从大到小排序
    face_bounding_boxes = sorted( face_bounding_boxes, key=lambda s: (s[1]-s[3])*(s[2]-s[0]), reverse=True )
    return face_bounding_boxes


# 从照片中获取人脸数据，返回所有能识别的人脸， 图片输入为 base64 编码
def extract_face_b64(b64_data, angle=None, required_size=(224, 224)):
    #start_time = datetime.now()
    # load image from file
    pixels = load_image_b64(b64_data)
    #print('[1 Time taken: {!s}]'.format(datetime.now() - start_time))
    # extract the bounding box from the first face
    face_bounding_boxes = face_recognition.face_locations(pixels)
    #print('[2 Time taken: {!s}]'.format(datetime.now() - start_time))

    # 可能返回 >0, 多个人脸
    if len(face_bounding_boxes) == 0:
        return [], []

    # 按人脸面积从大到小排序
    face_bounding_boxes = sorted( face_bounding_boxes, key=lambda s: (s[1]-s[3])*(s[2]-s[0]), reverse=True )
    # 只处理面积最大的一个脸
    face_bounding_boxes = face_bounding_boxes[:1]
    
    face_list = []
    for face_box in face_bounding_boxes: 
        top, right, bottom, left = face_box
        x1, y1, width, height = left, top, right-left, bottom-top
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)

        # 调整人脸角度
        image = adjust_face_angle(face, image, angle)

        # 调整尺寸
        image = image.resize(required_size)

        face_array = np.array(image, 'uint8')
        face_list.append(face_array)
        #print('[3 Time taken: {!s}]'.format(datetime.now() - start_time))

    return face_list, face_bounding_boxes


# 训练特征值， 按用户组
def train_by_group(group_id):
    from models import knn_db

    face_algorithm = ['vgg', 'evo']

    start_time = datetime.now()
    for algorithm in face_algorithm:
        # Train the KNN classifier and save it to disk
        classifier = knn_db.train(group_id, 
            model_save_path=os.path.join(TRAINED_MODEL_PATH, group_id + ALGORITHM[algorithm]['ext']), 
            n_neighbors=10,
            face_algorithm=algorithm,
            need_train=1) # need_train 置 1 的进行训练
        print('[Time taken: {!s} ({}, {})]'.format(datetime.now() - start_time, algorithm, group_id))


# 图片文件转base64编码
def load_image_to_base64(image_file):
    with open(image_file, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data)

# 根据两点坐标，旋转图片使两点水平
def _HorizontalEyes(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    k = (y2-y1) / (x2-x1)
    angle = np.arctan(k)/np.pi*180
    #angle = angle if abs(angle)>3 else 0 # 大于3度才修正
    return angle

# angle: None 不变， 0 只水平修正， 360 水平修正后左右镜面， 非0 水平修正后旋转
def adjust_face_angle(face, image, angle): # face为array, image为对应Image图像
    if angle is None:
        return image
    # 寻找特征点
    face_landmarks_list = face_recognition.face_landmarks(face)
    if len(face_landmarks_list)>0:
        # 先修正角度, 
        angle_0 = _HorizontalEyes([face_landmarks_list[0]['left_eye'][0]] + [face_landmarks_list[0]['right_eye'][0]])
    else:
        angle_0 = 0
    #print('angle: ', angle_0)
    # 旋转、镜像
    if angle==360:
        # 修正水平后，左右镜像
        image = image.rotate(angle_0) if angle_0!=0 else image 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        # 旋转
        image = image.rotate(angle_0+angle) if (angle_0+angle)!=0 else image
    #image.show()
    return image
