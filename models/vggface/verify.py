# -*- coding: utf-8 -*-

# face verification with the VGGFace2 model

import sys
import concurrent.futures
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from .keras_vggface.vggface import VGGFace
from .keras_vggface.utils import preprocess_input
import face_recognition
from facelib.utils import extract_face_b64, adjust_face_angle
from config.settings import ALGORITHM, VGGFACE_WEIGHTS


import tensorflow as tf

INPUT_SIZE = (224, 224)

graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph)
with graph.as_default():
    with session.as_default():
        # 装入识别模型 # pooling: None, avg or max # model: vgg16, senet50, resnet50
        model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg', weights=VGGFACE_WEIGHTS) 
        # https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        model._make_predict_function() # have to initialize before threading


# 从照片中获取人脸数据，返回所有能识别的人脸
def extract_face(filename, angle, required_size=(224, 224)):
    # load image from file
    pixels = face_recognition.load_image_file(filename)
    # extract the bounding box from the first face
    face_bounding_boxes = face_recognition.face_locations(pixels)

    # 可能返回 >0, 多个人脸
    if len(face_bounding_boxes) == 0:
        return [], []

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

        face_array = np.asarray(image, 'float32')
        face_list.append(face_array)

        # show face
        #image.show()

    return face_list, face_bounding_boxes


def load_face(filename, required_size=(224, 224)):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# 返回图片中所有人脸的特征
def get_features(filename, angle=None): # angle=None 识别时不修正角度， angle=0 识别时修正角度
    # extract faces
    faces, face_boxs = extract_face(filename, angle)
    if len(faces) == 0:
        return [], [], []
    yhat2 = get_features_array(faces)
    return yhat2, face_boxs, faces


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# 返回图片中所有人脸的特征
def get_features_b64(base64_data, angle=None):
    # extract faces
    faces, face_boxs = extract_face_b64(base64_data, angle=angle, required_size=INPUT_SIZE)
    faces = np.float32(faces)
    if len(faces) == 0:
        return [], [], []
    yhat2 = get_features_array(faces)
    return yhat2, face_boxs, faces


# 根据人脸列表返回特征
def get_features_array(faces):
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # perform prediction
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            yhat = model.predict(samples)
    yhat2 = yhat / np.linalg.norm(yhat)
    return yhat2


# 特征值距离
def face_distance(face_encodings, face_to_compare):
    return face_recognition.face_distance(np.array(face_encodings), np.array(face_to_compare))



# 比较两个人脸是否同一人, 多线程处理
def is_match_b64(b64_data1, b64_data2):
    from models.parallel.verify import get_features_b64_thread

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_b64_thread, 'vgg', b64_data1), # 0
            executor.submit(get_features_b64_thread, 'vgg', b64_data2), # 2
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    if len(results[0][1])==0 or len(results[1][1])==0:
        return False, [999]

    distance_vgg = face_distance([results[0][0][0]], results[1][0][0])
    if distance_vgg <= ALGORITHM['vgg']['distance_threshold']:
        return True, distance_vgg/ALGORITHM['vgg']['distance_threshold']

    # 均为匹配
    return False, distance_vgg/ALGORITHM['vgg']['distance_threshold'] # 只返回 vgg 结果


# 比较两个人脸是否同一人, encoding_list1来自已知db用户, 多对1, db里可能有多个脸，base64只取一个脸， 多线程处理
def is_match_b64_2(encoding_list_db, b64_data):
    from models.parallel.verify import get_features_b64_thread

    encoding_list1 = [[], []]  
    for i in range(len(encoding_list_db)):
        encoding_list1[0].extend(encoding_list_db[i]['vgg'].values()) # vgg 

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_b64_thread, 'vgg', b64_data), # 0
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    if len(results[0][1])==0:
        return False, [999], []

    distance_vgg = face_distance(encoding_list1[0], results[0][0][0])
    x = distance_vgg <= ALGORITHM['vgg']['distance_threshold']
    if x.any():
        return True, distance_vgg/ALGORITHM['vgg']['distance_threshold'], results[0][2]

    # 均未匹配
    return False, distance_vgg/ALGORITHM['vgg']['distance_threshold'], results[0][2] # 只返回 vgg 结果



# 与给定的若干人脸中识别, encoding_list_db来自已知db用户, 多对1, db里可能有多个脸，base64只取一个脸， 多线程处理
# 因为数据集比较少，通过比较排序搜索，用于双因素识别
def is_match_b64_3(encoding_list_data, b64_data):
    from models.parallel.verify import get_features_b64_thread

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_b64_thread, 'vgg', b64_data), # 0
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    if len(results[0][1])==0:
        return []

    # 搜索
    face_X, face_y = encoding_list_data
    # 计算距离
    distance_list = face_distance(face_X, results[0][0][0])
    # 合并结果，去掉大于阈值的结果
    labels = {}
    result_list = []
    for i,j in zip(face_y, distance_list):
        if j > ALGORITHM['vgg']['distance_threshold']:
            continue
        if i in labels.keys():
            labels[i] += 1
        else:
            labels[i] = 1
            result_list.append([i,j])

    #result_list = [ (i,j) for i,j in zip(face_y, distance_list) if j<= ALGORITHM['vgg']['distance_threshold'] ]
    # 按距离排序
    result_list = sorted(result_list, key=lambda s: s[1])
    result_list = [ i+[labels[i[0]]] for i in result_list ]
    # 返回格式 [(user_id1, distance1), (user_id2, distance2), ... ], face_boxes, face_image_array
    return result_list, results[0][1], results[0][2]

