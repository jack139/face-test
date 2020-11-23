# -*- coding: utf-8 -*-

from datetime import datetime
import cv2
from PIL import Image
import numpy as np
import concurrent.futures
import face_recognition
from config.settings import ALGORITHM
from facelib.utils import extract_face_b64

# 双算法： vggface + evoLVe
from models.vggface import verify as verify_vgg
from models.face_evoLVe import verify as verify_evo


def get_features_b64_thread(face_algorithm, b64_data):
    # https://discuss.streamlit.io/t/attributeerror-thread-local-object-has-no-attribute-value/574/3
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    #start_time = datetime.now()
    if face_algorithm=='vgg':
        encoding_list, face_boxes, faces = verify_vgg.get_features_b64(b64_data, angle=ALGORITHM[face_algorithm]['p_angle'])
    else:
        encoding_list, face_boxes, faces = verify_evo.get_features_b64(b64_data, angle=ALGORITHM[face_algorithm]['p_angle'])
    #print('[{} - Time taken: {!s}]'.format(face_algorithm, datetime.now() - start_time))
    return encoding_list, face_boxes, faces


# 比较两个人脸是否同一人 --------- 不会被knn_db调用, 多线程实现 获取特征值
def is_match_b64(b64_data1, b64_data2):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_b64_thread, 'vgg', b64_data1), # 0
            executor.submit(get_features_b64_thread, 'evo', b64_data1), # 1
            executor.submit(get_features_b64_thread, 'vgg', b64_data2), # 2
            executor.submit(get_features_b64_thread, 'evo', b64_data2), # 3
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    if len(results[0][1])==0 or len(results[2][1])==0:
        return False, [999]

    distance_vgg = face_distance([results[0][0][0]], results[2][0][0])
    if distance_vgg <= ALGORITHM['vgg']['distance_threshold']:
        return True, distance_vgg/ALGORITHM['vgg']['distance_threshold']

    distance_evo = face_distance([results[1][0][0]], results[3][0][0])
    if distance_evo <= ALGORITHM['evo']['distance_threshold']:
        return True, distance_evo/ALGORITHM['evo']['distance_threshold']

    # 均为匹配
    return False, distance_vgg/ALGORITHM['vgg']['distance_threshold'] # 只返回 vgg 结果


# 定位人脸，然后人脸的特征值列表，可能不止一个脸, 只取最大的一个脸(第1个脸)
def get_features_b64_old(b64_data):
    encoding_list1, face_boxes1 = verify_vgg.get_features_b64(b64_data, angle=ALGORITHM['vgg']['p_angle'])
    encoding_list2, face_boxes2 = verify_evo.get_features_b64(b64_data, angle=ALGORITHM['evo']['p_angle'])

    if len(face_boxes1) == 0:
        return [], []
    else:
        # 返回4个，第3个位置留给rec，第4个给deep, 保持一直， 2020-06-25
        #return [encoding_list1[0], encoding_list2[0], [], []], face_boxes1
        return {
            'vgg' : { 'None' : encoding_list1[0] },
            'evo' : { 'None' : encoding_list2[0] }
        } , face_boxes1


# 定位人脸，然后人脸的特征值列表，可能不止一个脸, 只取最大的一个脸(第1个脸)
# 串行版，api注册后，后台生成特征值时使用
def get_features_b64(b64_data, angle=None):
    # extract faces
    faces, face_boxs = extract_face_b64(b64_data, angle=angle, required_size=verify_vgg.INPUT_SIZE)
    if len(faces) == 0:
        return [], []

    # vgg 使用人脸
    faces_vgg = np.float32(faces)

    # evo 使用人脸
    faces_evo = []
    for i in faces:
        image = Image.fromarray(i)
        image = image.resize(verify_evo.INPUT_SIZE)
        faces_evo.append(np.array(image, 'uint8'))

    # 获取特征值
    encoding_list1 = verify_vgg.get_features_array(faces_vgg)
    encoding_list2 = verify_evo.get_features_array(faces_evo)

    return { 'vgg' : encoding_list1[0], 'evo' : encoding_list2[0] } , face_boxs, faces


# 特征值距离
def face_distance(face_encodings, face_to_compare):
    return face_recognition.face_distance(np.array(face_encodings), np.array(face_to_compare))



# 比较两个人脸是否同一人, encoding_list1来自已知db用户, 多对1, db里可能有多个脸，base64只取一个脸， 多线程处理
def is_match_b64_2(encoding_list_db, b64_data):
    encoding_list1 = [[], []]  # [ vgg, evo ]
    for i in range(len(encoding_list_db)):
        encoding_list1[0].extend(encoding_list_db[i]['vgg'].values()) # vgg 
        encoding_list1[1].extend(encoding_list_db[i]['evo'].values()) # evo db中特征值，使用新结构  2020-07-09

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_b64_thread, 'vgg', b64_data), # 0
            executor.submit(get_features_b64_thread, 'evo', b64_data), # 1
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    if len(results[0][1])==0:
        return False, [999]

    distance_vgg = face_distance(encoding_list1[0], results[0][0][0])
    x = distance_vgg <= ALGORITHM['vgg']['distance_threshold']
    if x.any():
        return True, distance_vgg/ALGORITHM['vgg']['distance_threshold']

    distance_evo = face_distance(encoding_list1[1], results[1][0][0])
    x = distance_evo <= ALGORITHM['evo']['distance_threshold']
    if x.any():
        return True, distance_evo/ALGORITHM['evo']['distance_threshold']

    # 均未匹配
    return False, distance_vgg/ALGORITHM['vgg']['distance_threshold'] # 只返回 vgg 结果


# 用于并行获取特征值的 thread
def get_features_array_thread(face_algorithm, face_data):
    # https://discuss.streamlit.io/t/attributeerror-thread-local-object-has-no-attribute-value/574/3
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    #start_time = datetime.now()
    if face_algorithm=='vgg':
        encoding_list = verify_vgg.get_features_array(face_data)
    else:
        encoding_list = verify_evo.get_features_array(face_data)
    #print('[{} - Time taken: {!s}]'.format(face_algorithm, datetime.now() - start_time))
    return encoding_list


# 定位人脸，然后人脸的特征值列表，可能不止一个脸, 只取最大的一个脸(第1个脸)
# 并行版，api search时生成特征值时使用
def get_features_b64_parallel(b64_data, request_id=''):
    # extract faces
    faces, face_boxs = extract_face_b64(b64_data, angle=ALGORITHM['evo']['p_angle'], required_size=verify_vgg.INPUT_SIZE)
    if len(faces) == 0:
        return [], []

    # vgg 使用人脸
    faces_vgg = np.float32(faces)

    # evo 使用人脸
    faces_evo = []
    for i in faces:
        image = Image.fromarray(i)
        image = image.resize(verify_evo.INPUT_SIZE)
        faces_evo.append(np.array(image, 'uint8'))

    # 获取特征值
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [
            executor.submit(get_features_array_thread, 'vgg', faces_vgg), # 0
            executor.submit(get_features_array_thread, 'evo', faces_evo), # 1
        ]
        for future in concurrent.futures.as_completed(future_list):
            pos = future_list.index(future)
            results[pos] = future.result()

    # 生成返回结果
    plus_features = {
        'vgg' : { 'None' : results[0][0].tolist() },
        'evo' : { 'None' : results[1][0].tolist() }
    }
    return plus_features, face_boxs, faces_vgg # 图片返回vgg的

