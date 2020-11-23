# -*- coding: utf-8 -*-

import os
import threading
import numpy as np
import math, operator
from datetime import datetime
from sklearn import neighbors
import pickle
from facelib import dbport
from config.settings import ALGORITHM, TRAINING_ANGLE, KERAS_THRESHOLD_PERCENTAGE
from facelib.utils import import_verify
from tqdm import tqdm
from .knn import score_acc_f1

import tensorflow as tf
graph = tf.Graph()
session = tf.Session(graph=graph)

CLF_CACHE = {}

cache_lock = threading.Lock() # 修改 CLF_CACHE 时需要锁住

# 训练
# face_algorithm 只 支持 evo 和 vgg
def train(group_id, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', 
    verbose=False, face_algorithm='rec', face_num=1000, max_user=None, need_train=None):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # 按用户分组从db装入特征数据, 装入所有数据
    start = 0
    max_length = 1000
    total = 0
    while 1:
        user_list = dbport.user_list_by_group(group_id, start=start, length=max_length, need_train=need_train)
        for i in tqdm(range(len(user_list))):
            if max_user and total>=max_user:
                break

            faces = dbport.user_face_list(group_id, user_list[i])
            for f in faces[:face_num]: # 同一个人，只训练指定数量的人脸，默认1000
                r = dbport.face_info(f)
                if r:
                    for angle in TRAINING_ANGLE: # 旋转不同角度训练 multi2
                        X.append(r['encodings'][face_algorithm][str(angle)])
                        y.append(user_list[i])

            total += 1

        if max_user and total>=max_user:
            break

        if len(user_list)<max_length: 
            break
        else: # 未完，继续
            start += max_length

    print('Data loaded:', total, len(X))

    # 没有可训练数据
    if len(X)==0:
        return None

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    start_time = datetime.now()
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    # 计算 threshold
    #opt_tau, opt_acc = score_acc_f1(X, y, show=False)
    #print('{}: Accuracy at threshold {:.4} = {:.4}'.format(face_algorithm, opt_tau, opt_acc))

    return knn_clf


# 识别
def predict(X_base64, group_id, model_path='', distance_threshold=0.6, face_algorithm='vgg', data_type='base64', request_id=''): 
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_base64: image data in base64 coding
    :param model_path: (optional) 已训练模型路径，默认当前路径
    """
    global CLF_CACHE

    # Load a trained KNN model (if one was passed in)
    clf_path = os.path.join(model_path, group_id+ALGORITHM[face_algorithm]['ext'])

    # 检查是否已缓存clf
    mtime = int(os.path.getmtime(clf_path)) # 模型最近修改时间

    with cache_lock:
        if (clf_path in CLF_CACHE.keys()) and (CLF_CACHE[clf_path][1]==mtime): 
            knn_clf = CLF_CACHE[clf_path][0]
            #print('Bingo clf cache!', group_id)
        else:
            with open(clf_path, 'rb') as f:
                knn_clf = pickle.load(f)
            # 放进cache
            CLF_CACHE[clf_path] = (knn_clf, mtime)
            print('Feeding CLF cache: ', CLF_CACHE.keys())

    if data_type=='base64':
        # 动态载入 verify库
        module_verify = import_verify(face_algorithm)

        # Load image file and find face locations
        # Find encodings for faces in the test iamge
        faces_encodings, X_face_locations, faces = module_verify.get_features_b64(X_base64, angle=ALGORITHM[face_algorithm]['p_angle'])

        if len(X_face_locations) == 0:
            return []

        # 保存人脸到临时表, 只保存vgg的
        if request_id!='' and face_algorithm=='vgg':
            dbport.face_save_to_temp(group_id, request_id, image=faces[0])

    else:
        # data_type = 'encodings'
        faces_encodings = X_base64
        X_face_locations = [(0,0,0,0)] # 从db来的数据没有人脸框坐标，只有一个人脸

    #print(faces_encodings)

    # Use the KNN model to find the first 5 best matches for the test face
    # 返回5个最佳结果
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=5)
    #are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    #return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    #print(closest_distances)

    # return multi results
    results = []
    for i in range(len(X_face_locations)):
        # 第一个超过阈值，说明未匹配到
        if closest_distances[0][i][0]>distance_threshold:
            results.append(['unknown', X_face_locations[i], round(closest_distances[0][i][0], 6), 0])
            continue
        # 将阈值范围内的结果均返回
        labels = {}
        temp_result = []
        for j in range(len(closest_distances[0][i])):
            if closest_distances[0][i][j]<=distance_threshold:
                # labels are in classes_
                l = knn_clf.classes_[knn_clf._y[closest_distances[1][i][j]]]
                #results.append( (l, X_face_locations[i], round(closest_distances[0][i][j], 6)) )
                if l not in labels.keys():
                    temp_result.append([
                        l, 
                        X_face_locations[i], 
                        #round(closest_distances[0][i][j], 6)
                        closest_distances[0][i][j]/distance_threshold
                    ])
                    labels[l] = 1
                else:
                    labels[l] += 1

        # 找到labels里count最大值
        max_count = max(labels.items(), key=operator.itemgetter(1))[1]
        # 相同人脸位置，labels 里 count最大的认为就是结果，如果count相同才返回多结果
        #results.extend([i+[labels[i[0]]] for i in temp_result if labels[i[0]]==max_count])
        # 当count最大的不是距离最短的结果时，同时返回距离最短的结果
        #results.extend( [result+[labels[result[0]]] for i,result in enumerate(temp_result) \
        #    if labels[result[0]]==max_count or (i==0 and labels[result[0]]!=max_count)] )
        # 最短距离的不是最大count，且距离短很多时，也加入结果
        temp_result2 = [i+[labels[i[0]]] for i in temp_result if labels[i[0]]==max_count]
        if labels[temp_result[0][0]]!=max_count and temp_result[0][2]/temp_result2[0][2]<0.5:
            temp_result2.insert(0, temp_result[0]+[labels[temp_result[0][0]]])
        results.extend(temp_result2)

    return results


# 识别， 使用 keras
def predict_K(X_base64, group_id, model_path='', face_algorithm='vgg', data_type='base64', request_id=''): 
    """
    Recognizes faces in given image using a trained Keras classifier

    """

    global CLF_CACHE

    # Load a trained Keras model (if one was passed in)
    clf_path = os.path.join(model_path, '%s.%s.h5'%(group_id, face_algorithm))

    # 检查是否已缓存clf
    mtime = int(os.path.getmtime(clf_path)) # 模型最近修改时间

    with cache_lock:
        if (clf_path in CLF_CACHE.keys()) and (CLF_CACHE[clf_path][1]==mtime): 
            model, label_y = CLF_CACHE[clf_path][0]
            #print('Bingo clf cache!', group_id)
        else:
            with graph.as_default():
                with session.as_default():
                    #with open(clf_path, 'rb') as f:
                    #    keras_clf = pickle.load(f)

                    # 读取模型，并识别
                    with open(clf_path+'.save', 'rb') as f:
                        input_dim, output_dim, label_y = pickle.load(f)

                    from train_classifier import get_model
                    model = get_model(input_dim, output_dim)
                    model.load_weights(clf_path)

            # 放进cache
            CLF_CACHE[clf_path] = ((model, label_y), mtime)
            print('Feeding CLF cache: ', CLF_CACHE.keys())

    if data_type=='base64':
        # 动态载入 verify库
        module_verify = import_verify(face_algorithm)

        # Load image file and find face locations
        # Find encodings for faces in the test iamge
        faces_encodings, X_face_locations, faces = module_verify.get_features_b64(X_base64, angle=ALGORITHM[face_algorithm]['p_angle'])

        if len(X_face_locations) == 0:
            return []

        # 保存人脸到临时表, 只保存vgg的
        if request_id!='' and face_algorithm=='vgg':
            dbport.face_save_to_temp(group_id, request_id, image=faces[0])

    else:
        # data_type = 'encodings'
        faces_encodings = X_base64
        X_face_locations = [(0,0,0,0)] # 从db来的数据没有人脸框坐标，只有一个人脸


    with graph.as_default():
        with session.as_default():
            results = []
            for x in range(len(X_face_locations)):

                # 按概率返回结果
                result = model.predict(np.array([faces_encodings[x]]))

                # 整理结果
                max_list = result[0].argsort()[-5:][::-1] # 返回 5 个概率最大的结果
                percent_list = [result[0][i] for i in max_list]
                class_list = label_y.inverse_transform(max_list)
                # 保留概率大于 10% 的结果, 这里返回的评分是（1-概率）, 为与距离表示一致：越小越接近
                result_list = [ [i, X_face_locations[x], 1-j, 1] for i,j in zip(class_list, percent_list) \
                        if j>KERAS_THRESHOLD_PERCENTAGE ] 
                #print(result_list)
                results.extend(result_list)

    return results
