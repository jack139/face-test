"""
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

"""
import os
import math, operator
from datetime import datetime
from sklearn import svm
import pickle
from facelib import dbport
from config.settings import ALGORITHM, TRAINED_MODEL_PATH
from facelib.utils import import_verify
from tqdm import tqdm

CLF_CACHE = {}


# 训练
# face_algorithm 只 支持 evo 和 vgg
def train(group_id, encodings_index=0, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', 
    verbose=False, face_algorithm='rec'):
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
        user_list = dbport.user_list_by_group(group_id, start=start, length=max_length)
        for i in tqdm(range(len(user_list))):
            faces = dbport.user_face_list(group_id, user_list[i])
            for f in faces:
                r = dbport.face_info(f)
                if r:
                    X.append(r['encodings'][encodings_index])
                    y.append(user_list[i])

        total += len(user_list)
        if len(user_list)<max_length: 
            break
        else: # 未完，继续
            start += max_length

    print('Data loaded:', total)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    start_time = datetime.now()
    svc_clf = svm.LinearSVC()
    svc_clf.fit(X, y)
    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(svc_clf, f)

    return svc_clf


# 识别
def predict(X_base64, group_id, model_path='', distance_threshold=0.6, face_algorithm='vgg', data_type='base64'): 
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_base64: image data in base64 coding
    :param model_path: (optional) 已训练模型路径，默认当前路径
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    global CLF_CACHE

    # Load a trained KNN model (if one was passed in)
    clf_path = os.path.join(model_path, group_id+ALGORITHM[face_algorithm]['ext'])

    # 检查是否已缓存clf
    mtime = int(os.path.getmtime(clf_path)) # 模型最近修改时间

    if (clf_path in CLF_CACHE.keys()) and (CLF_CACHE[clf_path][1]==mtime): 
        svc_clf = CLF_CACHE[clf_path][0]
        #print('Bingo clf cache!', group_id)
    else:
        with open(clf_path, 'rb') as f:
            svc_clf = pickle.load(f)
        # 放进cache
        CLF_CACHE[clf_path] = (svc_clf, mtime)
        print('Feeding CLF cache: ', CLF_CACHE.keys())

    if data_type=='base64':
        # 动态载入 verify库
        module_verify = import_verify(face_algorithm)

        # Load image file and find face locations
        # Find encodings for faces in the test iamge
        faces_encodings, X_face_locations = module_verify.get_features_b64(X_base64)

        if len(X_face_locations) == 0:
            return []
    else:
        # data_type = 'encodings'
        faces_encodings = X_base64
        X_face_locations = [(0,0,0,0)] # 从db来的数据没有人脸框坐标，只有一个人脸

    #print(faces_encodings)

    # Use the SVM model to find the first the best matches for the test face
    name = svc_clf.predict(faces_encodings)
    #print(name)

    # return multi results
    results = []

    for i in range(len(X_face_locations)):
        results.append([
            name[i], 
            X_face_locations[i], 
            0,
            1
        ])

    return results
