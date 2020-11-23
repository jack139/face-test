# -*- coding: utf-8 -*-

import math, operator
from datetime import datetime
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from config.settings import ALGORITHM, TRAINING_ANGLE
from facelib.utils import import_verify

#from . import verify

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# 训练
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False, face_algorithm='rec'):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # 动态载入 verify库
    module_verify = import_verify(face_algorithm)

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        print('training: ', class_dir)

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            for angle in TRAINING_ANGLE: # 旋转不同角度训练 multi2
                face_encodings, _, _ = module_verify.get_features(img_path, angle=angle)

                if len(face_encodings) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_encodings) < 1 else "Found more than one face"))
                elif len(face_encodings)>0:
                    # Add face encoding for current image to the training set
                    X.append(face_encodings[0])
                    y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    start_time = datetime.now()
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

        # 保存 X,y
        with open(model_save_path+'.xy', 'wb') as f:
            pickle.dump((X, y), f)

    return knn_clf


# 识别
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6, face_algorithm='rec'):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 动态载入 verify库
    module_verify = import_verify(face_algorithm)

    # Load image file and find face locations
    # Find encodings for faces in the test iamge
    #  angle=None 识别时不修正角度， angle=0 识别时修正角度
    faces_encodings, X_face_locations, _ = module_verify.get_features(X_img_path, angle=ALGORITHM[face_algorithm]['p_angle'])

    if len(X_face_locations) == 0:
        return []

    #print(faces_encodings)

    # Use the KNN model to find the first 5 best matches for the test face
    # 返回5个最佳结果
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=10)
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


# 在照片中显示结果
def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left), _ in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


# 计算X.y的f1和acc,确定threshold值
def score_acc_f1(X, y, face_algorithm='rec', title='', show=True):
    from sklearn.metrics import f1_score, accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np


    distances = [] # squared L2 distance between pairs
    identical = [] # 1 if same identity, 0 otherwise

    module_verify = import_verify(face_algorithm)

    num = len(y)

    from tqdm import tqdm
    for i in tqdm(range(num - 1)):
        for j in range(i + 1, num):
            distances.append(module_verify.face_distance([X[i]], X[j]))
            identical.append(1 if y[i] == y[j] else 0)
            
    distances = np.array(distances)
    identical = np.array(identical)

    thresholds = np.arange(0.1, 2.0, 0.01)

    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)

    if show:
        # Plot F1 score and accuracy as function of distance threshold
        plt.plot(thresholds, f1_scores, label='F1 score')
        plt.plot(thresholds, acc_scores, label='Accuracy')
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Accuracy at threshold {:.4} = {:.4}'.format(opt_tau, opt_acc))
        plt.xlabel(title + ' Distance threshold')
        plt.legend()
        plt.show()

    return opt_tau, opt_acc