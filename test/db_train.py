# -*- coding: utf-8 -*-

import os, sys
from config.settings import ALGORITHM
from models import knn_db
from facelib import dbport


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <group_id> [max_user] [face_num] [need_train]" % sys.argv[0])
        sys.exit(2)

    group_id = sys.argv[1]

    if len(sys.argv)>2:
        max_user = int(sys.argv[2]) # 最大训练人数
    else:
        max_user = None

    if len(sys.argv)>3:
        face_num = int(sys.argv[3]) # 同一个人训练的人脸数
    else:
        face_num = 1000

    if len(sys.argv)>4:
        need_train = int(sys.argv[4]) # need_train 标记
    else:
        need_train = None

    face_algorithm = ['vgg', 'evo' ] #, 'deep', 'rec'

    for algorithm in face_algorithm:
        # Train the KNN classifier and save it to disk
        print("Training KNN classifier...")
        classifier = knn_db.train(group_id, 
            model_save_path=group_id + ALGORITHM[algorithm]['ext'], 
            n_neighbors=10,
            face_algorithm=algorithm,
            max_user=max_user,
            face_num=face_num,
            need_train=need_train)
        print("Training complete!")
