# -*- coding: utf-8 -*-

import sys, time
import os
import os.path
import signal
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from config.settings import IMPORT_ANGLE, VGGFACE_WEIGHTS
from facelib.utils import import_verify
from facelib import dbport


'''
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
'''

# 导入图片特征： vgg evo rec deep

SHUTDOWN = False

def signal_handler(signal,frame):
    global SHUTDOWN
    SHUTDOWN = True
    print('You pressed Ctrl+C! please wait to SHUTDOWN ... ')

 
signal.signal(signal.SIGINT,signal_handler)


if __name__ == "__main__":
    if len(sys.argv)<3:
        print("usage: python3 %s <train_data_dir> <group_id>" % sys.argv[0])
        sys.exit(2)

    train_dir = sys.argv[1]
    group_id = sys.argv[2]

    if dbport.group_info(group_id)==-1:
        # 新建分组
        dbport.group_new(group_id)

    # 动态载入 verify库
    module_verify = [ 
        import_verify('vgg'), 
        import_verify('evo'), 
    ]

    # Loop through each person in the training set
    for class_dir in sorted(os.listdir(train_dir)):
        if SHUTDOWN: # 处理完一个目录后再退出
            print('done.')
            break

        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        if dbport.user_info(group_id, class_dir)!=-1:
            print('existed: ', class_dir)
            continue

        # 新建用户
        dbport.user_new(group_id, class_dir, name=class_dir)

        print('import: ', class_dir)

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            encodings = {
                'vgg' : { },
                'evo' : { }
            }
            face_image = None

            for angle in IMPORT_ANGLE: # 旋转不同角度训练 multi2
                face_encodings_vgg, _, face_list = module_verify[0].get_features(img_path, angle=angle)
                face_encodings_evo, _, _ = module_verify[1].get_features(img_path, angle=angle)

                if len(face_encodings_vgg) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_encodings) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    encoding_vgg = face_encodings_vgg[0]
                    encoding_evo = face_encodings_evo[0]
                    if type(encoding_vgg)!=type([]):
                        encoding_vgg = encoding_vgg.tolist()
                    if type(encoding_evo)!=type([]):
                        encoding_evo = encoding_evo.tolist()

                    encodings['vgg'][str(angle)] = encoding_vgg
                    encodings['evo'][str(angle)] = encoding_evo

                    if angle==None:
                        face_image = face_list[0]

            # 添加人脸特征
            filepath, filename = os.path.split(img_path)

            face_id = dbport.face_new('vgg_evo', encodings, image=face_image, file_ref=filename, weight_ref=VGGFACE_WEIGHTS)
            # 人脸数据添加到用户信息
            dbport.user_add_face(group_id, class_dir, face_id)

            time.sleep(0.5)
