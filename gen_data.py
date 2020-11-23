# -*- coding: utf-8 -*-

import os, shutil
import face_recognition
from PIL import Image
import numpy as np

ratio = 0.5  # train/total
max_person = 50 # 人数量
max_images = 20 # 使用的照片数量


path = '../../face_data/AFDB_face_dataset'
train = 'data/train2'
test = 'data/test2'

#path = '../../face_data/CASIA-maxpy-clean'
#train = 'data/train9'
#test = 'data/test9'

#path = '../../face_data/glint_asia'
#train = 'data/train6'
#test = 'data/test6'


def _HorizontalEyes(PILImg, pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    k = (y2-y1) / (x2-x1)
    angle = np.arctan(k)/np.pi*180
    #print('rotate angle:', angle)
    return PILImg.rotate(angle)


# 获取能取得人脸的照片
def get_face_image(path, file_list): 
    face_file = []
    for i in file_list:
        image = face_recognition.load_image_file(os.path.join(path, i))

        # 调整人脸角度， 按第一个人的角度调整
        #face_landmarks_list = face_recognition.face_landmarks(image)
        #if len(face_landmarks_list)==0:
        #    continue
        #pil_image = Image.fromarray(image)
        #pil_image = _HorizontalEyes(pil_image, [face_landmarks_list[0]['left_eye'][0]] + [face_landmarks_list[0]['right_eye'][0]])
        #image = np.array(pil_image)

        # 获取人脸
        face_bounding_boxes = face_recognition.face_locations(image)
        if len(face_bounding_boxes) == 1:
            face_file.append(i)

        if len(face_file)>max_images:
            break

    return face_file


if __name__ == "__main__":
    dir_list = os.listdir(path)
    dir_list = sorted(dir_list)

    n = 0
    for d in dir_list:
        if n>max_person:
            break

        # 所以文件
        file_list = os.listdir(os.path.join(path, d))
        file_list = sorted(file_list)

        if len(file_list)<max_images: # 至少需要max_images张照片
            print(d, len(file_list), 'skipped')
            continue

        file_list = get_face_image(os.path.join(path, d), file_list)

        # 能取得人脸的照片是否足够？
        if len(file_list)<max_images: # 至少需要max_images张照片
            print(d, len(file_list), 'less face, skipped')
            continue

        print(d, len(file_list))
        n += 1

        # 只使用 max_images 数量的照片
        file_list = file_list[:max_images] 

        # 建输出目录
        output_train = os.path.join(train, d) 
        output_test = os.path.join(test, d) 
        if not os.path.exists(output_train):
            os.mkdir(output_train)
        if not os.path.exists(output_test):
            os.mkdir(output_test)

        # 计算训练图片数量
        train_c = int(len(file_list)*ratio)

        # 生成train的数据
        for i in file_list[:train_c]: 
            shutil.copy(os.path.join(path,d,i), output_train)
            
        # 生成test的数据
        for i in file_list[train_c:]: 
            shutil.copy(os.path.join(path,d,i), output_test)



## 将图片调整为正方形
#
#    path = '../../face_data/AFDB_face_dataset'
#
#    dir_list = os.listdir(path)
#    dir_list = sorted(dir_list)
#
#    n = 0
#    for d in dir_list:
#
#        # 所以文件
#        file_list = os.listdir(os.path.join(path, d))
#        file_list = sorted(file_list)
#
#        print(d, len(file_list))
#
#        for i in file_list:
#            file_path = os.path.join(path, d, i)
#
#            im = Image.open(file_path)
#            n_size = max(im.size)
#            im = im.resize((n_size, n_size))
#            im.save(file_path, quality=95)
