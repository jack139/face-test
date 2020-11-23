# -*- coding: utf-8 -*-

# 特征值分类器训练
import sys
import numpy as np
from sklearn import preprocessing

from keras import models
from keras import layers
from keras import optimizers

from tqdm import tqdm

from facelib import dbport


# 1. 取得训练集 features(从db)
# 2. 取得测试集 features
# 3. 构建网络
# 4. 训练 softmax

#有标签索引对应的元素为 1。其代码实现如下。
def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def get_encodings(encodings_set, method):
    if method=='vgg':
        return encodings_set['vgg'].values()
    elif method=='evo':
        return encodings_set['evo'].values()
    #elif method=='rec':
    #    return encodings_set[2]
    elif method=='plus': # vgg+evo
        plus = []
        for i in encodings_set['vgg'].keys():
            plus.append(encodings_set['vgg'][i]+encodings_set['evo'][i])
        return plus
    elif method=='plus2': # evo+vgg
        plus = []
        for i in encodings_set['vgg'].keys():
            plus.append(encodings_set['evo'][i]+encodings_set['vgg'][i])
        return plus
    #elif method=='tri':
    #    return encodings_set[0]+encodings_set[1]+encodings_set[2]
    return []

def load_data(group_id, face_num=1000, max_user=None, method='vgg'):

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # 按用户分组从db装入特征数据, 装入所有数据
    start = 0
    max_length = 1000
    total = 0
    while 1:
        user_list = dbport.user_list_by_group(group_id, start=start, length=max_length)
        for i in tqdm(range(len(user_list))):
            if max_user and total>=max_user:
                break

            X = []
            y = []

            faces = dbport.user_face_list(group_id, user_list[i])
            for f in faces[:face_num]: # 训练集
                r = dbport.face_info(f)
                if r:
                    e_list = get_encodings(r['encodings'], method)
                    X_train.extend(e_list)
                    y_train.extend([user_list[i]]*len(e_list))

            for f in faces[face_num:]: # 验证集
                r = dbport.face_info(f)
                if r:
                    e_list = get_encodings(r['encodings'], method)
                    X_test.extend(e_list)
                    y_test.extend([user_list[i]]*len(e_list))

            total += 1

        if max_user and total>=max_user:
            break

        if len(user_list)<max_length: 
            break
        else: # 未完，继续
            start += max_length

    # y标签规范化
    label_y = preprocessing.LabelEncoder()
    label_y.fit(y_train)
    y_train = label_y.transform(y_train)
    y_train = to_one_hot(y_train, total)
    y_test = label_y.transform(y_test)
    y_test = to_one_hot(y_test, total)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), label_y



# 创建模型
def get_model(input_dim, output_dim):
    # 三层网络 模型定义
    model = models.Sequential()
    model.add(layers.Dense(1024, activation = 'relu', input_dim = input_dim))
    #model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation = 'softmax'))
    #编译模型
    model.compile(optimizer=optimizers.RMSprop(lr=2e-4), 
        loss='categorical_crossentropy', metrics=['accuracy']) 
    return model


# 模型参数
epochs_num = 100
batch_size = 50


if __name__ == '__main__':
    if len(sys.argv)<3:
        print("usage: python3 %s <algorithm> <group_id> [max_user] [face_num]" % sys.argv[0])
        sys.exit(2)

    method = sys.argv[1]
    group_id = sys.argv[2]

    if len(sys.argv)>3:
        max_user = int(sys.argv[3]) # 最大训练人数
    else:
        max_user = None

    if len(sys.argv)>4:
        face_num = int(sys.argv[4]) # 同一个人训练的人脸数
    else:
        face_num = 1000


    X_train, y_train, X_test, y_test, label_y = load_data(group_id, face_num=face_num, max_user=max_user, method=method)

    input_dim = len(X_train[0])
    output_dim = len(y_train[0])

    model = get_model(input_dim, output_dim)
    model.summary()

    print('input_dim=', input_dim, 'output_dim=', output_dim, ' batch_size=', batch_size, ' epochs_num=', epochs_num)

    history = model.fit(X_train, y_train, 
            epochs=epochs_num, 
            batch_size=batch_size, 
            verbose=1,
            #validation_data=(X_test, y_test)
        )

    # 评估预测结果
    #results = model.evaluate(X_test, y_test, verbose=1)
    #print('predict: ', results)


    # 保存模型 和 标签数据
    import pickle

    h5_filename = '%s.%s.h5'%(group_id, method)

    model.save(h5_filename)

    with open(h5_filename+'.save', 'wb') as f:
        pickle.dump((input_dim, output_dim, label_y), f)

    # 读取模型，并识别
    with open(h5_filename+'.save', 'rb') as f:
        input_dim, output_dim, label_y = pickle.load(f)

    model = get_model(input_dim, output_dim)
    model.load_weights(h5_filename)

    #result = model.predict_classes(X_test[:1])
    #name = label_y.inverse_transform(result)
    #print(name[0])

    # 按概率返回结果
    result = model.predict(X_train[:1])
    max_list = result[0].argsort()[-5:][::-1] # 返回 5 个概率最大的结果
    percent_list = [result[0][i] for i in max_list]
    class_list = label_y.inverse_transform(max_list)
    result_list = [ i for i in zip(class_list, percent_list) if i[1]>0.5 ] # 保留概率大于 50% 的结果
    print(result_list)
