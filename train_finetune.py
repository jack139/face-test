# -*- coding: utf-8 -*-

# vggface 数据增强训练, 
import time
import numpy as np
from sklearn import preprocessing

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from facelib import dbport
from models.vggface.keras_vggface.vggface import VGGFace

# 1. 生成训练集和验证集
# 2. 构建网络
# 3. 训练新增的 classifier
# 4. 解冻conv5_3，训练
# 5. 解冻conv5_2，训练
#


# 模型参数
epochs_num = 100
batch_size = 20
steps_per_epoch = 200  # batch_size * steps_per_epoch < 训练样本数 （train8 = 448*10）
target_size = (224, 224)
output_dim = 448
train_dir = '../data/train8'
validation_dir = '../data/test8'

# 创建模型
def get_model(output_dim):
    vgg_model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') 
    last_layer = vgg_model.get_layer('avg_pool').output
    x = layers.Flatten(name='flatten_added')(last_layer) # 
    #x2 = layers.Dense(1024, activation = 'relu')(x)
    #x3 = layers.Dropout(0.2)(x2)
    out = layers.Dense(output_dim, activation='softmax', name='classifier')(x) 
    custom_vgg_model = models.Model(vgg_model.input, out)
    return custom_vgg_model

def freeze(model, freeze_layer):
    set_trainable = False
    for layer in model.layers:
        if layer.name == freeze_layer: #'conv5_3_1x1_reduce': # 'flatten_added' 确定冻结的位置， 此层以前都冻结
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    #编译模型
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), 
        loss='categorical_crossentropy', metrics=['accuracy']) 
    return model


# 去掉分类器权重, 用于获取特征值
def extrace_weight(): 
    vgg_model = VGGFace(model='senet50', include_top=True, input_shape=(224, 224, 3), pooling='avg') 
    last_layer = vgg_model.get_layer('avg_pool').output
    no_classifier_model = models.Model(vgg_model.input, last_layer)
    no_classifier_model.summary()
    no_classifier_model.save('no_classifier.h5')

def extrace_weight2(weight_file): 
    #vgg_model = VGGFace(model='senet50', include_top=True, input_shape=(224, 224, 3), pooling='avg', weights=weight_file) 
    vgg_model = get_model(output_dim)
    vgg_model.load_weights(weight_file)
    last_layer = vgg_model.get_layer('avg_pool').output
    no_classifier_model = models.Model(vgg_model.input, last_layer)
    no_classifier_model.summary()
    no_classifier_model.save('no_classifier.h5')


if __name__ == '__main__':

    model = get_model(output_dim)


    print('output_dim=', output_dim, ' batch_size=', batch_size, ' epochs_num=', epochs_num)


    train_datagen = ImageDataGenerator(
        rescale=None,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=None)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')


    # 顺序训练时，冻结、解冻的层
    #freeze_layers = ['flatten_added', 'conv5_3_1x1_reduce', 'conv5_2_1x1_reduce', 'conv5_1_1x1_reduce']
    freeze_layers = ['flatten_added', 'conv5_3_1x1_reduce']

    for i in range(len(freeze_layers)):
        # 冻结相应的层
        model = freeze(model, freeze_layers[i])
        model.summary()

        #if i==0: # 装入预训练的权重
        #    print('load pretrained weights')
        #    model.load_weights('trained2_1594028604.h5')

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_num,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch//2)

        # 保存权重
        model.save('trained%d_%d.h5'%(i, int(time.time())))


