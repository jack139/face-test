# -*- coding: utf-8 -*-
from pymongo import MongoClient

############ mongodb 设置
#db_serv_list='172.17.0.1'
#cli = {
#    'web' : MongoClient(db_serv_list),
#}
#db_web = cli['web']['face_db']
#db_web.authenticate('ipcam','ipcam')
#db_primary = db_web

############# 算法相关设置

'''
                        train2  train3  train4  train8  train9  
    rec                 0.43    0.58    0.45    0.38    0.52
    vgg     senet50     0.79    0.92    0.85    0.73    0.85    
    vgg     resnet50    0.86    0.99
    evo     ir152       1.15    1.22    1.19                    
    evo     bh-ir50     1.12    1.14
    plus                1.45    1.56
'''
ALGORITHM = {
    'rec'   : { 'distance_threshold' : 0.58, 'p_angle':None, 'ext' : '.rec.clf',  'module' : 'models.face_rec.verify' },
    'vgg'   : { 'distance_threshold' : 0.92, 'p_angle':None, 'ext' : '.vgg.clf',  'module' : 'models.vggface.verify' }, 
    'evo'   : { 'distance_threshold' : 1.22, 'p_angle':0,    'ext' : '.evo.clf',  'module' : 'models.face_evoLVe.verify' },
    'plus'  : { 'distance_threshold' : 1.45, 'p_angle':None, 'ext' : '.plus.clf', 'module' : 'models.verify_plus' },
}

# 深度网络分类器，百分比阈值
KERAS_THRESHOLD_PERCENTAGE = 0.1

# 并行算法设置， 按分类器类型区分
algorithm_settings = {
    'knn' : {
        1 : [ 'vgg', 'data/model/train6.vgg.clf' ], # 优先返回
        2 : [ 'evo', 'data/model/train2_ir152.evo.clf' ],
        #2 : [ 'rec', 'data/model/train4.rec.clf' ],
        #2 : [ 'null', '' ], # 空算法 
    },
    'keras' : {
        #1 : [ 'vgg', '' ], # 优先返回
        #2 : [ 'evo', ''],
        1 : [ 'plus', '' ], # 特征合并 vgg+evo
        #2 : [ 'plus2', '' ], # 特征合并 evo+vgg
        2 : [ 'null', '' ], # 空算法 
    },
    'api' : { # api 中并行获取特征值的设置 -- 不要修改
        1 : [ 'vgg', '' ], # 优先返回
        2 : [ 'evo', ''],
    },
}

# 训练时角度修正：
#TRAINING_ANGLE = [None] # 不修正
TRAINING_ANGLE = [None, 360] # 水平镜像
#TRAINING_ANGLE = [None, 0, 360] # 按修正双眼水平，水平镜像
#TRAINING_ANGLE = [None, 0, -20, -5, 5, 20] # 水平修正后转多个角度

# 注册时角度修正：
IMPORT_ANGLE = [None, 360] # 水平镜像

# 特征值训练模型保存路径
TRAINED_MODEL_PATH = 'data/model'

# vgg 预训练权重，vggface使用默认权重文件，其他为自定义文件路径
#VGGFACE_WEIGHTS = 'vggface'
VGGFACE_WEIGHTS = 'data/h5/train_ft4a.h5'

# face.evoLVe 模型l路径
EVO_MODEL_BASE = '/home/gt/Codes/face_model/face.evoLVe.PyTorch/'
