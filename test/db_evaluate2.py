# -*- coding: utf-8 -*-

# 使用两个算法模型并行识别

import os, sys
import base64
from datetime import datetime
from models.predict_plus import predict_parallel, predict_thread_db
from facelib import dbport


if __name__ == "__main__":
    if len(sys.argv)<4:
        print("usage: python3 %s <group_id> <test_group_id> <knn|keras> [length_num] [last_face_num]" % sys.argv[0])
        sys.exit(2)

    group_id = sys.argv[1]
    test_group_id = sys.argv[2]
    classifier = sys.argv[3]

    if classifier not in ['knn', 'keras']:
        print('invalid classifier!')
        sys.exit(3)

    if len(sys.argv)>4:
        length_num = int(sys.argv[4])
    else:
        length_num = 1000

    if len(sys.argv)>5:
        last_face_num = int(sys.argv[5])
    else:
        last_face_num = 1000


    user_list = dbport.user_list_by_group(test_group_id, length=length_num)

    #persons = os.listdir(test_path)
    total_acc = total_acc2 = 0

    # name      total   correct wrong   fail    multi     second        acc      acc2            preci        elapsed time
    # 样本名     总数     正确数   错误数   失败数   返回多结果  非第一结果正确   正确率    非第一结果正确率   非失败正确率    耗时
    print('name\t\ttotal\tcorrect\twrong\tfail\tmulti\tsecond\tacc\tacc2\tpreci\telapsed time')
    #for p in persons:
    for i in range(len(user_list)):
        p = user_list[i]

        faces_list = dbport.user_face_list(test_group_id, user_list[i])

        # Using the trained classifier, make predictions for unknown images
        total = len(faces_list[-last_face_num:])
        correct = 0
        wrong = 0
        fail = 0
        multi = 0 # 匹配多个结果
        second = 0 # 不是首个匹配结果
        start_time = datetime.now()
        for face in faces_list[-last_face_num:]: # 指定人的倒数几个人脸，用于使用train做测试
            # 识别每个人脸
            r = dbport.face_info(face)
            if r is None:
                continue

            # 并行识别
            predictions = predict_parallel(predict_thread_db, r['encodings'], group_id, 'encodings', classifier=classifier)

            # Print results on the console
            if len(predictions)==0:
                fail += 1
            else:
                n = 0
                bingo = 0
                name_list = []
                for name, (top, right, bottom, left), distance, count in predictions:
                    if name==p:
                        if n==0:
                            correct += 1
                            bingo += 1
                        elif bingo==0:
                            second += 1
                    else:
                        if n==0:
                            wrong += 1

                    if name not in name_list:
                        name_list.append(name)

                    n += 1

                if len(name_list)>1:
                    multi += 1


        print('%10s\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s'%\
            (p, total, correct, wrong, fail, multi, second, correct/total, \
            (correct+second)/total, correct/(total-fail), datetime.now() - start_time))

        total_acc += correct/total
        total_acc2 += (correct+second)/total

    print('total_acc: %.3f'%(total_acc/len(user_list)))
    print('total_acc2: %.3f'%(total_acc2/len(user_list)))
