# -*- coding: utf-8 -*-

# 使用两个算法模型并行识别

import os, sys
import base64
#from datetime import datetime
from models.predict_plus import predict_parallel, predict_thread_db



if __name__ == "__main__":
    if len(sys.argv)<4:
        print("usage: python3 %s <knn|keras> <group_id> <test dir or file>" % sys.argv[0])
        sys.exit(2)

    #from facelib import api_func
    
    classifier = sys.argv[1]
    group_id = sys.argv[2]
    test_thing = sys.argv[3]

    if classifier not in ['knn', 'keras']:
        print('invalid classifier!')
        sys.exit(3)

    if os.path.isdir(test_thing):
        images = os.listdir(test_thing)
        images = [os.path.join(test_thing, i) for i in images]
    else:
        images = [ test_thing ]

    # Using the trained classifier, make predictions for unknown images
    for image_file in images:
        print("Looking for faces in {}".format(image_file))

        with open(image_file, 'rb') as f:
            image_data = f.read()

        image_b64 = base64.b64encode(image_data)

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        
        #predictions = api_func.face_search('', image_b64, group_id)

        #start_time = datetime.now()
        predictions = predict_parallel(predict_thread_db, image_b64, group_id, classifier=classifier)
        #print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        # Print results on the console
        for name, (top, right, bottom, left), distance, count in predictions:
            print("- Found {} at ({}, {}), distance={}, count={}".format(name, left, top, distance, count))
        #for i in predictions:
        #    print("- Found {} at {}, distance={}".format(i['user_id'], i['location'], i['score']))
        if len(predictions)==0:
            print('Face not found!')

        #print(predictions)
        # Display results overlaid on an image
        #knn.show_prediction_labels_on_image(image_file, predictions)


