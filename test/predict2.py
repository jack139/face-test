# -*- coding: utf-8 -*-

# 使用两个算法模型并行识别

import os, sys
from datetime import datetime
from models.predict_plus import predict_parallel, predict_thread


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <test dir or file>" % sys.argv[0])
        sys.exit(2)

    test_thing = sys.argv[1]

    if os.path.isdir(test_thing):
        images = os.listdir(test_thing)
        images = [os.path.join(test_thing, i) for i in images]
    else:
        images = [ test_thing ]

    # Using the trained classifier, make predictions for unknown images
    for image_file in images:
        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        
        start_time = datetime.now()
        predictions = predict_parallel(predict_thread, image_file)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        # Print results on the console
        for name, (top, right, bottom, left), distance, count in predictions:
            print("- Found {} at ({}, {}), distance={}, count={}".format(name, left, top, distance, count))
        if len(predictions)==0:
            print('Face not found!')

        # Display results overlaid on an image
        #knn.show_prediction_labels_on_image(image_file, predictions)


