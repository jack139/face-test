# -*- coding: utf-8 -*-

import os, sys
import base64
from datetime import datetime
from config.settings import ALGORITHM
from models import knn_db


if __name__ == "__main__":
    if len(sys.argv)<5:
        print("usage: python3 %s <algorithm> <group_id> <model_path> <test dir or file> [keras]" % sys.argv[0])
        sys.exit(2)

    face_algorithm = sys.argv[1]

    if face_algorithm not in ALGORITHM.keys():
        print('Algorithm not found!')
        sys.exit(2)

    group_id = sys.argv[2]
    model_path = sys.argv[3]
    test_thing = sys.argv[4]

    if len(sys.argv)==6:
        classifier = sys.argv[5]
    else:
        classifier = 'knn'

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
        start_time = datetime.now()
        if classifier=='knn':
            predictions = knn_db.predict(image_b64, group_id,
                model_path = model_path,
                distance_threshold=ALGORITHM[face_algorithm]['distance_threshold'],
                face_algorithm=face_algorithm)
        else:
            predictions = knn_db.predict_K(image_b64, group_id,
                model_path = model_path,
                face_algorithm=face_algorithm)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        # Print results on the console
        for name, (top, right, bottom, left), distance, count in predictions:
            print("- Found {} at ({}, {}), distance={}, count={}".format(name, left, top, distance, count))

        if len(predictions)==0:
            print('Face not found!')
