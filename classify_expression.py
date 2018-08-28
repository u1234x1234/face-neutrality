import argparse
import glob
import os
import shutil
from typing import List, Tuple

import cv2

from expression_classifier import NNExpressionClassifier
from face_alignment import FaceAlignment
from face_detector import FaceDetector
from utils import preprocess_image

SMILE_EXPRESSION_THRESHOLD = 0.35
MOUTH_OPEN_EXPRESSION_THRESHOLD = 0.3


def classify_expressions(image_paths: List[str],
                         facial_landmark_detection_model_path: str,
                         expression_classifier_model_prefix: str) -> Tuple[List[str], List[str]]:
    """Analyze each image from `image_paths` list and return two lists:
    1. Paths to the images with smile expression
    2. Paths to the iamges with mouth_open expression

    Parameters
    ----------
    image_paths : list
        List with images paths
    facial_landmark_detection_model_path : str
        Path to the dlib facail landmark detection model
    expression_classifier_model_prefix : str
        Path to the expression classifier
    """
    face_detector = FaceDetector()
    face_alignment_model = FaceAlignment(facial_landmark_detection_model_path)
    expression_classifier = NNExpressionClassifier(expression_classifier_model_prefix)

    images_with_smile = set()
    images_with_open_mouth = set()
    for path in image_paths:
        image = cv2.imread(path)
        image = preprocess_image(image)
        bboxes = face_detector.detect_faces(image)

        for bbox in bboxes:
            mouth_image = face_alignment_model.crop_mouth(image, bbox)
            smile_prob, mouth_open_prob = expression_classifier.predict(mouth_image)
            if smile_prob > SMILE_EXPRESSION_THRESHOLD:
                images_with_smile.add(path)
            if mouth_open_prob > MOUTH_OPEN_EXPRESSION_THRESHOLD:
                images_with_open_mouth.add(path)

    return list(images_with_smile), list(images_with_open_mouth)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--in_dir', type=str, required=True)
    arg_parser.add_argument('--out_dir', type=str, required=True)
    arg_parser.add_argument('--facial_landmark_detection_model_path', type=str, default='shape_predictor_68_face_landmarks.dat')
    arg_parser.add_argument('--expression_classifier_model_prefix', type=str, default='expression_classifier')
    args = arg_parser.parse_args()

    image_paths = glob.glob(args.in_dir + '/*.jpg')
    print('Number of images in dir {} is {}'.format(args.in_dir, len(image_paths)))
    images_with_smile, images_with_mouth_open = classify_expressions(
        image_paths, args.facial_landmark_detection_model_path, args.expression_classifier_model_prefix)

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir)
    for out_sub_dir, image_list in [['smile', images_with_smile], ['mouth_open', images_with_mouth_open]]:
        os.makedirs(os.path.join(args.out_dir, out_sub_dir))
        for p in image_list:
            dst = os.path.join(args.out_dir, out_sub_dir, os.path.basename(p))
            shutil.copy(p, dst)
