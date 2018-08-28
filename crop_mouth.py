import argparse
import glob
import multiprocessing
import os
from typing import List

import cv2

from face_alignment import FaceAlignment
from face_detector import FaceDetector
from utils import preprocess_image


def crop_mouth(image_paths: List[str], out_dir: str, face_alignment_model_path: str):
    face_detector = FaceDetector()
    face_alignment_model = FaceAlignment(face_alignment_model_path)
    for p in image_paths:
        image = cv2.imread(p)
        image = preprocess_image(image)
        bboxes = face_detector.detect_faces(image)

        for bbox in bboxes:
            mouth_image = face_alignment_model.crop_mouth(image, bbox)
            cv2.imwrite('{}/{}'.format(out_dir, os.path.basename(p)), mouth_image)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--in_path', type=str, required=True)
    arg_parser.add_argument('--out_path', type=str, required=True)
    arg_parser.add_argument('--face_alignment_model_path', type=str, default='shape_predictor_68_face_landmarks.dat')
    args = arg_parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    image_paths = glob.glob(args.in_path + '/*.jpg')
    pool = []
    n_processes = multiprocessing.cpu_count()
    for i in range(n_processes):
        chunk = image_paths[i::n_processes]
        p = multiprocessing.Process(target=crop_mouth, args=(chunk, args.out_path, args.face_alignment_model_path))
        p.start()
        pool.append(p)
    for p in pool:
        p.join()
