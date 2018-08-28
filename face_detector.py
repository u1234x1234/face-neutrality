from typing import List

import cv2
import numpy as np

import dlib


class FaceDetector:
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image: np.ndarray) -> List[dlib.rectangle]:
        raw_bboxes, raw_scores, raw_idxs = self._detector.run(image, 0, 0)
        bboxes = []
        for bbox, score, _ in zip(raw_bboxes, raw_scores, raw_idxs):
            if score < 0.:
                continue
            bboxes.append(bbox)
        return bboxes
