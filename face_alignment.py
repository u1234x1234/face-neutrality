from typing import List, Tuple

import cv2
import numpy as np

import dlib


class FaceAlignment:
    N_KEYPOINTS = 68
    LEFT_EYE_INDICES = range(36, 42)
    RIGHT_EYE_INDICES = range(42, 48)
    INNER_MOUTH_INDICES = range(48, 60)
    OUTER_MOUTH_INDICES = range(61, 68)

    def __init__(self, model_path):
        self._keypoint_detector = dlib.shape_predictor(model_path)

    def crop_mouth(self, image: np.ndarray, bbox: dlib.rectangle) -> np.ndarray:
        """Crop a mouth from the face `bbox`
        """
        mouth_y1 = 0.6
        mouth_y2 = 0.9
        mouth_x1 = 0.15
        mouth_x2 = 0.85

        keypoints = self.detect_keypoints(image, bbox)
        face_image, _ = FaceAlignment.crop_face(image, keypoints)
        h, w = face_image.shape[:2]
        mouth_image = face_image[int(mouth_y1*h): int(mouth_y2*h), int(mouth_x1*w): int(mouth_x2*w)]
        return mouth_image

    def detect_keypoints(self, image: np.ndarray, bbox: dlib.rectangle) -> List[Tuple[int, int]]:
        keypoints = self._keypoint_detector(image, bbox)
        keypoints = [(keypoints.part(i).x, keypoints.part(i).y) for i in range(keypoints.num_parts)]
        return keypoints

    @staticmethod
    def get_mean_position(keypoints):
        return tuple(np.mean(keypoints, axis=0).astype(np.int))

    @staticmethod
    def crop_face(image: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """Given an `image` and 68 `keypoints`, crop a face area

        Parameters
        ----------
        image : np.ndarray
            Image in cv2 format
        keypoints : list
            List with 68 facial keypoints

        Returns
        -------
        mouth_image : np.ndarray
            Image cropped from the detected mouth area
        """
        left_eye_dst = (50, 50)
        right_eye_dst = (150, 50)
        mouth_dst = (100, 150)
        face_size = 200

        left_eye_src = FaceAlignment.get_mean_position([keypoints[idx] for idx in FaceAlignment.LEFT_EYE_INDICES])
        right_eye_src = FaceAlignment.get_mean_position([keypoints[idx] for idx in FaceAlignment.RIGHT_EYE_INDICES])
        mouth_src = FaceAlignment.get_mean_position([keypoints[idx] for idx in FaceAlignment.INNER_MOUTH_INDICES])

        src_pts = np.float32([left_eye_src, right_eye_src, mouth_src])
        dst_pts = np.float32([left_eye_dst, right_eye_dst, mouth_dst])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        face_image = cv2.warpAffine(image, M, (face_size, face_size))

        return face_image, M
