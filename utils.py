import cv2
import numpy as np
from imgaug import augmenters


def sometimes(augmenter):
    return augmenters.Sometimes(0.3, augmenter)


spatial_invariant_augmenters = augmenters.Sequential([
    sometimes(augmenters.Add(value=(-10, 30))),
    sometimes(augmenters.ContrastNormalization((0.5, 2.0))),
    sometimes(augmenters.AdditiveGaussianNoise(scale=(0, 0.03*255))),
    sometimes(augmenters.Multiply(mul=(0.9, 1.1))),
    sometimes(augmenters.GaussianBlur(sigma=(0.0, 1.0))),
])


def get_transform(image_w: int, image_h: int, w: int, h: int,
                  min_scale_ratio: float, max_scale_ratio: float, max_rotate_angle: int, shift_ratio: float):
    scale = h / image_h
    scale *= np.random.uniform(min_scale_ratio, max_scale_ratio)

    angle = np.random.randint(0, max_rotate_angle + 1)
    center = (image_w / 2, image_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=scale)

    sx = (w - scale * image_w) / 2
    sy = (h - scale * image_h) / 2
    sx += sx*np.random.uniform(-shift_ratio, shift_ratio)
    sy += sy*np.random.uniform(-shift_ratio, shift_ratio)
    M[0][2] = sx
    M[1][2] = sy
    return M


def preprocess_mouth_image(image: np.ndarray, w: int=140, h: int=60, augment: bool=False):
    if augment:
        M = get_transform(image_w=image.shape[1], image_h=image.shape[0], w=w, h=h, min_scale_ratio=0.9,
                          max_scale_ratio=1.1, shift_ratio=0.1, max_rotate_angle=10)
    else:
        M = get_transform(image_w=image.shape[1], image_h=image.shape[0], w=w, h=h, min_scale_ratio=1.0,
                          max_scale_ratio=1.0, shift_ratio=0., max_rotate_angle=0)

    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if augment:
        image = spatial_invariant_augmenters.augment_image(image)

    return image


def preprocess_image(image, size=400, scale=0.7):
    scale = size / float(min(image.shape[:2])) * scale
    M = np.array([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (size - scale * image.shape[1]) / 2
    M[1][2] = (size - scale * image.shape[0]) / 2
    image = cv2.warpAffine(image, M, (size, size))
    return image
