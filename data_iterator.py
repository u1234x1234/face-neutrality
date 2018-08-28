import cv2
import mxnet as mx
import numpy as np
import multiprocessing
from collections import defaultdict

from utils import preprocess_mouth_image


class DataIterator:
    def __init__(self, labels, context, h: int=140, w: int=224,
                 batch_size: int=32, n_processes: int=2, epoch_size: int=1000, augment: bool=False):
        self.labels = labels
        self.context = context
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.augment = augment

        self._cursor = -1
        self._manager = multiprocessing.Manager()
        self._process_pool = []
        self._batch_queue = self._manager.Queue(8)

        self._images_with_smile = []
        self._images_with_open_mouth = []
        for p, (is_smile, is_open_mouth) in labels:
            if is_smile:
                self._images_with_smile.append((p, (is_smile, is_open_mouth)))
            if is_open_mouth:
                self._images_with_open_mouth.append((p, (is_smile, is_open_mouth)))

        for _ in range(n_processes):
            pt = multiprocessing.Process(target=self._gen, args=())
            pt.start()
            self._process_pool.append(pt)

    def _gen(self):
        np.random.seed()
        while True:
            if self.augment:
                batch = []

                n_images_with_smile = int(0.2 * self.batch_size)
                idxs = np.random.randint(0, len(self._images_with_smile), size=n_images_with_smile)
                batch += [self._images_with_smile[idx] for idx in idxs]

                n_images_with_open_mouth = int(0.2 * self.batch_size)
                idxs = np.random.randint(0, len(self._images_with_open_mouth), size=n_images_with_open_mouth)
                batch += [self._images_with_open_mouth[idx] for idx in idxs]

                n_rem = self.batch_size - n_images_with_smile - n_images_with_open_mouth
                idxs = np.random.randint(0, len(self.labels), size=n_rem)
                batch += [self.labels[idx] for idx in idxs]
            else:
                idxs = np.random.randint(0, len(self.labels), size=self.batch_size)
                batch = [self.labels[idx] for idx in idxs]

            images = []
            labels = []

            for p, l in batch:
                image = cv2.imread(p)
                if self.augment:
                    image = preprocess_mouth_image(image, augment=True)

                images.append(image)
                labels.append(l)

            images = np.array(images).transpose(0, 3, 1, 2)
            labels = np.array(labels)

            self._batch_queue.put((images, labels))

    def reset(self):
        self._cursor = -1

    def iter_next(self):
        self._cursor += self.batch_size
        if(self._cursor < self.epoch_size):
            return True
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_next():
            data, labels = self._batch_queue.get(block=True)
            return mx.nd.array(data, self.context), mx.nd.array(labels, self.context)
        else:
            self.reset()
            raise StopIteration
