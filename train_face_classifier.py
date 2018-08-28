import argparse
import glob
import os
import pandas as pd
import mxnet as mx
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

from data_iterator import DataIterator


class ClassificationBlock(mx.gluon.HybridBlock):
    def __init__(self, context, **kwargs):
        super(ClassificationBlock, self).__init__(**kwargs)

        pretrained = mx.gluon.model_zoo.vision.mobilenet0_25(pretrained=True, ctx=context)
        fe = pretrained(mx.sym.var('data'))
        internals = fe.get_internals()
        fe_sym = internals['mobilenet0_relu22_fwd_output']

        with self.name_scope():
            self.backbone = mx.gluon.SymbolBlock(outputs=fe_sym, inputs=mx.sym.var('data'), params=pretrained.collect_params())

            self.class_predictor = mx.gluon.nn.Dense(2)
            self.class_predictor.collect_params().initialize(mx.init.Xavier(), ctx=context)

            self.gap = mx.gluon.nn.GlobalAvgPool2D()

    def hybrid_forward(self, F, x):
        features = self.backbone(x)
        out = self.class_predictor(self.gap(features))
        return F.sigmoid(out)


def train_model(train_data: list, test_data: list, batch_size: int=16,
                train_epoch_size: int=5000, test_epoch_size: int=2000,
                n_epochs: int=1000):
    context = mx.gpu(0)

    train_iter = DataIterator(train_data, context, batch_size=batch_size, n_processes=8,
                              epoch_size=train_epoch_size, augment=True)
    test_iter = DataIterator(test_data, context, batch_size=batch_size, n_processes=4,
                             epoch_size=test_epoch_size, augment=False)

    face_classification_net = ClassificationBlock(context)
    face_classification_net.hybridize()
    trainer = mx.gluon.Trainer(face_classification_net.collect_params(), 'adam')

    calc_log_loss = mx.gluon.loss.HuberLoss()

    for epoch in range(0, n_epochs):
        for (data, label) in train_iter:
            with mx.autograd.record():
                predictions = face_classification_net(data)
                loss = calc_log_loss(label, predictions)

            loss.backward()
            trainer.step(batch_size)

        smile_gt = []
        smile_pred = []
        mouth_pred = []
        mouth_gt = []
        for (data, label) in test_iter:
            predictions = face_classification_net(data).asnumpy()
            smile_pred += predictions[:, 0].tolist()
            mouth_pred += predictions[:, 1].tolist()
            label = label.asnumpy()
            smile_gt += label[:, 0].tolist()
            mouth_gt += label[:, 1].tolist()

        smile_auc = roc_auc_score(smile_gt, smile_pred)
        mouth_auc = roc_auc_score(mouth_gt, mouth_pred)
        print(epoch, loss.mean().asnumpy(), smile_auc, mouth_auc)
        face_classification_net.export('face_classification1', 0)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_data_dir', type=str, default='mouth_selfie')
    arg_parser.add_argument('--train_labels_path', type=str, default='labels_selfie.csv')
    arg_parser.add_argument('--test_data_dir', type=str, default='mouth_test')
    arg_parser.add_argument('--test_labels_path', type=str, default='labels_test.csv')
    args = arg_parser.parse_args()

    def match_labels(image_paths, labels_path):
        df = pd.read_csv(labels_path).values
        name_to_label = dict(zip(df[:, 0], df[:, 1:]))
        labels = []
        for p in image_paths:
            example_labels = name_to_label[os.path.basename(p)]
            labels.append((p, example_labels))
        return labels

    train_image_paths = glob.glob(args.train_data_dir + '/*.jpg')
    train_labels = match_labels(train_image_paths, args.train_labels_path)

    test_image_paths = glob.glob(args.test_data_dir + '/*.jpg')
    test_labels = match_labels(test_image_paths, args.test_labels_path)

    train_model(train_labels, test_labels)
