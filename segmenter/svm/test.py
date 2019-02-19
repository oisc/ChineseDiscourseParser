# coding: UTf-8
import logging
import pickle
from itertools import chain
from segmenter.svm import SVMSegmenter
from dataset import CDTB
from structure import node_type_filter, Sentence, TEXT


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with open("data/models/segmenter.svm.model", "rb") as model_fd:
        model = pickle.load(model_fd)
    segmenter = SVMSegmenter(model)
    cdtb = CDTB("data/CDTB", "TRAIN", "VALIDATE", "TEST", ctb_dir="data/CTB", preprocess=True, cache_dir="data/cache")
    ctb = cdtb.ctb

    gold = []
    seged = []
    for paragraph in chain(*cdtb.test):
        gold.append(paragraph)
        root = paragraph.root_relation()
        if root:
            text = "".join([t[0] for t in root.iterfind(node_type_filter(TEXT))])
            seged.append(segmenter.cut(text))
