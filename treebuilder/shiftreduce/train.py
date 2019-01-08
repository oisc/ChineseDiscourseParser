# coding: UTF-8
import logging
import argparse
import random
import torch
import numpy as np
from itertools import chain
from dataset import CDTB
from collections import Counter
from structure import EDU, Sentence, Relation
from structure.vocab import Label, Vocab


def _oracle(root):
    trans, children = [], []
    if isinstance(root, EDU):
        trans.append(("SHIFT", None, None))
        children.append(root)
    elif isinstance(root, Sentence):
        for node in root:
            _trans, _children = _oracle(node)
            trans.extend(_trans)
            children.extend(_children)
    elif isinstance(root, Relation):
        rel_children = []
        for node in root:
            _trans, _children = _oracle(node)
            trans.extend(_trans)
            rel_children.extend(_children)
        while len(rel_children) > 1:
            rel_children.pop()
            trans.append(("REDUCE", root.nuclear, root.ftype))
        children.append(root)
    else:
        raise ValueError("unhandle node type %s" % repr(type(root)))
    return trans, children


def oracle(tree):
    if tree.root_relation() is None:
        raise ValueError("Can not conduct transitions from forest")

    trans, _ = _oracle(tree.root_relation())
    return trans


def build_vocab(trees, trans):
    trans_label = Label("transition", Counter(chain(*trans)))

    words_counter = Counter()
    poses_counter = Counter()
    for tree in trees:
        edus = list(tree.edus())
        words = [getattr(edu, "words") for edu in edus]
        poses = [getattr(edu, "tags") for edu in edus]
        words_counter.update(chain(*words))
        poses_counter.update(chain(*poses))
    word_vocab = Vocab("word", words_counter)
    pos_vocab = Vocab("part of speech", poses_counter)
    return word_vocab, pos_vocab, trans_label


def numericalize(trees, word_vocab, pos_vocab):
    ...


def main(args):
    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)

    trees = [tree for tree in chain(*cdtb.train) if tree.root_relation() is not None]
    trans = [oracle(tree) for tree in trees]
    word_vocab, pos_vocab, trans_label = build_vocab(trees, trans)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")

    arg_parser.add_argument("--seed", default=21, type=int)
    main(arg_parser.parse_args())
