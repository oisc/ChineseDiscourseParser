# coding: utf-8
from structure import EDU, Sentence, Relation
import numpy as np


def factorize_tree(tree, binarize=True):
    quads = set()   # (span, nuclear, coarse relation, fine relation)

    def factorize(root, offset=0):
        if isinstance(root, EDU):
            return [(offset, offset+len(root.text))]
        elif isinstance(root, Sentence):
            children_spans = []
            for child in root:
                spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
            return children_spans
        elif isinstance(root, Relation):
            children_spans = []
            for child in root:
                spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
            if binarize:
                while len(children_spans) >= 2:
                    right = children_spans.pop()
                    left = children_spans.pop()
                    quads.add(((left, right), root.nuclear, root.ctype, root.ftype))
                    children_spans.append((left[0], right[1]))
            else:
                quads.add((tuple(children_spans), root.nuclear, root.ctype, root.ftype))
            return [(children_spans[0][0], children_spans[-1][1])]

    factorize(tree.root_relation())
    return quads


def evaluation_trees(parses, golds, binarize=True, treewise_avearge=True):
    num_gold = np.zeros(len(golds))
    num_parse = np.zeros(len(parses))
    num_corr_span = np.zeros(len(parses))
    num_corr_nuc = np.zeros(len(parses))
    num_corr_ctype = np.zeros(len(parses))
    num_corr_ftype = np.zeros(len(parses))

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_factorized = factorize_tree(parse, binarize=binarize)
        gold_factorized = factorize_tree(gold, binarize=binarize)
        num_gold[i] = len(gold_factorized)
        num_parse[i] = len(parse_factorized)

    print(num_gold)
    print(num_parse)
