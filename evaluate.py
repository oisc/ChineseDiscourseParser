# coding: UTF-8
import logging
import argparse
import pickle
from itertools import chain
import torch
from dataset import CDTB
from segmenter.svm import SVMSegmenter
from structure import node_type_filter, EDU, Paragraph, TEXT, Sentence
from treebuilder.partptr import PartPtrParser
from util import eval
from tqdm import tqdm

from util.eval import edu_eval, gen_edu_report

logger = logging.getLogger("evaluation")


def evaluate(args):
    with open("pub/models/segmenter.svm.model", "rb") as segmenter_fd:
        segmenter_model = pickle.load(segmenter_fd)
    with open("pub/models/treebuilder.partptr.model", "rb") as parser_fd:
        parser_model = torch.load(parser_fd, map_location="cpu")
        parser_model.use_gpu = False
        parser_model.eval()
    segmenter = SVMSegmenter(segmenter_model)
    parser = PartPtrParser(parser_model)

    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    golds = list(filter(lambda d: d.root_relation(), chain(*cdtb.test)))
    parses = []

    if args.use_gold_edu:
        logger.info("evaluation with gold edu segmentation")
    else:
        logger.info("evaluation with auto edu segmentation")

    for para in tqdm(golds, desc="parsing", unit=" para"):
        if args.use_gold_edu:
            edus = []
            for edu in para.edus():
                edu_copy = EDU([TEXT(edu.text)])
                setattr(edu_copy, "words", edu.words)
                setattr(edu_copy, "tags", edu.tags)
                edus.append(edu_copy)
            parse = parser.parse(Paragraph(edus))
            parses.append(parse)
        else:
            edus = []
            for sentence in para.sentences():
                if list(sentence.iterfind(node_type_filter(EDU))):
                    setattr(sentence, "parse", cdtb.ctb[sentence.sid])
                    edus.extend(segmenter.cut_edu(sentence))
            parse = parser.parse(Paragraph(edus))
            parses.append(parse)

    # edu score
    scores = edu_eval(golds, parses)
    logger.info("EDU segmentation scores:")
    logger.info(gen_edu_report(scores))

    # parser score
    cdtb_macro_scores = eval.parse_eval(parses, golds, average="macro")
    logger.info("CDTB macro (strict) scores:")
    logger.info(eval.gen_parse_report(*cdtb_macro_scores))

    # nuclear scores
    nuclear_scores = eval.nuclear_eval(parses, golds)
    logger.info("nuclear scores:")
    logger.info(eval.gen_category_report(nuclear_scores))

    # relation scores
    ctype_scores, ftype_scores = eval.relation_eval(parses, golds)
    logger.info("coarse relation scores:")
    logger.info(eval.gen_category_report(ctype_scores))
    logger.info("fine relation scores:")
    logger.info(eval.gen_category_report(ftype_scores))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")
    arg_parser.add_argument("--use_gold_edu", dest="use_gold_edu", action="store_true")
    arg_parser.set_defaults(use_gold_edu=False)
    evaluate(arg_parser.parse_args())
