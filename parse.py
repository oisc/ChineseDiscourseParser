# coding: UTF-8
import argparse
import pickle
import torch
from segmenter.svm import SVMSegmenter
from structure import Discourse
from treebuilder.partptr import PartPtrParser
import logging
import threading
import tqdm


main_logger = logging.getLogger("dp")


def build_pipeline():
    logger = logging.getLogger("parsing thread %s" % threading.current_thread().name)
    logger.info("initialize pipeline")
    with open("pub/models/segmenter.svm.model", "rb") as segmenter_fd:
        segmenter_model = pickle.load(segmenter_fd)
        segmenter = SVMSegmenter(segmenter_model)
    with open("pub/models/treebuilder.partptr.model", "rb") as parser_fd:
        parser_model = torch.load(parser_fd, map_location="cpu")
        parser_model.use_gpu = False
        parser_model.eval()
        parser = PartPtrParser(parser_model)

    def _pipeline(text):
        seged = segmenter.cut(text)
        parsed = parser.parse(seged)
        return parsed

    return _pipeline


def run(args):
    disc = Discourse()
    pipeline = build_pipeline()
    with open(args.source, "r", encoding=args.encoding) as source_fd:
        for line in tqdm.tqdm(source_fd, desc="parsing %s" % args.source, unit=" para"):
            line = line.strip()
            if line:
                para = pipeline(line)
                if args.draw:
                    para.draw()
                disc.append(para)
    main_logger.info("save parsing to %s" % args.save)
    disc.to_xml(args.save, encoding=args.encoding)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("source")
    arg_parser.add_argument("save")
    arg_parser.add_argument("--encoding", default="utf-8")
    arg_parser.add_argument("--draw", dest="draw", action="store_true")
    arg_parser.set_defaults(draw=False)
    run(arg_parser.parse_args())
