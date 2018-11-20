# coding: UTF-8
import argparse
import logging
import random
import torch
import numpy as np
from dataset import CDTB
from collections import Counter
from itertools import chain
from structure.vocab import Vocab, Label
from structure.nodes import node_type_filter, Paragraph, EDU, Relation, Sentence, TEXT
from treebuilder.partptr.model import PartitionPtr
from treebuilder.partptr.parser import PartitionPtrParser
import torch.optim as optim


def build_vocab(dataset):
    word_freq = Counter()
    pos_freq = Counter()
    nuc_freq = Counter()
    for paragraph in chain(*dataset):
        for node in paragraph.iterfind(filter=node_type_filter([EDU, Relation])):
            if isinstance(node, EDU):
                word_freq.update(node.words)
                pos_freq.update(node.tags)
            elif isinstance(node, Relation):
                nuc_freq[node.nuclear] += 1

    word_vocab = Vocab("word", word_freq)
    pos_vocab = Vocab("part of speech", pos_freq)
    nuc_label = Label("nuclear", nuc_freq)
    return word_vocab, pos_vocab, nuc_label


def gen_decoder_data(root, edu2ids):
    # splits    s0  s1  s2  s3  s4  s5  s6
    # edus    s/  e0  e1  e2  e3  e4  e5  /s
    splits = []  # [(0, 3, 6, NS), (0, 2, 3, SN), ...]
    child_edus = []  # [edus]

    if isinstance(root, EDU):
        child_edus.append(root)
    elif isinstance(root, Sentence):
        for child in root:
            _child_edus, _splits = gen_decoder_data(child, edu2ids)
            child_edus.extend(_child_edus)
            splits.extend(_splits)
    elif isinstance(root, Relation):
        children = [gen_decoder_data(child, edu2ids) for child in root]
        if len(children) < 2:
            raise ValueError("relation node should at least 2 children")

        while children:
            left_child_edus, left_child_splits = children.pop(0)
            if children:
                last_child_edus, _ = children[-1]
                start = edu2ids[left_child_edus[0]]
                split = edu2ids[left_child_edus[-1]] + 1
                end = edu2ids[last_child_edus[-1]] + 1
                nuc = root.nuclear
                splits.append((start, split, end, nuc))
            child_edus.extend(left_child_edus)
            splits.extend(left_child_splits)
    return child_edus, splits


def numericalize(dataset, word_vocab, pos_vocab, nuc_label):
    instances = []
    for paragraph in filter(lambda d: d.root_relation(), chain(*dataset)):
        encoder_inputs = []
        decoder_inputs = []
        pred_splits = []
        pred_nucs = []
        edus = list(paragraph.edus())
        for edu in edus:
            edu_word_ids = [word_vocab[word] for word in edu.words]
            edu_pos_ids = [pos_vocab[pos] for pos in edu.tags]
            encoder_inputs.append((edu_word_ids, edu_pos_ids))
        edu2ids = {edu: i for i, edu in enumerate(edus)}
        _, splits = gen_decoder_data(paragraph.root_relation(), edu2ids)
        for start, split, end, nuc in splits:
            decoder_inputs.append((start, end))
            pred_splits.append(split)
            pred_nucs.append(nuc_label[nuc])
        instances.append((encoder_inputs, decoder_inputs, pred_splits, pred_nucs))
    return instances


def gen_batch_iter(instances, batch_size, use_gpu=False):
    random_instances = np.random.permutation(instances)
    num_instances = len(instances)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset+batch_size)]

        # find out max seqlen of edus and words of edus
        num_batch = batch.shape[0]
        max_edu_seqlen = 0
        max_word_seqlen = 0
        for encoder_inputs, decoder_inputs, pred_splits, pred_nucs in batch:
            max_edu_seqlen = max_edu_seqlen if max_edu_seqlen >= len(encoder_inputs) else len(encoder_inputs)
            for edu_word_ids, edu_pos_ids in encoder_inputs:
                max_word_seqlen = max_word_seqlen if max_word_seqlen >= len(edu_word_ids) else len(edu_word_ids)

        # batch to numpy
        e_input_words = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_input_poses = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_masks = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.uint8)

        d_inputs = np.zeros([num_batch, max_edu_seqlen-1, 2], dtype=np.long)
        d_outputs = np.zeros([num_batch, max_edu_seqlen-1], dtype=np.long)
        d_output_nucs = np.zeros([num_batch, max_edu_seqlen-1], dtype=np.long)
        d_masks = np.zeros([num_batch, max_edu_seqlen-1, max_edu_seqlen+1], dtype=np.uint8)

        for batchi, (encoder_inputs, decoder_inputs, pred_splits, pred_nucs) in enumerate(batch):
            for edui, (edu_word_ids, edu_pos_ids) in enumerate(encoder_inputs):
                word_seqlen = len(edu_word_ids)
                e_input_words[batchi][edui][:word_seqlen] = edu_word_ids
                e_input_poses[batchi][edui][:word_seqlen] = edu_pos_ids
                e_masks[batchi][edui][:word_seqlen] = 1

            for di, decoder_input in enumerate(decoder_inputs):
                d_inputs[batchi][di] = decoder_input
                d_masks[batchi][di][decoder_input[0]+1: decoder_input[1]] = 1
            d_outputs[batchi][:len(pred_splits)] = pred_splits
            d_output_nucs[batchi][:len(pred_nucs)] = pred_nucs

        # numpy to torch
        e_input_words = torch.from_numpy(e_input_words).long()
        e_input_poses = torch.from_numpy(e_input_poses).long()
        e_masks = torch.from_numpy(e_masks).byte()
        d_inputs = torch.from_numpy(d_inputs).long()
        d_outputs = torch.from_numpy(d_outputs).long()
        d_output_nucs = torch.from_numpy(d_output_nucs).long()
        d_masks = torch.from_numpy(d_masks).byte()

        if use_gpu:
            e_input_words = e_input_words.cuda()
            e_input_poses = e_input_poses.cuda()
            e_masks = e_masks.cuda()
            d_inputs = d_inputs.cuda()
            d_outputs = d_outputs.cuda()
            d_masks = d_masks.cuda()

        yield (e_input_words, e_input_poses, e_masks), (d_inputs, d_masks), (d_outputs, d_output_nucs)
        offset = offset + batch_size


def parse_and_eval(dataset, model):
    model.eval()
    parser = PartitionPtrParser(model)

    grounded = list(filter(lambda d: d.root_relation(), chain(*dataset)))
    stripped = []
    for paragraph in grounded:
        edus = []
        for edu in paragraph.edus():
            edu_copy = EDU(TEXT(edu.text))
            setattr(edu_copy, "words", edu.words)
            setattr(edu_copy, "tags", edu.tags)
            edus.append(edu_copy)
        stripped.append(edus)
    parsed = []
    for edus in stripped:
        parse = parser.parse(edus)
        parsed.append(parse)
    return ...


def main(args):
    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    # build vocabulary
    word_vocab, pos_vocab, nuc_label = build_vocab(cdtb.train)
    # trainset, validateset, testest = (numericalize(dataset, word_vocab, pos_vocab, nuc_label)
    #                                   for dataset in [cdtb.train, cdtb.validate, cdtb.test])
    trainset = numericalize(cdtb.train, word_vocab, pos_vocab, nuc_label)
    logging.info("num of instances trainset: %d" % len(trainset))
    # build model
    model = PartitionPtr(hidden_size=args.hidden_size, dropout=args.dropout,
                         word_vocab=word_vocab, pos_vocab=pos_vocab, nuc_label=nuc_label, pos_size=args.pos_size,
                         pretrained=args.pretrained, w2v_size=args.w2v_size, w2v_freeze=args.w2v_freeze,
                         use_gpu=args.use_gpu)
    if args.use_gpu:
        model.cuda()
    logging.info("model:\n%s" % str(model))

    # train and evaluate
    niter = 0
    log_loss = 0.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for nepoch in range(1, args.epoch + 1):
        batch_iter = gen_batch_iter(trainset, args.batch_size, args.use_gpu)
        for nbatch, (e_inputs, d_inputs, grounds) in enumerate(batch_iter, start=1):
            niter += 1
            model.train()
            optimizer.zero_grad()
            loss = model.loss(e_inputs, d_inputs, grounds)
            loss.backward()
            optimizer.step()
            log_loss += loss.item()
            if niter % args.log_every == 0:
                logging.info("[iter %-6d]epoch: %-3d, batch %-5d, train loss: %.5f" % (niter, nepoch, nbatch, log_loss))
                log_loss = 0.
            if niter % args.validate_every == 0:
                scores = parse_and_eval(cdtb.validate, model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")

    # model parameters
    arg_parser.add_argument("-hidden_size", default=128, type=int)
    arg_parser.add_argument("-dropout", default=0.1, type=float)
    w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    w2v_group.add_argument("-pretrained")
    w2v_group.add_argument("-w2v_size", type=int)
    arg_parser.add_argument("-w2v_freeze", type=bool, default=False)
    arg_parser.add_argument("-pos_size", default=30, type=int)

    # model parameters
    arg_parser.set_defaults(use_gpu=False)
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-batch_size", default=32, type=int)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-log_every", default=10, type=int)
    arg_parser.add_argument("-validate_every", default=200, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.add_argument("--seed", default=21, type=int)

    main(arg_parser.parse_args())
