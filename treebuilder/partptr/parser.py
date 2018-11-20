# coding: UTF-8
from structure.nodes import Paragraph, Relation


class PartitionPtrParser:
    def __init__(self, model):
        self.model = model

    def parse(self, edus):
        # TODO implement beam search
        session = self.model.init_session(edus)
        while not session.terminate():
            score, state = self.model(session)
            split = score.argmax()
            session = session.forward(score, state, split)
        # build tree by splits (left, split, right)
        root_relation = self.build_tree(edus, session.splits[:])
        discourse = Paragraph([root_relation])
        return discourse

    def build_tree(self, edus, splits):
        left, split, right = splits.pop(0)
        if split - left == 1:
            left_node = edus[left]
        else:
            left_node = self.build_tree(edus, splits)

        if right - split == 1:
            right_node = edus[split]
        else:
            right_node = self.build_tree(edus, splits)

        relation = Relation([left_node, right_node])
        return relation
