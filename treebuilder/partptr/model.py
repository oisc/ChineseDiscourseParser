# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskedGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__()
        self.rnn = nn.GRU(batch_first=True, *args, **kwargs)
        self.hidden_size = self.rnn.hidden_size

    def forward(self, padded, lengths, initial_state=None):
        # [batch*edu]
        zero_mask = lengths != 0
        lengths[lengths == 0] += 1  # in case there is 0 length instance
        _, indices = lengths.sort(descending=True)
        _, rev_indices = indices.sort()

        # [batch*edu, max_word_seqlen, embedding]
        padded_sorted = padded[indices]
        lengths_sorted = lengths[indices]
        padded_packed = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True)
        outputs_sorted_packed, hidden_sorted = self.rnn(padded_packed, initial_state)
        # [batch*edu, max_word_seqlen, ]
        outputs_sorted, _ = pad_packed_sequence(outputs_sorted_packed, batch_first=True)
        # [batch*edu, max_word_seqlen, output_size]
        outputs = outputs_sorted[rev_indices]
        # [batch*edu, output_size]
        hidden = hidden_sorted.transpose(1, 0).contiguous().view(outputs.size(0), -1)[rev_indices]

        outputs = outputs * zero_mask.view(-1, 1, 1).float()
        hidden = hidden * zero_mask.view(-1, 1).float()
        return outputs, hidden


class BiGRUEDUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUEDUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = MaskedGRU(input_size, hidden_size, bidirectional=True)
        self.output_size = hidden_size * 2

    def forward(self, inputs, masks):
        # [batch_size*max_edu_seqlen]
        lengths = masks.sum(-1)
        outputs, hidden = self.rnn(inputs, lengths)
        return hidden


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.rnn = MaskedGRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, inputs, masks):
        inputs = inputs.transpose(-2, -1)
        splits = self.conv(inputs).transpose(-2, -1)
        masks = torch.cat([(masks.sum(-1, keepdim=True) > 0).type(masks.dtype), masks], dim=1)
        lengths = masks.sum(-1)
        outputs, hidden = self.rnn(splits, lengths)
        return outputs, masks, hidden


class PartitionPtr(nn.Module):
    def __init__(self, hidden_size, dropout,
                 word_vocab, pos_vocab, nuc_label,
                 pretrained=None, w2v_size=None, w2v_freeze=False, pos_size=30,
                 use_gpu=False):
        super(PartitionPtr, self).__init__()
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.nuc_label = nuc_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, dim=w2v_size, freeze=w2v_freeze, use_gpu=use_gpu)
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        # component
        self.edu_encoder = BiGRUEDUEncoder(self.w2v_size+self.pos_size, hidden_size)
        self.encoder = Encoder(self.edu_encoder.output_size, hidden_size, dropout)

    def forward(self, inputs):
        ...

    def encode_edus(self, e_inputs):
        e_input_words, e_input_poses, e_masks = e_inputs
        batch_size, max_edu_seqlen, max_word_seqlen = e_input_words.size()
        # [batch_size, max_edu_seqlen, max_word_seqlen, embedding]
        word_embedd = self.word_emb(e_input_words)
        pos_embedd = self.pos_emb(e_input_poses)
        concat_embedd = torch.cat([word_embedd, pos_embedd], dim=-1) * e_masks.unsqueeze(-1).float()
        # encode edu
        # [batch_size*max_edu_seqlen, max_word_seqlen, embedding]
        inputs = concat_embedd.view(batch_size*max_edu_seqlen, max_word_seqlen, -1)
        # [batch_size*max_edu_seqlen, max_word_seqlen]
        masks = e_masks.view(batch_size*max_edu_seqlen, max_word_seqlen)
        edu_encoded = self.edu_encoder(inputs, masks)
        # [batch_size, max_edu_seqlen, edu_encoder_output_size]
        edu_encoded = edu_encoded.view(batch_size, max_edu_seqlen, self.edu_encoder.output_size)
        e_masks = (e_masks.sum(-1) > 0).int()
        return edu_encoded, e_masks

    def loss(self, e_inputs, d_inputs, preds):
        e_inputs, e_masks = self.encode_edus(e_inputs)
        e_outputs, e_outputs_masks, e_contexts = self.encoder(e_inputs, e_masks)
