# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


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


class Decoder(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_dense = nn.Linear(inputs_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_size = hidden_size
        self.hx = None

    def forward(self, input, init_state):
        return self.run_step(input)

    def run_batch(self, inputs, init_states, masks):
        inputs = self.input_dense(inputs) * masks.unsqueeze(-1).float()
        outputs, _ = self.rnn(inputs, init_states.unsqueeze(0))
        outputs = outputs * masks.unsqueeze(-1).float()
        return outputs

    def reset(self):
        self.hx = None

    def run_step(self, input):
        ...


class BiLinearAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, hidden_size):
        super(BiLinearAttention, self).__init__()
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W = nn.Parameter(torch.empty(1, hidden_size, hidden_size, dtype=torch.float))
        nn.init.xavier_normal_(self.W)

    def forward(self, e_outputs, d_outputs, masks):
        e_outputs = self.e_mlp(e_outputs)
        d_outputs = self.d_mlp(d_outputs)
        attn = d_outputs.bmm(e_outputs.matmul(self.W).transpose(-2, -1))
        attn[masks == 0] = -1e8
        return attn


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
        self.context_dense = nn.Linear(self.encoder.output_size, hidden_size)
        self.decoder = Decoder(self.encoder.output_size*2, hidden_size)
        self.attention = BiLinearAttention(self.encoder.output_size, self.decoder.output_size, hidden_size)

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

    def _decode_batch(self, e_outputs, e_outputs_masks, e_contexts, d_inputs):
        d_inputs_indices, d_masks = d_inputs
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)

        d_init_states = self.context_dense(e_contexts)

        d_inputs = e_outputs[torch.arange(e_outputs.size(0)), d_inputs_indices.permute(2, 1, 0)].permute(2, 1, 0, 3)
        d_inputs = d_inputs.contiguous().view(d_inputs.size(0), d_inputs.size(1), -1)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()

        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks)
        return d_outputs, d_outputs_masks, d_masks

    def loss(self, e_inputs, d_inputs, grounds):
        e_inputs, e_masks = self.encode_edus(e_inputs)
        e_outputs, e_outputs_masks, e_contexts = self.encoder(e_inputs, e_masks)
        d_outputs, d_outputs_masks, d_masks = self._decode_batch(e_outputs, e_outputs_masks, e_contexts, d_inputs)

        attn = self.attention(e_outputs, d_outputs, d_masks)
        splits_predict = attn.log_softmax(dim=2)
        splits_ground, nucs_ground = grounds
        splits_ground = splits_ground.view(-1)
        splits_predict = splits_predict.view(splits_ground.size(0), -1)
        splits_masks = d_outputs_masks.view(-1).float()
        splits_loss = F.nll_loss(splits_predict, splits_ground, reduction="none")
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()

        loss = splits_loss
        return loss
