import torch
from models.layers.transformer_layers import FeedForward
from models.layers.generic_layers import EmbeddingLayer
from models.layers.transformer_layers import MultiHeadAttention
from utils.masking import create_pad_mask, create_no_peak_and_pad_mask
from models.captioning_model import CaptioningModel

import torch.nn.functional as F

import numpy as np
import torch.nn as nn


class StaticExpansionBlock(nn.Module):
    def __init__(self, d_model, num_exp, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model
        self.num_exp = num_exp
        self.query_exp_vectors = nn.Embedding(num_exp, d_model)
        self.bias_exp_vectors = nn.Embedding(num_exp, d_model)
        self.key_embed = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_b_embed = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)

        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, mask):
        bs, enc_len, _ = x.shape

        query_exp = self.query_exp_vectors(n_indexes)
        bias_exp = self.bias_exp_vectors(n_indexes)
        x_key = self.key_embed(x)

        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / np.sqrt(self.d_model)
        z = self.Z_dropout(z)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mask == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mask == 0, 0.0)
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)

        class_a = torch.matmul(class_a_fw, self.class_a_embed(x)) + bias_exp
        class_b = torch.matmul(class_b_fw, self.class_b_embed(x)) + bias_exp
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))
        class_a_bw = class_a_bw / (class_a_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_bw = class_b_bw / (class_b_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_bw, class_a)
        class_b = torch.matmul(class_b_bw, class_b)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b
        return x_result


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_exp, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

        self.stc_exp = StaticExpansionBlock(d_model, num_exp, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.stc_exp(x=x2, n_indexes=n_indexes, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DynamicExpansionBlock(nn.Module):
    def __init__(self, d_model, num_exp, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model

        self.num_exp = num_exp
        self.cond_embed = nn.Linear(d_model, d_model)

        self.query_exp_vectors = nn.Embedding(self.num_exp, d_model)
        self.bias_exp_vectors = nn.Embedding(self.num_exp, d_model)

        self.key_linear = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_b_embed = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)
        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, mask):
        bs, dec_len, _ = x.shape

        cond = self.cond_embed(x).view(bs, dec_len, 1, self.d_model)
        query_exp = self.query_exp_vectors(n_indexes).unsqueeze(1)
        bias_exp = self.bias_exp_vectors(n_indexes).unsqueeze(1)
        query_exp = (query_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)
        bias_exp = (bias_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)

        x_key = self.key_linear(x)
        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / np.sqrt(self.d_model)
        z = self.Z_dropout(z)

        mod_mask_1 = mask.unsqueeze(2).expand(bs, dec_len, self.num_exp, dec_len).contiguous(). \
            view(bs, dec_len * self.num_exp, dec_len)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_fw, self.class_a_embed(x))
        class_b = torch.matmul(class_b_fw, self.class_b_embed(x))
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        mod_mask_2 = mask.unsqueeze(-1).expand(bs, dec_len, dec_len, self.num_exp).contiguous(). \
            view(bs, dec_len, dec_len * self.num_exp)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))
        class_a_bw = class_a_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_b_bw = class_b_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_a_bw = class_a_bw / (class_a_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_bw = class_b_bw / (class_b_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_bw, class_a + bias_exp)
        class_b = torch.matmul(class_b_bw, class_b + bias_exp)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_exp, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.dyn_exp = DynamicExpansionBlock(d_model, num_exp, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, cross_connection_x, input_attention_mask, cross_attention_mask):

        # Pre-LayerNorm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.dyn_exp(x=x2, n_indexes=n_indexes, mask=input_attention_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha(q=x2, k=cross_connection_x, v=cross_connection_x,
                                        mask=cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class ExpansionNet(CaptioningModel):
    def __init__(self, d_model, N_enc, N_dec, ff, num_heads, num_exp_enc, num_exp_dec,
                 output_word2idx, max_seq_len, drop_args, rank=0):
        super().__init__()
        self.output_word2idx = output_word2idx
        self.max_seq_len = max_seq_len

        self.num_exp_dec = num_exp_dec
        self.num_exp_enc = num_exp_enc

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.d_model = d_model

        self.encoders = nn.ModuleList([EncoderLayer(d_model, ff, num_exp_enc, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, ff, num_exp_dec, drop_args.dec) for _ in range(N_dec)])

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(2048, d_model)
        self.vocab_linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_enc_dropout = nn.Dropout(drop_args.other)
        self.out_dec_dropout = nn.Dropout(drop_args.other)

        self.out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        self.enc_reduce_group = nn.Linear(d_model * self.N_enc, d_model)
        self.enc_reduce_norm = nn.LayerNorm(d_model)
        self.dec_reduce_group = nn.Linear(d_model * self.N_dec, d_model)
        self.dec_reduce_norm = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.trained_steps = 0
        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads):
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), self.num_exp_enc, enc_input.size(1)),
                                   pad_along_row_input=[0] * enc_input.size(0),
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        x = self.input_embedder_dropout(self.input_linear(enc_input))
        pos_x = torch.arange(self.num_exp_enc).unsqueeze(0).expand(enc_input.size(0), self.num_exp_enc).to(self.rank)
        x_list = []
        for i in range(self.N_enc):
            x = self.encoders[i](x=x, n_indexes=pos_x, mask=pad_mask)
            x_list.append(x)
        x_list = torch.cat(x_list, dim=-1)
        x = x + self.out_enc_dropout(self.enc_reduce_group(x_list))
        x = self.enc_reduce_norm(x) 
        return x

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=False):

        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
                                mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
                                num_pads=dec_input_num_pads,
                                rank=self.rank)

        pad_mask = create_pad_mask(mask_size=(dec_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_along_row_input=dec_input_num_pads,
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        y = self.out_embedder(dec_input)
        pos_x = torch.arange(self.num_exp_dec).unsqueeze(0).expand(dec_input.size(0), self.num_exp_dec).to(self.rank)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1)).to(self.rank)
        y = y + self.pos_encoder(pos_y)
        y_list = []
        for i in range(self.N_dec):
            y = self.decoders[i](x=y,
                                 n_indexes=pos_x,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask)
            y_list.append(y)
        y_list = torch.cat(y_list, dim=-1)
        y = y + self.out_dec_dropout(self.dec_reduce_group(y_list))
        y = self.dec_reduce_norm(y)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y
