# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class INTERGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(INTERGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.ensemble_linear = nn.Linear(1, 2)
        self.layer_norm = nn.LayerNorm(768, eps=1e-6)
        self._norm_fact = 1 / math.sqrt(opt.bert_dim)
        self.w_q = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.w_k = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.w_v = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.fc_a = nn.Linear(opt.bert_dim, opt.bert_dim)

        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc4 = GraphConvolution(opt.bert_dim, opt.bert_dim)


        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1,self.opt.max_seq_len)):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def position(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        d = 5
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(text_len[i]):
                dis = abs(aspect_double_idx[i,0] - j)
                if dis == 0:
                    weight[i].append(1)
                if dis >= 1 and dis <= d:
                    weight[i].append(1 - dis / context_len)
                if dis > d:
                    weight[i].append(0)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).float().to(self.opt.device)
        return weight * x

    def MultiheadAttention(self, query, key, value):
        hid_dim = 768
        n_heads = 12
        dropout = 0.1
        mask = None
        assert hid_dim % n_heads == 0
        do = nn.Dropout(dropout)
        scale = math.sqrt(torch.FloatTensor([hid_dim // n_heads]))
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, n_heads, hid_dim //
                   n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, n_heads, hid_dim //
                   n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, n_heads, hid_dim //
                   n_heads).permute(0, 2, 1, 3)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention = do(torch.softmax(attention, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, n_heads * (hid_dim // n_heads))
        x = self.fc_a(x)
        return x, attention

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, self.opt.max_seq_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i,1]+1, self.opt.max_seq_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, aspect_indices, aspects_indices, left_indices, text_indices, adj= inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        encoder_layer_aspect, pooled_output = self.bert(aspect_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        encoder_layer_aspects, pooled_output = self.bert(aspects_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        text_out = encoder_layer
        aspect_out = encoder_layer_aspect
        aspects_out = encoder_layer_aspects

        # sentic-aspect
        x_s_1 = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # x_2 = F.relu(self.gc2(self.position_weight(x_1, aspect_double_idx, text_len, aspect_len), adj))
        # x_3 = F.relu(self.gc3(self.position_weight(x_2, aspect_double_idx, text_len, aspect_len), adj))
        x_s = F.relu(self.gc2(self.position_weight(x_s_1, aspect_double_idx, text_len, aspect_len), adj))

        # inter-aspect
        x_inter, atte = self.MultiheadAttention(aspects_out, aspects_out, aspects_out)
        x_i_1 = F.relu(self.gc1(self.position_weight(x_inter, aspect_double_idx, text_len, aspect_len), adj))
        x_inter = F.relu(self.gc2(self.position_weight(x_i_1, aspect_double_idx, text_len, aspect_len), adj))

        # atte = atte[0][0]
        # print(atte)
        # gate
        # x = x_s + att_out
        x_t = x_inter + x_s
        x_t = nn.LeakyReLU(inplace=True)(x_t)
        x = x_inter * x_t + x_s * (1 - x_t)

        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        print(alpha.shape)
        print(alpha)
        x = torch.matmul(alpha, text_out).squeeze(1)
        # x_pool = torch.max(x,dim=0).values

        output = self.fc(x)
        print(output)
        return output