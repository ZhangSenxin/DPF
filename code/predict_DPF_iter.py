#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import zsx_some_tools as st

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def feature_reader(feature_path, batch_size, start_line=0, end_line=None):
    value_list = []
    index_list = []
    with open(feature_path, 'r') as feature_file:
        for idx, line in enumerate(feature_file):
            if not idx >= start_line:
                continue

            if end_line is not None:
                if idx >= end_line:
                    break

            line = line.strip().split('\t')
            index_list += [line[0]]
            value_list += [line[1:]]

            if len(value_list) == batch_size:
                value_list = np.array(value_list).astype(np.float32)
                yield index_list, value_list
                value_list = []
                index_list = []

        if len(value_list) > 0:
            value_list = np.array(value_list).astype(np.float32)
            yield index_list, value_list


class BernoulliStraightThrough(torch.autograd.Function):
    """伯努利采样的直通估计器"""

    @staticmethod
    def forward(ctx, probs):
        # 前向：离散化采样
        sample = torch.bernoulli(probs)
        ctx.save_for_backward(probs, sample)
        return sample

    @staticmethod
    def backward(ctx, grad_output):
        probs, sample = ctx.saved_tensors
        # 直通估计器：梯度直接传回probs，忽略采样操作
        grad_probs = grad_output.clone()
        return grad_probs


class BernoulliMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=None, batch_first=None):
        super(BernoulliMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x1, x2):
        batch_size, seq_length, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)

        # Apply sigmoid instead of softmax
        attn_probs = torch.sigmoid(attn_scores)

        # wrong
        #         attn_probs = torch.bernoulli(attn_probs)  # 采样 {0,1}
        # STE
        attn_probs = BernoulliStraightThrough.apply(attn_probs)

        # Apply attention to V
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_length, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)

        return self.out_proj(attn_output), None


class CSA2(nn.Module):
    def __init__(self, in_dim=1280, h_dim=100, out_dim=2, dropout=0., num_heads=4):
        super(CSA2, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # 线性投影层，用于将输入投影到 Self-Attention 所需的维度
        self.fc_input = nn.Linear(in_features=in_dim, out_features=h_dim, bias=True)

        # 第一层 多头自注意力 + LayerNorm
        self.multihead_attn_1 = BernoulliMultiHeadAttention(embed_dim=h_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(h_dim)  # 第一层 LayerNorm

        # 第二层 多头自注意力 + LayerNorm
        self.multihead_attn_2 = BernoulliMultiHeadAttention(embed_dim=h_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(h_dim)  # 第二层 LayerNorm

        # 前馈神经网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
        )

        # 分类头
        self.fc_output = nn.Linear(h_dim, out_dim)

        # 其他组件
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 线性投影
        x = self.fc_input(x)  # (batch_size, h_dim)
        x = x.unsqueeze(1)    # (batch_size, seq_len=1, h_dim)

        # 第一层 MultiheadAttention + LayerNorm + 残差连接
        attn_output1, _ = self.multihead_attn_1(x, x, x)  # (batch_size, seq_len, h_dim)
        x = self.norm1(x + attn_output1)  # 残差连接 + 归一化

        # 第二层 MultiheadAttention + LayerNorm + 残差连接
        attn_output2, _ = self.multihead_attn_2(x, x, x)  # (batch_size, seq_len, h_dim)
        x = self.norm2(x + attn_output2)  # 残差连接 + 归一化

        return x


def accuracy(output, target):
    return torch.mean((torch.argmax(output, dim=1) == target).float())


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)


class MixtureOfExperts(nn.Module):
    def __init__(self, Input_expert, input_dim, hidden_dim, output_dim, num_experts, expert_hidden_dim,
                 out_dim, dropout):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # 创建多个专家
        self.experts = nn.ModuleList([Input_expert(input_dim, expert_hidden_dim, output_dim) for _ in range(num_experts)])

        # 创建门控网络
        self.gating_network = GatingNetwork(input_dim, hidden_dim, num_experts)

        # 前馈神经网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
        )

        # 分类头
        self.fc_output = nn.Linear(expert_hidden_dim, out_dim)

        # 其他组件
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 门控网络计算每个专家的权重
        gating_weights = self.gating_network(x)

        # 每个专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        # 加权平均专家的输出
        gating_weights = torch.softmax(gating_weights.unsqueeze(1), dim=-1)  # 对权重进行 softmax 归一化

        # 加权求和
        x = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=-1)

        # 去掉时间维度
        x = x.squeeze(1)  # (batch_size, h_dim)

        # 前馈神经网络
        x = self.ffn(x)  # (batch_size, h_dim)
        x = self.relu(x)
        x = self.dropout(x)

        # 分类层
        x = self.fc_output(x)  # (batch_size, out_dim)
        x = self.softmax(x)  # 归一化为概率

        return x


def evaluate(data_loader, model, criterion, device):
    model.eval()

    output_all, target_all = None, None
    for idx, (feature, target) in enumerate(data_loader):
        feature = feature.to(device)
        target = target.to(device)
        output = model(feature)
        if idx == 0:
            output_all = output
            target_all = target
        else:
            output_all = torch.cat([output_all, output], 0)
            target_all = torch.cat([target_all, target], 0)

    loss = criterion(output_all, target_all)
    acc1 = accuracy(output_all, target_all)

    return loss.item(), acc1.item(), output_all, target_all


def main():
    usage = "usage: python %(prog)s -i input_path -o output_path"
    description = "-i -o option is needed"
    parser = argparse.ArgumentParser(prog="%prog 1.0", description=description, usage=usage, add_help=False)
    parser.add_argument("--sequence_path", type=str, default=None,
                        help='Path of data for sequence.')
    parser.add_argument("--ckpt_path", type=str,
                        help='predict model to load.')
    parser.add_argument("--save_path", type=str,
                        help='save predict result path.')

    parser.add_argument("--seed", type=int, default=42,
                        help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='Number of batch size.')
    parser.add_argument("--device", type=str, default='cuda:0',
                        help='Device to use.')

    args = parser.parse_args()

    if not args.sequence_path:
        parser.print_help()
        sys.exit(1)
    if not args.ckpt_path:
        parser.print_help()
        sys.exit(1)
    if not args.save_path:
        parser.print_help()
        sys.exit(1)

    seed = args.seed
    sequence_path = args.sequence_path
    ckpt_path = args.ckpt_path
    save_path = args.save_path
    batch_size = args.batch_size

    feature_folders = ['dlmyotCjDc_' + str(i) for i in range(6)]
    folders = ['catalytic_activity', 'DSD_binding', 'folding_stability', 'Immunogenicity', 'Photoactivity',
               'pH_stability', 'redox_activity', 'SSD_binding', 'thermal_stability']

    set_seed(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device(args.device)

    # 模型参数
    best_params = {'learning_rate': 0.00010636954006135154,
                   'weight_decay': 1.0649236744357727e-06,
                   'num_experts': 9}
    dropout = 0.2
    input_dim = 1280  # 输入维度
    hidden_dim = 256  # 门控网络的隐藏层维度
    output_dim = 2  # 输出维度（二分类任务，使用sigmoid输出）
    num_experts = best_params['num_experts']  # 专家的数量
    expert_hidden_dim = 100  # 专家网络的隐藏层维度

    for feature_folder in feature_folders:
        feature_path_use = sequence_path + feature_folder + '/' + feature_folder + '_model_feature.txt'
        save_path_folder = save_path + feature_folder + '/'
        st.mkdir(save_path_folder)

        end_line = st.wc_py(feature_path_use)

        for folder in folders:
            save_path_use = save_path_folder + 'result_' + folder + '.txt'
            is_Exist = os.path.exists(save_path_use)
            start_line = st.wc_py(save_path_use) if is_Exist else 0

            # 创建 Mixture of Experts 模型
            set_seed(seed)
            model_eval = MixtureOfExperts(CSA2, input_dim, hidden_dim, output_dim, num_experts, expert_hidden_dim, output_dim, dropout=dropout)
            model_eval = model_eval.to(device)

            # load model
            set_seed(seed)
            ckpt_file_use = ckpt_path + folder + '/MOE_' + folder + '-best_model.pth'
            checkpoint = torch.load(ckpt_file_use, weights_only=False)
            pa_dict = checkpoint['model_state_dict']
            model_eval.load_state_dict(pa_dict)

            # predict
            set_seed(seed)
            model_eval.eval()

            set_seed(seed)
            for index_info, data_x in feature_reader(feature_path_use, batch_size,
                                                     start_line=start_line, end_line=end_line):
                data_x = torch.tensor(data_x, dtype=torch.float32)
                output = model_eval(data_x.to(device))
                output = output.cpu().detach().numpy()

                string = '\n'.join(
                    ['\t'.join(
                        [ind, str(value[1])]
                    ) for ind, value in zip(index_info, output)]
                )

                with open(save_path_use, 'a+') as file:
                    file.write(string + '\n')


if __name__ == '__main__':
    main()
