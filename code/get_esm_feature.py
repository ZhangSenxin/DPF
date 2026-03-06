#!/usr/bin/env python
# coding: utf-8

import os
import torch
import esm
import sys
import argparse
import zsx_some_tools as st
from typing import Generator, List, Tuple
import gzip
import subprocess
from datetime import datetime


def get_last_line_tail(file_path):
    result = subprocess.run(['tail', '-n', '1', file_path],
                          stdout=subprocess.PIPE)
    return result.stdout.decode().strip()


def batch_read_fasta(
        fasta_file: str,
        batch_size: int = 1000,
        gzipped: bool = False
) -> Generator[List[Tuple[str, str]], None, None]:
    """
    分批读取 FASTA 文件，每次返回 batch_size 条序列（header, sequence）。

    Args:
        fasta_file: FASTA 文件路径（支持普通文件或 .gz 压缩文件）。
        batch_size: 每批返回的序列数量（默认 1000）。
        gzipped: 是否处理 gzip 压缩文件（若文件以 .gz 结尾可自动检测，无需手动指定）。

    Yields:
        每批序列的列表，每个元素是 (header, sequence) 元组。
    """
    # 自动检测是否为 gzip 文件
    if fasta_file.endswith('.gz'):
        gzipped = True

    opener = gzip.open if gzipped else open
    batch = []
    current_header = None
    current_seq = []

    with opener(fasta_file, 'rt') as f:  # 'rt' 表示以文本模式读取
        for line in f:
            line = line.strip()
            if line.startswith('>'):  # 新序列头
                if current_header is not None:  # 保存前一个序列
                    batch.append((current_header, ''.join(current_seq)))
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                current_header = line[1:]  # 去掉 '>'
                current_seq = []
            else:
                current_seq.append(line)

        # 添加最后一个序列
        if current_header is not None:
            batch.append((current_header, ''.join(current_seq)))

    if batch:  # 返回剩余不足 batch_size 的序列
        yield batch


def main():
    usage = "usage: python %(prog)s -o order"
    description = "-o option is needed"
    parser = argparse.ArgumentParser(prog="%prog 1.0", description=description, usage=usage, add_help=False)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help message and exit.")

    parser.add_argument("-i", "--inputpath", dest="inputpath", type=str,
                        help="inputpath")

    parser.add_argument("-o", "--outputpath", dest="outputpath", type=str,
                        help="outputpath")

    parser.add_argument("-m", "--model_path", dest="model_path", type=str,
                        help="model_path")

    parser.add_argument("-n", "--model_name", dest="model_name", type=str, default='esm2_t33_650M_UR50D',
                        help="model_name")

    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32,
                        help="batch_size")

    parser.add_argument("--dimension", dest="dimension", type=int, default=0,
                        help="get which dimension, default is 0")

    parser.add_argument("--target_function", dest="target_function", type=str, default='mean',
                        help="target_function of output, default is mean, other is max")

    parser.add_argument("-l", "--max_length", dest="max_length", type=int, default=1000,
                        help="tokenizer dir-path")

    parser.add_argument("-f", "--feature_type", dest="feature_type", type=str, default='mean',
                        help="feature type with 'mean', 'max', and 'flat'")

    parser.add_argument("-d", "--device", dest="device", type=str, default='gpu',
                        help="cpu or gpu")

    parser.add_argument("-c", "--cuda_num", dest="cuda_num", type=str, default='0',
                        help="0 or 1")

    args = parser.parse_args()
    if not args.inputpath:
        parser.print_help()
        sys.exit(1)
    if not args.outputpath:
        parser.print_help()
        sys.exit(1)
    if not args.model_name:
        parser.print_help()
        sys.exit(1)

    fa_path = args.inputpath
    save_path1 = args.outputpath
    model_path = args.model_path
    model_name = args.model_name
    max_length = args.max_length
    batch_size = args.batch_size
    dimension = args.dimension
    target_function = args.target_function
    device_use = args.device
    cuda_num = args.cuda_num

    st.mkdir(save_path1)
    save_path1 = st.path_diagnosis(save_path1)
    name = fa_path.rsplit('/', 1)[1].split('.')[0]
    save_path = save_path1 + name + '_model_feature.txt'

    start = False
    is_Exist = os.path.exists(save_path)
    if is_Exist:
        last_index = get_last_line_tail(save_path).split('\t')[0]
    else:
        start = True

    device = torch.device('cuda:' + cuda_num) if device_use == 'gpu' else 'cpu'

    model_data = torch.load(model_path + model_name + '.pt')
    regression_data = torch.load(model_path + model_name + '-contact-regression.pt')
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)
    model.to(device)

    batch_converter = alphabet.get_batch_converter()
    model.eval()

    start_line = 0
    finish_line = 0
    w_count = 1
    for data in batch_read_fasta(fa_path, batch_size=batch_size):
        data = [da for da in data if len(da[1]) <= max_length]
        if len(data) == 0:
            continue
        batch_use = len(data)

        index_use = [da[0] for da in data]
        start_line += batch_size

        if not start:
            if sum([1 for i in index_use if i in last_index]):
                start = True
                print(start_line)
                with open(save_path1 + name + '_logging.txt', 'a+') as f:
                    f.write(str(start_line) + '\n')
            continue

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            try:
                results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
            except:
                print(data)
                raise ValueError('!!!')

        token_representations = results["representations"][33]
        token_representations = token_representations.cpu().detach().numpy()

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            if target_function == 'mean':
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(dimension))
            elif target_function == 'max':
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].max(dimension))

        with open(save_path, 'a+') as file:
            for i in range(batch_use):
                info = [index_use[i]] + list(sequence_representations[i])
                line = '\t'.join([str(info_info) for info_info in info])
                file.write(line + '\n')
                finish_line += 1

        if finish_line / 10000 > w_count:
            w_count += 1
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(save_path1 + name + '_count_logging.txt', 'a+') as f:
                f.write(str(start_line) + '\t' + formatted_time + '\n')

        del results, batch_tokens, token_representations  # 显式删除中间变量
        torch.cuda.empty_cache()  # 强制释放未使用的缓存


if __name__ == "__main__":
    main()
