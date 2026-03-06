#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import zsx_some_tools as st


def main():
    usage = "usage: python %(prog)s -i input_path -o output_path"
    description = "-i -o option is needed"
    parser = argparse.ArgumentParser(prog="%prog 1.0", description=description, usage=usage, add_help=False)
    parser.add_argument("--input_path", type=str, default=None,
                        help='Path of predict result.')
    parser.add_argument("--save_path", type=str,
                        help='save summary result file path.')

    args = parser.parse_args()

    input_path = args.input_path
    save_path = args.save_path

    feature_folders = ['dlmyotCjDc_' + str(i) for i in range(6)]

    folders = ['catalytic_activity', 'DSD_binding', 'folding_stability', 'Immunogenicity', 'Photoactivity',
               'pH_stability', 'redox_activity', 'SSD_binding', 'thermal_stability']

    thresholds = np.arange(0.001, 1.001, 0.001)
    thresholds_add = np.array([0] + [np.round(0.1 ** i, i) for i in range(4, 19)][::-1])
    thresholds = np.concatenate([thresholds_add, thresholds])

    result = []
    for feature_folder in feature_folders:
        input_path_folder = input_path + feature_folder + '/'
        data_list = []
        for folder in folders:
            input_path_use = input_path_folder + 'result_' + folder + '.txt'
            data = st.read_file(input_path_use, header=None, index_col=0)
            data_list += [data]
            # 快速统计每个阈值以上的元素数量
            counts = np.sum(data.iloc[:, -1].values >= thresholds[:, None], axis=1)

            result += [[feature_folder, folder] + list(counts)]

        data_df = pd.concat(data_list, axis=1)
        max_data = np.max(data_df, axis=1)
        counts = np.sum(max_data.values >= thresholds[:, None], axis=1)

        result += [[feature_folder, 'max_score'] + list(counts)]

    result_df = pd.DataFrame(result, columns=['feature_folder', 'folder'] + list(thresholds))
    st.write_file(save_path, result_df, index=False)


if __name__ == '__main__':
    main()
