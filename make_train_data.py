"""
Preprocess
- encode property change
- build vocabulary
- split data into train, validation and test
"""
import os
import argparse
import pickle
import re
import math
from glob import glob

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce
import pandas as pd
from pathlib import Path
from const import seq_interval, bool_interval
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import time
# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")

def encode_seq(value) -> str:
    # 遍历每个区间字符串
    for interval in seq_interval:
        # 使用正则表达式提取边界
        bounds = re.findall(r'[\(\[]([^,]+),\s*([^,\)\]]+)[\)\]]', interval)
        if bounds:
            lower, upper = bounds[0]

            # 处理无穷大
            if lower == '-inf':
                lower = -math.inf
            else:
                lower = float(lower)

            if upper == 'inf':
                upper = math.inf
            else:
                upper = float(upper)

            # 检查数值是否属于当前区间
            if (lower < value <= upper) or (math.isclose(value, lower, rel_tol=1e-9) and '(' not in interval[0]):
                return interval
    return 'error'

def encode_bool(value) -> str:
    if value == 1:
        return bool_interval[0]
    elif value == -1:
        return bool_interval[1]
    elif value == 0:
        return bool_interval[2]
    return 'error'



def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=False)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()

def merge_csv_file(csv_files, output_file='./merged.csv', chunksize=100000):
    # 初始化一个标志位，用于在第一个文件的第一个块写入时包含头部
    first_chunk = True
    with open(output_file, 'w', newline='') as fout:
        total = len(csv_files)
        for idx, file in enumerate(csv_files):
            start = time.time()
            # 使用 chunksize 参数分批次读取每个 CSV 文件
            for idx2, chunk in enumerate(pd.read_csv(file, chunksize=chunksize)):
                # 将数据流式写入到输出文件，只在第一个块时写入头部
                chunk.to_csv(fout, header=first_chunk, index=False, mode='a')
                
                # 确保后续写入不再包括头部
                if first_chunk:
                    first_chunk = False
            end = time.time()
            print(f"===({idx}/{total})write file {end - start}s | {file}")

SEED = 42

def gen_train_data(file_path):
    dfInput=pd.read_csv(file_path)
    # 小于5的样本丢弃，不好区分测试集和验证集
    if len(dfInput) < 5:
        return
    dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
    dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES','Value_Diff', 'main_cls', 'minor_cls', 'value_type', 'target_name', 'sequence']]
    dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES','Delta_Value', 'main_cls', 'minor_cls', 'value_type', 'target_name', 'sequence']
    newPath=Path(file_path).parent.joinpath("train_valid_test_full.csv")   ## will be saved
    dfInput.to_csv(newPath, index=None)
    # args.input_data_path=newPath.as_posix()

    # 将数值转换成编码区间
    data = dfInput
    value_type = data['value_type'].iloc[0]
    # 判断是连续值还是bool值
    if value_type == 'seq':
        data['Delta_Value'] = data['Delta_Value'].apply(encode_seq)
    elif value_type == 'bool':
        data['Delta_Value'] = data['Delta_Value'].apply(encode_bool)
    
    # save encodeed file
    output_file = file_path.split('.csv')[0] + '_encoded.csv'
    LOG.info("Saving encoded property change to file: {}".format(output_file))
    data.to_csv(output_file, index=False)

    # split data
    train, test = train_test_split(
        data, test_size=(1-SPLIT_RATIO)/2, random_state=SEED)
    train, validation = train_test_split(train, test_size=(1-SPLIT_RATIO)/2, random_state=SEED)
    LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

    parent = uf.get_parent_dir(file_path)
    train.to_csv(os.path.join(parent, "train.csv"), index=False)
    validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
    test.to_csv(os.path.join(parent, "test.csv"), index=False)

def task(file, idx, total):
    LOG.info(f"\n===({idx}/{total}) handling {file}")
    gen_train_data(file)
    print(f"\n===({idx}/{total}) finished {file}")

if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()
    SPLIT_RATIO = args.train_ratio

    root = '/home/yichao/zhilian/GenAICode/new_paper_code/mmp_finished/*'
    csvFiles = glob(f"{root}/*_MMP.csv")
    start = time.time()
    p = Pool(int(cpu_count()/2))
    listLen = len(csvFiles)
    for idx, file in enumerate(csvFiles):
        p.apply_async(task, args=(file, idx + 1, listLen))
    p.close()
    p.join()
    end = time.time()
    print("encode文件总共用时{}秒".format((end - start)))
    
    # merge train data
    trainFiles = glob(f"{root}/train.csv")
    merge_csv_file(trainFiles, './train.csv')

    # merge validation data
    valFiles = glob(f"{root}/validation.csv")
    merge_csv_file(valFiles, './validation.csv')

    # merge test data
    testFiles = glob(f"{root}/test.csv")
    merge_csv_file(testFiles, './test.csv')
