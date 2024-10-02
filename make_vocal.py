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

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce
import pandas as pd
from pathlib import Path
from glob import glob
import generate as gn
from const import seq_interval, bool_interval
from common.utils import Data_Type

# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")

def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=False)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()



if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()

    def record_vocal(vocabulary, file, data_type=Data_Type.base):
        dfInput=pd.read_csv(file)
        if len(dfInput) < 1:
            return
        LOG.info("===finish reading")
        # add property name before property change; save to file
        property_condition = []
        # 添加main_cls
        property_condition.append(dfInput['main_cls'].iloc[0])
        # 添加minor_cls
        property_condition.append(dfInput['minor_cls'].iloc[0])
        # 添加额外信息
        extra_list = []
        if data_type == Data_Type.base:
            extra_list = []
        elif data_type == Data_Type.target_name:
            extra_list = list(dfInput['target_name'].iloc[0])
        elif data_type == Data_Type.seq:
            seq = dfInput['sequence'].iloc[0]
            extra_list = list(seq if isinstance(seq, str) else '')
        elif data_type == Data_Type.all:
            seq = dfInput['sequence'].iloc[0]
            extra_list = list(dfInput['target_name'].iloc[0])
            extra_list.extend(list(seq if isinstance(seq, str) else ''))
        
        dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
        dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES']]
        dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES']
        
        # # 将数值转换成编码区间
        LOG.info("Building vocabulary")
        tokenizer = mv.SMILESTokenizer()
        # 获取所有SMILES分子式列表
        smiles_list = pd.unique(dfInput[['constantSMILES', 'fromVarSMILES', 'toVarSMILES']].values.ravel('K'))
        # 将SMILES和属性编码传入 属性编码直接转换成数字，SMILES token化后转换成数字
        tokens = set()
        for smi in smiles_list:
            tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

        vocabulary.update(extra_list)
        vocabulary.update(sorted(tokens))
        vocabulary.update(property_condition)
        

    vocabulary = mv.Vocabulary()
    # pad=0, start=1, end=2, default_key for key error
    vocabulary.update(["*", "^", "$", "default_key"])
    interval_token = []
    # 改为固定区间
    # 连续值区间
    interval_token.extend(seq_interval)
    # 布尔值区间
    interval_token.extend(bool_interval)
    csvFiles = glob(f"/home/yichao/zhilian/GenAICode/new_paper_code/mmp_finished/*/*_MMP.csv")
    # 记录smiles main_cls minor_cls
    total = len(csvFiles)
    for idx, file in enumerate(csvFiles):
        LOG.info(f"===handling({idx}/{total}) {file}")
        record_vocal(vocabulary, file, Data_Type.all)
        # if idx > 500:
        #     break
    vocabulary.update(interval_token)
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)
    # Save vocabulary to file
    output_file = './vocab.pkl'
    with open('./vocab.pkl', 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))