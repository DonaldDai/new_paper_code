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
# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")

# 连续值编码
seq_interval = ['(-inf, -10.5]', '(-10.5, -8.5]','(-8.5, -6.5]','(-6.5, -4.5]','(-4.5, -2.5]','(-2.5, -1.5]','(-1.5, -1.0]','(-1.0, -0.5]','(-0.5, 0.0]','(0.0, 0.5]','(0.5, 1.0]','(1.0, 1.5]','(1.5, 2.5]','(2.5, 4.5]','(4.5, 6.5]','(6.5, 8.5]','(8.5, 10.5]','(10.5, inf]']
# 布尔值编码
bool_interval = ['0-1','1-0', '0-0']

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

def save_df_property_encoded(file_name, LOG=None):
    data = pd.read_csv(file_name)
    value_type = data['value_type'][0]
    # 判断是连续值还是bool值
    if value_type == 'seq':
        data['Delta_Value'] = data['Delta_Value'].apply(encode_seq)
    elif value_type == 'bool':
        data['Delta_Value'] = data['Delta_Value'].apply(encode_bool)

    output_file = file_name.split('.csv')[0] + '_encoded.csv'
    LOG.info("Saving encoded property change to file: {}".format(output_file))
    data.to_csv(output_file, index=False)
    return output_file


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=True)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()


if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()
    raw_df=pd.read_csv(args.input_data_path)
    if args.drop_duplicated:
        print("Duplicated ['constantSMILES','fromVarSMILES','toVarSMILES'] will be dropped!")
        dfInput=pd.read_csv(args.input_data_path)
        dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
        dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES','Value_Diff', 'name', 'value_type']]
        dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES','Delta_Value', 'name', 'value_type']
        newPath=Path(args.input_data_path).parent.joinpath("train_valid_test_full.csv")   ## will be saved
        dfInput=dfInput.to_csv(newPath, index=None)
        args.input_data_path=newPath.as_posix()

    # add property name before property change; save to file
    property_condition = []
    # 改为固定区间
    # 连续值区间
    property_condition.extend(seq_interval)
    # 布尔值区间
    property_condition.extend(bool_interval)
    # 添加name
    property_condition.append(raw_df['name'][0])
     
    LOG.info("Property condition tokens: {}".format(len(property_condition)))
    # 将数值转换成编码区间
    encoded_file = save_df_property_encoded(args.input_data_path, LOG)
    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    # 获取所有SMILES分子式列表
    smiles_list = pdp.get_smiles_list(args.input_data_path)  ## updated for constant SMILES
    # 将SMILES和属性编码传入 属性编码直接转换成数字，SMILES token化后转换成数字
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer, property_condition=property_condition)
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

    # Save vocabulary to file
    # 保存词典
    parent_path = uf.get_parent_dir(args.input_data_path)
    output_file = os.path.join(parent_path, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))
# 通过 args.train_ratio 参数来设置训练集的比例
    # Split data into train, validation, test
    # 所有数据分割成训练集和测试集，从训练集中再分出验证集 最后将所有文件保存
    train, validation, test = pdp.split_data(encoded_file, args.train_ratio, LOG)

