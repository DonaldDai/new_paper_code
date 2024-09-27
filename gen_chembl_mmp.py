import os
import pandas as pd
import numpy as np
from glob import glob
import math
from multiprocessing import Pool, cpu_count
import time
from functools import partial
import numpy as np
import datetime
from common.utils import create_MMP

def get_chembl_main_cls(minor_cls) -> str:
    ret = 'unknown'
    if minor_cls in ['IC50', 'Ki', 'Kd', 'EC50']:
        ret = 'activity'
    elif minor_cls in ['Drug uptake', 'Papp', 'permeability', 'F', 'Fu', 'Tmax', 'Cmax']:
        ret = 'absorption'
    elif minor_cls in ['Vdss', 'Cp']:
        ret = 'distribution'
    elif minor_cls in ['Drug metabolism', 'LogD', 'LogP']:
        ret = 'metabolism'
    elif minor_cls in ['AUC', 'CL', 'FC']:
        ret = 'excretion'
    elif minor_cls in ['Stability', 'LD50', 'LC50', 'T1/2']:
        ret = 'toxicity'
    elif minor_cls in ['solubility', 'Thermal melting change', 'pKa']:
        ret = 'general physical properties'

    if ret == 'unknown':
        print(f'ERROR: cannot find a chembl main class. ({minor_cls})')
    return ret

def get_chembl_minor_cls(file) -> str:
    ret = 'unknown'
    df = pd.read_csv(file)
    if len(df['standard_type']) > 0:
        cls = df['standard_type'][0]
        if cls:
            ret = cls
    return ret

def get_fmt(file):
    df = pd.read_csv(file)
    data = df['standard_value']
    # 计算极值
    min_value = data.min()
    max_value = data.max()
    # 计算范围
    range_value = max_value - min_value
    def do_nothing(x):
        return x
    def do_log(x):
        ret = np.log10(x)
        if math.isinf(ret):
            return np.nan
        return ret
    if range_value >= 10:
        return do_log
    return do_nothing

def task(file, rootDir, idx, total):
    print(f'{datetime.datetime.now()}===({idx}/{total})handle {file}')
    df = pd.read_csv(file)
    if len(df) < 2:
        return
    create_MMP_p=partial(create_MMP, workDir=rootDir, target_dir='mmp_finished')
    minor_cls = get_chembl_minor_cls(file)
    fmt_fn = get_fmt(file)
    main_cls = get_chembl_main_cls(minor_cls)
    df['fmt_target_name'] = df['target_name'] + df['target_organism']
    all_targets = df['fmt_target_name'].unique()
    print(f'===({idx}/{total}){main_cls}|{minor_cls} target counts: {len(all_targets)}')
    for idx, target_name in enumerate(all_targets):
        filtered_df = df[df["fmt_target_name"] == target_name]
        print(f'===({idx}/{total})total data num({target_name}): {len(filtered_df)}')
        if len(filtered_df) < 2:
            continue
        # 多进程下这里要用 iloc 方法取值
        seq = filtered_df['target_sequence'].iloc[0]
        create_MMP_p(filtered_df, file, main_cls, minor_cls, target_name, idx, seq, fmt_fn, keys=['canonical_smiles', 'standard_value'])

def handle_chembl():
    # ChemBl数据处理
    rootDir=os.getcwd()
    csvFiles=glob(f"{rootDir}/raw_chembl_data/*.csv")
    start = time.time()
    p = Pool(int(cpu_count()/2))
    listLen = len(csvFiles)
    for idx, file in enumerate(csvFiles):
        p.apply_async(task, args=(file, rootDir, idx + 1, listLen))
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))

if __name__ == '__main__':
    handle_chembl()