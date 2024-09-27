import pandas as pd
import numpy as np
import os
import re
import sqlite3
from glob import glob
from pathlib import Path
import math
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import time
from functools import partial
from my_toolset.my_utils import get_mol
from rdkit.Chem.SaltRemover import SaltRemover
import numpy as np
import datetime
import shutil
from common.utils import create_MMP

def fmt_bingdb_value(x):
        ret = np.nan
        try:
            ret  = 9 - np.log10(x)
            if math.isinf(ret):
                return np.nan
            return ret
        except Exception as e:
            print(f'===========fmt error', x)
            return np.nan
def task(key, idx, total):
    print(f'===({idx}/{total}) handle {key}')
    rootDir=os.getcwd()
    create_MMP_p=partial(create_MMP, workDir=rootDir, target_dir="mmp_finished_b")
    fmt_minor_cls=key.split(' (')[0]
    df = pd.read_csv("/home/yichao/zhilian/BindingDB_All_202407.csv")
    all_targets = df['Target Name'].unique()
    main_cls = 'activity'
    print(f'===({idx}/{total}) {main_cls}|{fmt_minor_cls} target counts: {len(all_targets)}')
    for idx, target_name in enumerate(all_targets):
        filtered_df = df[df["Target Name"] == target_name]
        print(f'===({idx}/{len(all_targets)}) total data num({target_name}): {len(filtered_df)}')
        if len(filtered_df) < 2:
            continue
        seq_df = filtered_df['BindingDB Target Chain Sequence']
        if len(seq_df) < 1:
            print(f'=== No seq for {target_name}')
            continue
        seq = seq_df.iloc[0]
        create_MMP_p(filtered_df, "/home/yichao/zhilian/BindingDB_All_202407.csv", 'activity', fmt_minor_cls, target_name, idx, seq, fmt_bingdb_value, keys=['Ligand SMILES', key])

def handle_bindingdb():
    # BDingDB 数据处理
    start = time.time()
    list = ['Ki (nM)','IC50 (nM)','Kd (nM)','EC50 (nM)']
    p = Pool(4)
    listLen = len(list)
    for idx, key in enumerate(list):
        # task(key, idx + 1, listLen)
        p.apply_async(task, args=(key, idx + 1, listLen))
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
        

if __name__ == '__main__':
    handle_bindingdb()