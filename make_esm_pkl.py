import pickle
import torch
import esm
import pandas as pd
import time
from glob import glob
import re
from pathlib import Path
from multiprocessing import Pool
import os

CHUNK_SIZE = 32
CUT_SIZE = 2000

# 目录路径
directory = Path("esm_temp")

# 创建新目录
if not directory.exists():
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{directory}' created successfully.")

print('start')
start = time.time()
data = []
chunk_data = []
csvFiles = glob(f"/home/yichao/zhilian/GenAICode/new_paper_code/mmp_finished/*/*_MMP.csv")
total = len(csvFiles)
for idx, file in enumerate(csvFiles):
    print(f'===handling({idx}/{total}) {file}')
    df = pd.read_csv(file, nrows=2)
    if len(df) < 1:
        continue
    seq = df['sequence'].iloc[0]
    seq = seq if isinstance(seq, str) else ''
    seq = seq.upper()
    # 过长的要截断
    seq = seq[:CUT_SIZE]
    print(f'==chunk_data len: {len(chunk_data)}', seq)
    if not bool(seq):
        continue
    chunk_data.append((f'protein{idx}', seq))
    if len(chunk_data) >= CHUNK_SIZE or idx == (total - 1):
        data.append(chunk_data)
        chunk_data = []
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
print(f'===protrains chunk count: {len(data)}')

def task(chunk_data, idx, total):
    print(f"==handling({idx}/{total}) ")
    tar_filename = f'./esm_temp/seq_vec_dict_{idx}.pkl'
    if os.path.exists(tar_filename):
        print(f"File exists, skip. {tar_filename}")
        return
    sub_start = time.time()
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    batch_labels, batch_strs, batch_tokens = batch_converter(chunk_data)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    seq_record = {}
    for i, (_, seq) in enumerate(chunk_data):
        seq_record[seq] = token_representations[i, 1 : len(seq) + 1].mean(0)

    key_list = list(seq_record.keys())
    first_vec = seq_record[key_list[0]]
    print(f'===record keys count: {len(key_list)}')
    print(f'===vector len: {len(first_vec)} | type: {type(first_vec)}')

    # 打开一个新文件用于写入，注意'b'代表二进制模式
    with open(tar_filename, 'wb') as file:
        pickle.dump(seq_record, file)
    sub_end = time.time()
    print(f"chunk esm 文件({idx}/{total})总共用时{sub_end - sub_start}秒")

start = time.time()
p = Pool(1)
listLen = len(data)
for idx, chunk_data in enumerate(data):
    p.apply_async(task, args=(chunk_data, idx + 1, listLen))
p.close()
p.join()

end = time.time()
print(f"===make esm file {end - start}s")
