import pandas as pd
from glob import glob
import shutil
import os
from const import ROOT
import time

def lower_in(a, b):
  return a.lower() in b.lower()

def check_tar(name):
  ret = False
  # Orexin 1/2
  if lower_in('Orexin', name):
    ret = True
  # Dopamine D3
  elif lower_in('Dopamine D3', name) or (lower_in('D(3)', name) and lower_in('dopamine', name)):
    ret = True
  # Adenosine receptor A3
  elif lower_in('Adenosine receptor A3', name) or (lower_in('Adenosine', name) and lower_in('A3', name)):
    ret = True
  # Mu-type opioid receptor
  elif lower_in('mu', name) and lower_in('opioid', name) and not lower_in('mus', name):
    ret = True
  # Cannabinoid CB1 receptor
  elif lower_in('Cannabinoid', name) or (lower_in('cb1', name) and lower_in('1', name)):
    ret = True
  # Serotonin 6 (5-HT6) receptor
  elif lower_in('5-hydroxytryptamine', name) or (lower_in('Serotonin', name) and lower_in('HT6', name)):
    ret = True
  # Adenosine receptor A1
  elif lower_in('Adenosine', name) and lower_in('A1', name):
    ret = True
  # Adenosine receptor A2a
  elif lower_in('Adenosine', name) and lower_in('A2a', name):
    ret = True
  # Kappa opioid receptor
  elif lower_in('Kappa', name) and lower_in('opioid', name):
    ret = True
  return ret

start = time.time()
# TODO 更改路径
files = glob(f'{ROOT}/mmp_finished/*/*_MMP.csv')
files_len = len(files)
for idx, file_path in enumerate(files):
  df = pd.read_csv(file_path)
  if len(df) < 1:
    print(f'skip, no data on {file_path}')
    continue
  
  print(f'==={idx+1}/{files_len} handling')
  # 获取target_name列的第一个值
  first_value = df['target_name'].iloc[0]

  # 检查是否包含字符串"abc"
  if check_tar(first_value):
      # 获取文件的目录
      file_dir = os.path.dirname(file_path)
      
      # 设置目标目录
      target_dir = f'{ROOT}/mmp_finished_rm'
      
      # 移动文件夹
      shutil.move(file_dir, target_dir)
      print(f"=============move\n {file_dir}\n to\n {target_dir}\n===========\n\n")

end = time.time()
print("总共用时{}秒".format((end - start)))