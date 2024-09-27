import pandas as pd
import numpy as np
import os
import re
import sqlite3
from glob import glob
from pathlib import Path
import math
from rdkit import Chem
from my_toolset.my_utils import get_mol
from rdkit.Chem.SaltRemover import SaltRemover
import numpy as np
import shutil
from enum import Enum

# pandarallel.initialize(nb_workers=64)
env_bin='/home/yichao/dev_tools/miniconda3/envs/drugassist-jay/bin/'
def match_number(strIn):
    try:
        strIn=str(strIn)
        numberList = re.findall('[0-9\.]+', strIn)
        if len(numberList)>0:
            finalNum=float(numberList[0])
            return finalNum
        else:
            return np.nan
    except:
        return np.nan
    
def rmSalt(smi):
    if not bool(smi):
        return np.nan
    try:
        mol=get_mol(smi)
        if mol==None:
            raise ValueError("Invalid value provided")
            return np.nan
        remover = SaltRemover()  ## default salt remover
        stripped = remover.StripMol(mol) 
        if stripped==None:
            return np.nan
        canoSmi=Chem.MolToSmiles(stripped)
        if not bool(canoSmi):
            return np.nan
        return canoSmi
    except Exception as e:
        print('============rmSalt Error')
        return np.nan

def format_zero(val):
    return 0.0 if val == 0 else val

def check_seq_or_bool(df_series) -> str:
    for v in df_series:
        if v !=0 or v!= 1:
            return 'seq'
    return 'bool'

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

def fmt_cls_to_path(cls) -> str:
    ret = cls
    ret = ret.replace(' ', '-')
    ret = ret.replace('/', '-')
    return ret

def create_MMP(df, icsvFile, main_cls: str, minor_cls: str, target_name: str, target_idx: int, seq: str, fmtValue, keys=['canonical_smiles','Pki'], workDir='', target_dir='MMPFinised'):
    rootPath=Path(workDir)
    ifPreprocess=1
    ifFragment=1
    ifIndex=1
    ifProp=1
    ifOutputPair=1
    ifMove=1  # move the finined MMP folder to finished

    # filename不再用于读取数据，只用于命名目标文件名
    fileName=f'{main_cls}_{fmt_cls_to_path(minor_cls)}_target{target_idx}__{Path(icsvFile).stem}'
    # 之前已经跑过的数据不处理
    if rootPath.joinpath(target_dir).joinpath(fileName).is_dir():
        print(f'===== exist skip handle for : {rootPath.joinpath(target_dir).joinpath(fileName)}')
        return
    rootPath.joinpath(fileName).mkdir(exist_ok=True, parents=True)
    value_type = check_seq_or_bool(df[keys[1]])
    if ifPreprocess:
        # df=pd.read_csv(icsvFile)
        df=df[keys] ## the header of activity should be activity
        # 去除空值
        df=df.dropna(subset=[keys[1]])
        df.columns=['SMILES','value']
        # 过滤空值
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df=df.dropna(subset=['value'])
        df['value']=df['value'].apply(fmtValue)  ## to pIC50 or pKi
        # 去盐
        df['SMILES']=df['SMILES'].apply(rmSalt)
        df=df.dropna(subset=['value','SMILES'])
        # 去重
        df=df.drop_duplicates(subset=['SMILES'])  ## if there is duplicated SMILES the MMP job will be failed
        df['title']=[f'CPD-{i}' for i in range(len(df))]
        for i in df['SMILES']:
            break
        os.chdir(rootPath.joinpath(fileName))
        df.to_csv(f"{fileName}.csv",index=False)
        
        dfSmi=df[['SMILES','title']]
        dfSmi.to_csv(f"{fileName}.smi",sep='\t',index=False,header=False)
        
        dfProp=df[['title','value']]
        dfProp.columns=['ID','value']
        dfProp.to_csv(f"{fileName}_Props.csv",sep='\t',index=False)

    if ifFragment:
        os.chdir(rootPath.joinpath(fileName))
        # print(os.getcwd())
        os.system(f"{env_bin}mmpdb fragment {fileName}.smi -o {fileName}.fragments")
        
    if ifIndex:
        os.chdir(rootPath.joinpath(fileName))
        os.system(f"{env_bin}mmpdb index {fileName}.fragments -o {fileName}.mmpdb")
        
    if ifProp:
        os.chdir(rootPath.joinpath(fileName))
        os.system(f"{env_bin}mmpdb loadprops -p {fileName}_Props.csv {fileName}.mmpdb")
        
    if ifOutputPair:
        os.chdir(rootPath.joinpath(fileName))
        query1='''SELECT pair.compound1_id,
            pair.compound2_id,
            compound.input_smiles,
            compound_property.value,
            constant_smiles.smiles,
            rule_environment.rule_id
        FROM   pair
        LEFT JOIN constant_smiles ON pair.constant_id=constant_smiles.id 
        LEFT JOIN rule_environment ON pair.rule_environment_id=rule_environment.id 
        LEFT JOIN compound ON pair.compound1_id=compound.id 
        LEFT JOIN compound_property ON pair.compound1_id=compound_property.compound_id'''
        # WHERE  pair.compound1_id=compound.id AND pair.compound1_id=compound_property.compound_id'''
                
        query2='''SELECT compound1_id,
            compound2_id,
            input_smiles,
            value
        FROM   pair
        LEFT JOIN compound ON pair.compound2_id=compound.id 
        LEFT JOIN compound_property ON pair.compound2_id=compound_property.compound_id'''
        
                
        query3='''SELECT rule.id,
            rule.from_smiles_id,
            rule.to_smiles_id,
            rule_smiles.smiles
        FROM   rule
        LEFT JOIN rule_smiles ON rule.from_smiles_id=rule_smiles.id'''
        
        query4='''SELECT rule.id,
            rule_smiles.smiles
        FROM   rule
        LEFT JOIN rule_smiles ON rule.to_smiles_id=rule_smiles.id'''

        query_pair='''SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='pair';'''

        # create a connection to the SQLite database
        conn = sqlite3.connect(f'{fileName}.mmpdb')
        table_df = pd.read_sql_query(query_pair, conn)
        if len(table_df) < 1:
            print(f'======WARNING: NO PAIR EXIST: {fileName}')
            directory_path = rootPath.joinpath(fileName)
            if directory_path.exists() and directory_path.is_dir():
                shutil.rmtree(directory_path)
                print("目录已被删除")
            else:
                print("指定的路径不存在或不是一个目录")
            return
        # execute the SQL query and store the results in a Pandas DataFrame
        df1 = pd.read_sql_query(query1, conn)
        df1.columns=['cpd1ID','cpd2ID','cpd1SMILES','cpd1Value','constantSMILES','ruleId']
        df1=df1.drop_duplicates(subset=['cpd1ID','cpd2ID'])
        df2 = pd.read_sql_query(query2, conn)
        df2.columns=['cpd1ID','cpd2ID','cpd2SMILES','cpd2Value']
        df2=df2.drop_duplicates(subset=['cpd1ID','cpd2ID'])
        
        ''' varable smiles '''
        df3 = pd.read_sql_query(query3, conn)
        df3.columns=['ruleId','from_smiles_id',
            'to_smiles_id','fromVarSMILES']
        df4 = pd.read_sql_query(query4, conn)
        df4.columns=['ruleId','toVarSMILES']
        dfRule=pd.merge(df3,df4,on=['ruleId'],how='inner')
        
        
        dfComb=pd.merge(df1,df2,on=['cpd1ID','cpd2ID'],how='inner')
        dfComb=pd.merge(dfComb,dfRule,on=['ruleId'],how='inner')
        
        '''   order the cpds   '''
        # TODO: 排序可能不需要了
        dfComb=dfComb.dropna(subset=['cpd1Value','cpd2Value'])
        for idx,irow in dfComb.copy().iterrows():
            if irow['cpd1Value']>irow['cpd2Value']:
                ''' Switch value '''
                tmp_cpd1Value=dfComb.loc[idx,'cpd1Value']
                dfComb.loc[idx,'cpd1Value']=dfComb.loc[idx,'cpd2Value']
                dfComb.loc[idx,'cpd2Value']=tmp_cpd1Value
                ''' Switch SMILES '''
                tmp_smi=dfComb.loc[idx,'cpd1SMILES']
                dfComb.loc[idx,'cpd1SMILES']=dfComb.loc[idx,'cpd2SMILES']
                dfComb.loc[idx,'cpd2SMILES']=tmp_smi
                ''' switch the cpd1ID and cpd2ID'''
                tmp_cpd1ID=dfComb.loc[idx,'cpd1ID']
                dfComb.loc[idx,'cpd1ID']=dfComb.loc[idx,'cpd2ID']
                dfComb.loc[idx,'cpd2ID']=tmp_cpd1ID
                ''' Switch varibale SMILES '''
                tmp_var=dfComb.loc[idx,'fromVarSMILES']
                dfComb.loc[idx,'fromVarSMILES']=dfComb.loc[idx,'toVarSMILES']
                dfComb.loc[idx,'toVarSMILES']=tmp_var
                
        dfComb['Value_Diff']=dfComb['cpd2Value']-dfComb['cpd1Value']
        dfComb['Value_Diff']=dfComb['Value_Diff'].round(2)
        # 增加反向数据，这里故意没有去除diff为0的数据行，增加数据多样性
        new_rows = dfComb.copy()
        new_rows['cpd1Value'], new_rows['cpd2Value'] = dfComb['cpd2Value'], dfComb['cpd1Value']
        new_rows['cpd1SMILES'], new_rows['cpd2SMILES'] = dfComb['cpd2SMILES'], dfComb['cpd1SMILES']
        new_rows['cpd1ID'], new_rows['cpd2ID'] = dfComb['cpd2ID'], dfComb['cpd1ID']
        new_rows['fromVarSMILES'], new_rows['toVarSMILES'] = dfComb['toVarSMILES'], dfComb['fromVarSMILES']
        new_rows['Value_Diff'] = -dfComb['Value_Diff']

        finalDf = pd.concat([dfComb, new_rows], ignore_index=True)
        # 处理 -0.0
        finalDf['Value_Diff'] = finalDf['Value_Diff'].apply(format_zero)
        finalDf['value_type'] = value_type
        finalDf['main_cls'] = main_cls
        finalDf['minor_cls'] = minor_cls
        finalDf['target_name'] = target_name
        finalDf['sequence'] = seq
        # 保存
        finalDf.to_csv(f"{fileName}_MMP.csv", index=None)
        conn.close()

    if ifMove:
        finishedPath=rootPath.joinpath(target_dir)
        finishedPath.mkdir(exist_ok=True, parents=True)
        os.system(f"mv {rootPath.joinpath(fileName)} {finishedPath}")

class Data_Type(Enum):
    target_name = 'target'
    base = 'base'
    seq = 'seq'
    seq_esm = 'seq_esm'
    all = 'all'