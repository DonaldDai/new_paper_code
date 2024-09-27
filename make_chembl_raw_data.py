import sqlite3
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

def get_data(conn, type):
    query=f'''SELECT assays.chembl_id,
               activities.standard_type,
               activities.standard_relation,
               activities.standard_value,
               activities.standard_units,
               target_dictionary.chembl_id,
               compound_structures.canonical_smiles,
               molecule_dictionary.chembl_id,
               docs.chembl_id,
               assays.confidence_score,
               target_dictionary.pref_name,
               target_dictionary.organism,
               component_sequences.sequence
        FROM   activities,
               assays,
               compound_structures,
               molecule_dictionary,
               target_dictionary,
               docs,
               target_components,
               component_sequences
        WHERE  activities.standard_type='{type}' and
               activities.assay_id=assays.assay_id and
               activities.molregno = compound_structures.molregno and
               activities.molregno = molecule_dictionary.molregno and
               activities.doc_id=docs.doc_id and
               assays.tid=target_dictionary.tid and
               target_dictionary.tid = target_components.tid and
               target_components.component_id = component_sequences.component_id;'''
    
    # execute the SQL query and store the results in a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    
    # save the DataFrame to a CSV file
    df.columns=["chembl_id_assay","standard_type","standard_relation","standard_value","standard_units","chembl_id_target","canonical_smiles","chembl_id_molecule","chembl_id_doc","confidence_score","target_name", "target_organism", "target_sequence"]
    conn.close()
    return df
def task(type, idx, total):
    print(f'==={type}({idx}/{total}) handling')
    # create a connection to the SQLite database
    conn = sqlite3.connect('/home/yichao/zhilian/chembl_34.db')
    df = get_data(conn, type)
    value_counts = df['standard_units'].value_counts()
    unit = ''
    for value, count in value_counts.items():
        unit = value
        break
    print(f'==={type}({idx}/{total}) use unit {unit}')
    df_filter = df[df['standard_units'] == unit]
    name_suffix = type.replace(' ', '-')
    name_suffix = type.replace('/', '-')
    df_filter.to_csv(f'./raw_chembl_data/chembl34_{name_suffix}.csv', index=False)
    print(f'==={type}({idx}/{total}) finished')
start = time.time()
p = Pool(int(cpu_count()/2))
list = ['IC50', 'Ki', 'Kd', 'EC50', 'T1/2', 'CL', 'FC', 'Stability', 'LC50', 'Drug uptake', 'AUC', 'F', 'Drug metabolism', 'LogD', 'Cmax', 'LD50', 'Vdss', 'permeability', 'Papp', 'LogP', 'Cp', 'Thermal melting change', 'pKa', 'Tmax', 'Fu', 'solubility']
listLen = len(list)
for idx, type in enumerate(list):
    p.apply_async(task, args=(type, idx, listLen))
p.close()
p.join()
end = time.time()
print("总共用时{}秒".format((end - start)))
