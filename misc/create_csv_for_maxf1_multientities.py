
import os
import pandas as pd
import numpy as np
data = 'SMD'
path = './' #'../outputs/Anomaly_HyperParam_Search'
file = f'{data}.xlsx'

file_path = os.path.join(path, file)

df = pd.read_excel(file_path)
out_df = pd.DataFrame(columns = df.columns)

unique_entity = df['entity_id'].unique()
for i in unique_entity:
    rowidx = df[df['entity_id'] == i]['loss'].idxmin()
    df_row = df.loc[rowidx: rowidx]
    out_df = pd.concat([out_df, df_row], ignore_index = True)
    
out_df.to_excel(f'{data}_bestF1.xlsx', index = False)