'''
Credit to this code goes to: Nominality Score Conditioned Time Series Anomaly
Detection by Point/Sequential Reconstruction.
https://github.com/andrewlai61616/NPSR
'''
'''
this is the originial code taken from https://github.com/andrewlai61616/NPSR.
However, the original data does not provide attack_label columns, though this code used
attack label column. So, It causes error.
To solve that, I added a code at the bottom(commented) to create a label data frame.
I also slightly modified this code for the generated attack label
'''

###########################################
# 1. First make the label file using make_label.py file
# 2. Then run this file to create pk.
###########################################

import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt

print('read csv files')
print('Note that this study takes the 2017 data')
trn = pd.read_csv('WADI_14days.csv', skiprows=3)
tst = pd.read_csv('WADI_attackdata.csv')
label = pd.read_csv('Attack_label.csv') # Below I mentioned

print('shorten column labels and separate labels')
# shorten column labels
cols = trn.columns.to_numpy()
target_str = '\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\'
for i in range(len(cols)):
    if target_str in cols[i]:
        cols[i] = cols[i][len(target_str):]
trn.columns = cols
lab_tst = label['Attack'].values.astype(np.int32) # tst[tst.columns[-1]].to_numpy()

assert len(set(lab_tst)) == 2

# tst = tst.drop(columns = [tst.columns[-1]])
tst.columns = cols

print('drop columns and rows')
# drop Row, Date, Time
trn = trn[cols[3:]]
tst = tst[cols[3:]]
cols = cols[3:]

# drop columns that have excessive NaNs
drop_cols = cols[np.isnan(trn.to_numpy()).sum(axis=0) > len(trn) // 2]
tst = tst.drop(columns=drop_cols)
trn = trn.drop(columns=drop_cols)

# convert to numpy array
print('convert to numpy array')
trn_np = trn.to_numpy().astype(np.float32)
tst_np = tst.to_numpy().astype(np.float32)
cols = trn.columns.to_numpy()

# fill NAs
print('fill NAs for trn')
nanlist = np.isnan(trn_np).sum(axis=0)
print(nanlist)
for j, nancnt in enumerate(nanlist):
    if nancnt > 0:
        for i in range(len(trn_np)):
            if np.isnan(trn_np[i, j]):
                trn_np[i, j] = trn_np[i-1, j]
                nancnt -= 1
                if nancnt == 0:
                    break
assert np.isnan(trn_np).sum() == 0 and np.isnan(tst_np).sum() == 0

print('save to pickle file')
with open('WADI.pk', 'wb') as file:
    pk.dump({'x_trn': trn_np, 'x_tst': tst_np, 'lab_tst': lab_tst, 'cols': cols}, file)
    
print('done, final x_trn, x_tst, lab_tst shape: ', trn_np.shape, tst_np.shape, lab_tst.shape)

#%% Creating the labeld df: For the actual start end as per I created
# import pandas as pd
# import numpy as np
# import pickle as pk
# import datetime as dt

# df_attk_details = pd.read_csv('WADI_attacklabels.csv')
# tst = pd.read_csv('WADI_attackdata.csv')

# attack_date = pd.to_datetime(df_attk_details['Date'], format='%m/%d/%Y')
# attack_start = pd.to_datetime(df_attk_details['Start Time'], format='%H:%M:%S').dt.time # df_attk_details['Start Time']
# attack_end = pd.to_datetime(df_attk_details['End Time'], format='%H:%M:%S').dt.time # df_attk_details['End Time']

# num_attack = attack_date.shape[0] # should be 12
# df = pd.DataFrame({'Attack': [0] * tst.shape[0]}) # tst.shape[0] should be 172801
# df['Date'] = pd.to_datetime(tst['Date'], format='%m/%d/%Y')
# df['Time'] = pd.to_datetime(tst['Time'], format='%I:%M:%S.%f %p').dt.time
# df = df[['Date', 'Time', 'Attack']]

# for i in range(num_attack):
#     condition1 = df['Date'] == attack_date[i]
#     condition2 = (df['Time'] >= attack_start[i]) * (df['Time'] <= attack_end[i])
#     df['Attack'][condition1 * condition2] = 1
# print('Total anom: {}%'.format(df['Attack'].sum() / len(df) * 100))
# df.to_csv('Attack_label.csv')

