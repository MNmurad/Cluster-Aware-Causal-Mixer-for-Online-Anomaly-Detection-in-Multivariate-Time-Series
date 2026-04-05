import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt

df_attk_details = pd.read_csv('WADI_attacklabels.csv')
tst = pd.read_csv('WADI_attackdata.csv')

attack_date = pd.to_datetime(df_attk_details['Date'], format='%m/%d/%Y')
attack_start = pd.to_datetime(df_attk_details['Start Time'], format='%H:%M:%S').dt.time # df_attk_details['Start Time']
attack_end = pd.to_datetime(df_attk_details['End Time'], format='%H:%M:%S').dt.time # df_attk_details['End Time']

num_attack = attack_date.shape[0] # should be 12
df = pd.DataFrame({'Attack': [0] * tst.shape[0]}) # tst.shape[0] should be 172801
df['Date'] = pd.to_datetime(tst['Date'], format='%m/%d/%Y')
df['Time'] = pd.to_datetime(tst['Time'], format='%I:%M:%S.%f %p').dt.time
df = df[['Date', 'Time', 'Attack']]

for i in range(num_attack):
    condition1 = df['Date'] == attack_date[i]
    condition2 = (df['Time'] >= attack_start[i]) * (df['Time'] <= attack_end[i])
    df['Attack'][condition1 * condition2] = 1
print('Total anom: {}%'.format(df['Attack'].sum() / len(df) * 100))
df.to_csv('Attack_label.csv')