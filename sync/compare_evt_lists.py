#!/usr/bin/python

import sys
import csv
import numpy as np
import pandas as pd

# sort out command line args
print('\n...running with args:', str(sys.argv))

# test mode looks at default files
test_mode = False
if (len(sys.argv) < 2): test_mode = True
if test_mode:
    file_1 = open('test_files/test_1.csv', 'r')
    file_2 = open('test_files/test_2.csv', 'r')
else:
    file_1 = open(sys.argv[1], 'r')
    file_2 = open(sys.argv[2], 'r')

# build a dataframe for each file
df_1 = pd.read_csv(file_1)
df_2 = pd.read_csv(file_2)
print(df_1)
print(df_2)

mutual_cols = [c for c in df_1.columns.values 
               if c in df_2.columns.values]
print('...comparing the following mutual variables:\n', mutual_cols)

overlap = pd.merge(df_1, df_2, on=mutual_cols)
print('...overlap in event lists:', overlap)

isin_1_and_2 = df_1[mutual_cols].isin(df_2[mutual_cols]).dropna()
isin_2_and_1 = df_2[mutual_cols].isin(df_1[mutual_cols]).dropna()
only_in_1 = df_1[np.sum(~isin_1_and_2, axis=1).astype(bool)]
only_in_2 = df_2[np.sum(~isin_2_and_1, axis=1).astype(bool)]

print('...overlap: {0}, in_1_only: {1}, in_2_only: {2}'
      .format(overlap.shape[0], only_in_1.shape[0], only_in_2.shape[0]))

if test_mode:
    only_in_1.to_csv('output_files/test_out_1.csv')
    only_in_2.to_csv('output_files/test_out_2.csv')
else:
    only_in_1.to_csv(sys.argv[1])
    only_in_2.to_csv(sys.argv[2])
