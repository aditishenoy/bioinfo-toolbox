# Script to create a new file based on following condition
## If C1 of file1 is C1 of file2 then save row with C1 value in new file

import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Script to extract line from values in one column")
parser.add_argument("--ids_file", type=str, help="")
parser.add_argument("--mappings", type=str, help="")
parser.add_argument("--output", type=str, help="")

args = parser.parse_args()

idsfile = args.ids_file 
mappings = args.mappings
outputfile = args.output

ids_file = open(idsfile, "r")
ids = []

for i in ids_file:
    ids.append(i.strip())

df = pd.read_csv(mappings, header=None, delimiter=' ')
print (df)

ind = []

for i in df.itertuples():
    if i[1] in ids:
        ind.append((i[0]))

df_new = df.loc[ind]
print (df_new)

df_new.to_csv(outputfile, index=False)


