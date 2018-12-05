import pandas as pd
import os

gt_src = 'gt_orig'
upc_brands = {}

brand = pd.read_csv("upc_brand.txt",skiprows=0,sep=" ",dtype=str)
brand.columns = ['upc','brand']
brand.head()
upc = brand['upc']
br = brand['brand']
for i in range(len(upc)):
    upc_brands[upc[i]] = br[i]

txt_files = os.listdir(gt_src)

missed_upcs = []
for txt_file in txt_files:
    path = os.path.join(gt_src,txt_file)
    #print(path)
    try:
        df = pd.read_csv(path,sep=' ',dtype=str)
        df.columns = ['x','y','w','h','upc']
        missing = df.loc[~df['upc'].isin(upc_brands.keys())]
        missed = missing['upc'].tolist()
        missed_upcs.extend(missed)
    except pd.io.common.EmptyDataError:
        print(path, " is empty and has been skipped.") 

missed_upcs = list(set(missed_upcs))
#missed_upcs = [int(x) for x in missed_upcs]
print("Missed UPC's are")
for i in missed_upcs:
    print(i)

