import pandas as pd
import numpy as np

df=pd.read_csv("data/relation2id.csv",names=["rel","id"])
print(df.head(10))

df2=pd.read_csv("data/triple.csv")
a=set(df["rel"])

count=len(a)
file=open("/usr/local/graph-embedding/transR_model/trainData/relation2id.txt","w")
file.write(str(count)+"\n")
index=0
for items in df.values:
    if(items[0] not  in a):
        continue
    file.write(items[0]+"\t"+str(index)+"\n")
    index=index+1
file.close()
