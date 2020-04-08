import pandas as pd
import numpy as np

df=pd.read_csv("data/entity.csv")
print(df.head(10))
print(np.shape(df[["id","nid"]])[0])
#df=df[df["lab"]!="taggp"]
#print(np.shape(df[["id","nid"]])[0])
df=df[["id","lab","nid"]]
#count=np.shape(df[["id","lab","nid"]])[0]

df2=pd.read_csv("data/triple.csv")

a=set(df2["hid"])
b=set(df2["tid"])

c= a | b
count=len(c)
print(count)

index=0
file=open("/usr/local/graph-embedding/transR_model/trainData/entity2id.txt","w",encoding="utf-8")
file.write(str(count)+"\n")
for items in df.values:
    print(items)
    if(items[2] not in c):
        continue
    file.write(str(items[2])+"\t"+str(index)+"\n")
    index=index+1
file.close()
