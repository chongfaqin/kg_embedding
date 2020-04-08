import pandas as pd
import numpy as np

df=pd.read_csv("data/triple.csv")
df=df[["hid","tid","rel"]]
print(df.head(10))

relation2id={}
file=open("/usr/local/graph-embedding/transR_model/trainData/relation2id.txt","r",encoding="utf-8")
for line in file.readlines():
    line=line.strip()
    arr=line.split("\t")
    if(len(arr)==2):
        relation2id[str(arr[0])]=arr[1]
        #print(line.strip())
relation2id["product"]=1
file.close()
print(relation2id)

entity2id_dict={}
entity2id_file=open("/usr/local/graph-embedding/transR_model/trainData/entity2id.txt","r",encoding="utf-8")
for line in entity2id_file.readlines():
    line=line.strip()
    arr=line.split("\t")
    #print(arr)
    if(len(arr)==2):
        entity2id_dict[str(arr[0])]=arr[1]
entity2id_file.close()

#print(entity2id_dict)

count=np.shape(df)[0]

file_writer=open("/usr/local/graph-embedding/transR_model/trainData/train2id.txt","w",encoding="utf-8")
file_writer.write(str(count)+"\n")
for items in df.values:
    print(items)
    file_writer.write(str(entity2id_dict[str(items[0])])+" "+str(entity2id_dict[str(items[1])])+" "+str(relation2id[str(items[2])])+"\n")
file_writer.close()
