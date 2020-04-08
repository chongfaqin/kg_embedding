#import config
#import models
#import tensorflow as tf
import numpy as np
import pandas as pd
import json

entity2id={}
file=open("/usr/local/graph-embedding/transR_model/trainData/entity2id.txt","r")
for line in file.readlines():
	print(line.strip())
	line_arr=line.strip().split("\t")
        #print(line_arr)
	if(len(line_arr)==1):
		continue
	entity2id[int(line_arr[1])]=line_arr[0]
file.close()

entity_info={}
df=pd.read_csv("data/entity.csv")
for items in df.values:
	#print(line)
	entity_info[str(items[0])]=list(items)

#con = config.Config()
f = open("/usr/local/graph-embedding/transR_model/res/embedding.vec.json", "r")
embeddings = json.loads(f.read())
#print(embeddings)
f.close()

result_all=[]
embedding_arr=embeddings["ent_embeddings"]
for id,embedding in enumerate(embedding_arr):
	embedding=np.round(embedding,decimals=4)
	print(id,embedding)
	nid=entity2id[id]
	items=entity_info[str(nid)]
        #embeddings=[str(v) for v in embedding]
	embedding_str=",".join([str(v) for v in embedding])
	items.append(embedding_str)
	result_all.append(items)

file_writer=open("/usr/local/graph-embedding/result/embedding.txt","w",encoding="utf-8")
for items in result_all:
	file_writer.write("\t".join([str(v) for v in items])+"\n")
file_writer.close()
