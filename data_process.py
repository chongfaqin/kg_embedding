import numpy as np
import pandas as pd
import sys

code=sys.argv[1]
print(code)
#has embedding item
goods_df=pd.read_csv("../data/embedding.txt",delimiter="\t",names=["nid","id","name","tag","embedding"])
print(np.shape(goods_df))
goods_df=goods_df[goods_df["tag"]=="goods"]
goods_set=set(goods_df["id"])

item_set=set()
user_embd_dict={}
item_dict={}
item_index=0
item_list=[]
file_name="../data/user_action"+str(code)+".data"
print(file_name)
with open(file_name,"r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.strip().replace("[","").replace("]","").replace("\"","").split("#")
        #print(arr)
        if(len(arr)==2):
            lt=[]
            actions=arr[1].split(",")
            for actionItem in actions:
                actionArr=actionItem.split(":")
                if(actionArr[0]=='click' and int(actionArr[1]) in goods_set):
                    # lt.append(str(actionArr[1]))
                    if(actionArr[1] not in item_dict.keys()):
                        lt.append(item_index)
                        item_dict[actionArr[1]]=item_index
                        item_list.append(item_index)
                        item_index=item_index+1
                    else:
                        lt.append(item_dict[actionArr[1]])
            if(len(lt)>0):
                user_embd_dict[arr[0]]=lt
#encode
index=0
user_index={}
user_index_clk={}
user_clk_count={}
for k,v in user_embd_dict.items():
    user_index[k]=index
    user_index_clk[index]=v
    user_clk_count[index]=len(v)
    index=index+1
    #print(index,k)

#neg samples
user_index_neg={}
all_count=len(item_list)
np.random.shuffle(item_list)
for k,c in user_clk_count.items():
    count=c*1
    loc_index=np.random.randint(0,high=all_count-count)
    v=item_list[loc_index:loc_index+count]
    #print(all_count,loc_index,count,k, v)
    user_index_neg[k]=v

print(len(user_index_neg))

with open("../result"+str(code)+"/user_index.txt","w",encoding="utf-8") as file:
    for k,v in user_index.items():
        file.write(k+","+str(v)+"\n")

with open("../result"+str(code)+"/item_index.txt","w",encoding="utf-8") as file:
    for k,v in item_dict.items():
        file.write(str(k)+","+str(v)+"\n")

with open("../result"+str(code)+"/user_index_clk.txt","w",encoding="utf-8") as file:
    for k,v in user_index_clk.items():
        v_str=" ".join([str(vi) for vi in v])
        file.write(str(k)+","+v_str+"\n")

with open("../result"+str(code)+"/user_index_neg.txt","w",encoding="utf-8") as file:
    for k,v in user_index_neg.items():
        v_str=" ".join([str(vi) for vi in v])
        file.write(str(k)+","+v_str+"\n")
