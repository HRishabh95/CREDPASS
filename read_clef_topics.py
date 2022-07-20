f=open("/tmp/pycharm_project_889/clef2020_queries_task11.txt","rb").readlines()

ids=[]
querys=[]
for i in f:
    if "<id>" in i.decode():
        ids.append(int(i.decode().split(">")[1].split("<")[0].rstrip().lstrip()))
    if "<en>" in i.decode():
        querys.append(i.decode().split(">")[1].split("<")[0].rstrip().lstrip())

clef_topics=[]
for j in zip(ids,querys):
    clef_topics.append([j[0],j[1]])

import pandas as pd
pd.DataFrame(clef_topics).to_csv("/home/ubuntu/rupadhyay/CREDPASS/clef_topics.csv",sep=' ',header=False,index=False)


import pandas as pd
d=pd.read_csv('/tmp/pycharm_project_329/topics_des.csv',sep=' ',header=None,index_col=0)
d.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec_topics_desc.csv',index=None,header=None,sep=' ')