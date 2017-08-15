#encoding=utf-8 

import numpy as np
import random
import codecs 
import sys  
import re  
import codecs  
import os 

def load_qapairs(qa_file):
    qadict = {}
    qa=codecs.open(qa_file,'r',encoding='utf-8')
    line=qa.readline()
    while line!='':
        if (line=='\n')or (line=='\r')or (line==' ' ):
            print('tab exist line')
            line=qa.readline()
            continue
        items = line.strip().split(' ')
        if len(items)==2:
            qapairs=items[1].split('#')
            if len(qapairs)==2:
                qid = items[0].split(':')[1]
                #question=qapairs[0]
                #answer=qapairs[1]
                if not qid in qadict:
                    qadict[qid]=[]
                #print(items[1])
                qadict[qid].append(items[1])
            else:
                print('not complete qapairs')
        else:
            print('not complete qapairs')
        line=qa.readline()
    #print('qapairs len:'+str(len(qadict)))
    qa.close()
    print('qapairs done.....')
    return qadict


def load_featurelist_all(feature_file):
    feature_dict={}
    test_file=codecs.open(feature_file,'r',encoding='utf-8')
    line=test_file.readline()
    i=int(0)
    while line!='':
        if i>=0:
            items=line.strip().split('_')
            if len(items)==2:
                id=items[0].strip().split(':')[1]       
                value=items[1].strip().replace('[','').replace(']','').split(',')
                a=[]
                for j in range(0,2000):
                #print(value[i])
                    a.append(float(value[j]))
                if not id in feature_dict:
                    feature_dict[id]=[]  
                feature_dict[id].append(a)
            #idlist.append(id)
            line=test_file.readline()
            i+=1
        else:
            i+=1
            #line=test_file.readline()
            #break
    test_file.close()
    print('question feature done.....')
    #return np.array(x),idlist
    return feature_dict
	
def comb_qa_feature(qadict,feature_dict,res_file):
    res = open(res_file, 'w', encoding='utf-8')    
    for k,v in feature_dict.items():
        if k in qadict.keys():
            print(k)
            #print(type(qadict[k]))
            #print(qadict[k][0])
            #print('value type'+str(type(v)))
            #print('value shape'+str(v[0]))
            res.write(str(qadict[k][0]).replace(' ','').replace('\t','').replace('\u3000','')+'\t'+str(v[0]).replace(', ','|').replace('[','').replace(']','')+'\n')
        else:
            print('qa dont contain: '+str(k))
    print('combine done')
    res.close()

if __name__=='__main__':
    qa_di=load_qapairs("./model3/data/qid_qapairs_new.txt")
    fea_l=load_featurelist_all("./model/data/newdata/qid_question_exten_feature.txt")
    comb_qa_feature(qa_di,fea_l,"./model/data/newdata/qa_vec8.txt")