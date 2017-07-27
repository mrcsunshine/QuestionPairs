import datetime
import numpy as np
import codecs
import faiss
import sys

def load_featurelist_all(feature_file1):
    #feature_file1为利用模型计算question的feature向量，参考restore.py，加载为feature矩阵，用来生成index
    x=[]
    idlist=[]
        #featurelist = []
    test_file=codecs.open(feature_file1,'r',encoding='utf-8')
    line=test_file.readline()
    while line!='':
        items=line.strip().split('_')
        if len(items)==2:
            id=items[0].strip().split(':')[1]
            value=items[1].strip().replace('[','').replace(']','').split(',')
            a=[]
            for j in range(0,2000):
                #print(value[i])
                a.append(float(value[j]))
            x.append(np.array(a))
            idlist.append(id)
        line=test_file.readline()
    test_file.close()
    #print(type(1.0))
    #print(sys.getsizeof((1.0)))
    print('question feature done.....')
    return np.array(x).astype('float32'),np.array(idlist).astype('int')

def search_cos(features,idlist):
    d = 2000                           # dimension
    res=codecs.open('./test_out2.txt','r',encoding='utf-8')    #测试question的feature向量数据，用来测试index召回效果
    line=res.readline()
    xq=[]
    while line!='':
        value=line.strip().replace('[','').replace(']','').split(',')
        a=[]
        for j in range(0,2000):
                #print(value[i])
            a.append(float(value[j]))
        xq.append(np.array(a))
        line=res.readline()
    res.close()
    print('xq done') 
    xq=np.array(xq).astype('float32')
    ncentroids = 12500
    nquantizers = 16
    k = 10
    ####利用pca降维
    #coarse_quantizer = faiss.IndexFlatL2 (256)
    #sub_index = faiss.IndexIVFPQ (coarse_quantizer, 256, nlist, 16, 8)
    # PCA 2048->256
    # also does a random rotation after the reduction (the 4th argument)
    #pca_matrix = faiss.PCAMatrix (d, 256, 0, True) 
    #- the wrapping index
    #index = faiss.IndexPreTransform (pca_matrix, sub_index)
    
    
    ####index=faiss.read_index("./large4.index")   ###读取之前训练好的index

    ####利用点积做精确搜索
    ##index=faiss.IndexFlatIP(d)
    
	####将数据聚类后建立index
    quantizer=faiss.IndexFlatIP(d)      ####利用点积作为quantizer
    index=faiss.IndexIVFFlat(quantizer,d,ncentroids)    ####对数据进行聚类并建立倒排索引IVF生成index

    ####利用PQ压缩建立index
    #index = faiss.IndexFlatIP(d)  
    ###index = faiss.IndexIVFPQ(quantizer, d, ncentroids, nquantizers, 8)
                                 # 8 specifies that each sub-vector is encoded as 8 bits

    ####利用feature vector进行训练    
    index.train(features)

    index2=faiss.IndexIDMap(index)       ####将index转换成可以加载有id数据的类型
    index2.add_with_ids(features,idlist)    ####加载有id的feature数据
    faiss.write_index(index2, '/mnt/large5.index')    ###将index保存下来

   
    D, I = index.search(features[:5], k) # 利用feature本身前5条搜索，进行检查，D为距离，I为搜索到的id
    print(I)
    print(D)
    index.nprobe = 500              # 设定搜索时搜索的聚类数
    start=datetime.datetime.now()
    D, I = index.search(xq, k)     # 利用xq在index中搜索k的最近邻
    end=datetime.datetime.now()
    print("search time:"+str((end-start).microseconds/1000))
    print(I)
   

if __name__ == '__main__':
    featurelist,idlist=load_featurelist_all("/mnt/norm_data/qid_question_feature_1_norm.txt")
    search_cos(featurelist,idlist)
    #print(featurelist.shape)
    #print(sys.getsizeof(featurelist))
