import numpy as np
from numpy import random
import codecs
import pandas
import sys
import os 

empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))
val_file= './seqlen100/rand/qq_test_head1500.txt'
# def build_vocab():
    # code = int(0)
    # vocab = {}
    # vocab['UNKNOWN'] = code
    # code += 1
    # resName = "train_test_vocabulary_100"+".txt"
    # if os.path.exists(resName):
       # os.remove(resName)
    # result= codecs.open(resName, 'w', 'utf-8') 
    # train_file=codecs.open('./query_question_train100.txt','r',encoding='utf-8')
    # #train_file=codecs.open('./train_sample.txt','r',encoding='utf-8')
    # line =train_file.readline()
    # while line!='':
        # items = line.strip().split(' ')
        # if len(items)!=4:
            # print('only contain:'+str(len(items))+':'+line+'\n')
            # line=train_file.readline()
        # else:
            # for i in range(2, 3):
                # words = items[i].split('_')
                # for word in words:
                    # if not word in vocab:
                        # vocab[word] = code
                        # code += 1
            # line=train_file.readline()
    # train_file.close()
    # print('train_vocab done.....')
    # test_file=codecs.open('./query_question_test100_partial.txt','r',encoding='utf-8')
    # #test_file=codecs.open('./test_sample.txt','r',encoding='utf-8')
    # line=test_file.readline()
    # while line!='':
        # items = line.strip().split(' ')
        # if len(items)!=4:
            # print('only contain:'+str(len(items))+':'+line+'\n')
            # line=test_file.readline()
        # else:
            # for i in range(2, 3):
                # words = items[i].split('_')
                # for word in words:
                    # if not word in vocab:
                        # vocab[word] = code
                        # code += 1
            # line=test_file.readline()
    # print('length of vocab:{}'.format(len(vocab)))
    # for k,v in vocab.items():
        # result.write(str(k)+' '+str(v)+'\n')
    # test_file.close()
    # result.close()
    # print('test_vocab done.....')
    # print('vocab done..........')
    # #return vocab

def read_vocab():
    vocab={}
    file=codecs.open('../character/charactor_vocab_new1.txt','r',encoding='utf-8')
    #file=codecs.open('./seqlen100/train_test_vocabulary_100.txt','r',encoding='utf-8')
    line=file.readline()
    while line!='':
        items=line.split(' ')
        if len(items)!=2:
            print('vocab dont legal,len is:'+str(len(items))+'that is:'+str(line))
            line=file.readline()
        else:
            vocab[items[0]]=items[1]
            line=file.readline()
        #vocab[items[0]]=items[1]
    # vocab['下缘']=3
    # vocab['刘琦']=137
    # vocab['stackoverflow']=703
    # vocab['赵长林']=2640
    # vocab['马睿超']=3711
    # vocab['郝璐']=11272
    print('vocab read done')
    return vocab
	
def rand_qa(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def read_alist():
    alist = []
    #file=codecs.open('./query_question_train.txt','r',encoding='utf-8')
    #file=codecs.open('./train_sample.txt','r',encoding='utf-8')
    file=codecs.open('./seqlen100/rand/qq_train100_all.txt','r',encoding='utf-8')
    line=file.readline()
    while line!='':
        if (line=='\n')or (line=='\r') or(line==' '):
            print('tab exist line')
            line=file.readline()
            continue		
        items = line.strip().split(' ')
        if len(items)!=4:
            print('only contain:'+str(len(items))+':'+'\n')
        else:
            alist.append(items[3])
        line=file.readline()
    file.close()
    print('read_alist done ......')
    return alist

# def vocab_plus_overlap(vectors, sent, over, size):
    # global onevector
    # global zerovector
    # oldict = {}
    # words = over.split('_')
    # if len(words) < size:
        # size = len(words)
    # for i in range(0, size):
        # if words[i] == '<a>':
            # continue
        # oldict[words[i]] = '#'
    # matrix = []
    # words = sent.split('_')
    # if len(words) < size:
        # size = len(words)
    # for i in range(0, size):
        # vec = read_vector(vectors, words[i])
        # newvec = vec.copy()
        # #if words[i] in oldict:
        # #    newvec += onevector
        # #else:
        # #    newvec += zerovector
        # matrix.append(newvec)
    # return matrix

# def load_vectors():
    # vectors = {}
    # file=codecs.open('./qq_uniq_re_fenci.vector','r',encoding='utf-8')
    # line=file.readline()
    # while line!='':
        # items = line.strip().split(' ')
        # if (len(items) < 401):
            # line=file.readline()
            # continue
        # vec = []
        # for i in range(1, 401):
            # vec.append(float(items[i]))
        # vectors[items[0]] = vec
        # line =file.readline()
    # vec=[]
    # r=random.uniform(-1,1,[1,400])
    # for i in range(0,400):
        # vec.append(float(r[0][i]))
    # vectors['UNKNOWN']=vec
    # print('vector done.....'+str(len(vectors)))
    # file.close()
    # return vectors

# def read_vector(vectors, word):
    # global empty_vector
    # if word in vectors:
        # return vectors[word]
    # else:
        # return empty_vector
        # #return vectors['</s>']

def load_test_and_vectors():
    testList = []
    #test_file=codecs.open('./query_question_test_random.txt','r',encoding='utf-8')
    #test_file=codecs.open('./partial_16500_test.txt','r',encoding='utf-8')
    #test_file=codecs.open('./test_sample.txt','r',encoding='utf-8')
    #test_file=codecs.open('./query_question_test100_partial.txt','r',encoding='utf-8')
    test_file=codecs.open(val_file,'r',encoding='utf-8')
    line=test_file.readline()
    while line!='':
        if (line=='\n')or (line=='\r') or(line==' '):
            print('tab exist line')
            line=test_file.readline()
            continue
        items = line.strip().split(' ')
        if len(items)==4:
            testList.append(line.strip())
        else:
            print(str(len(items))+' : '+'\n')
        line=test_file.readline()
    #vectors = load_vectors()
    test_file.close()
    print('test and vector done.....')
    return testList


def read_raw():
    raw = []
    #file=codecs.open('./query_question_train.txt','r',encoding='utf-8')
    #file=codecs.open('./train_sample.txt','r',encoding='utf-8')
    file=codecs.open('./seqlen100/rand/qq_train100_all.txt','r',encoding='utf-8')
    line=file.readline()
    while line!='':
        if (line=='\n')or (line=='\r') or(line==' '):
            print('tab exist line')
            line=file.readline()
            continue
        items = line.strip().split(' ')
        if items[0] == '1':
            if len(items)==4:
                raw.append(items)
            else:
                print(str(len(items))+' : '+line+'\n')
        line=file.readline()
    file.close()
    print('read raw done')
    return raw

def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, 100):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def emedding_encode(vector,string,size):
    x=[]
    words=string.split('_')
    for i in range(0,size):
        if words[i] in vector:
            x.append(vector[words[i]])
        else:
            x.append(vector['UNKNOWN'])
    return x
            
    
def load_data_6(vocab, alist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    i=int(0)
    while i<size:
        try:
            items = raw[random.randint(0, len(raw) - 1)]
            nega = rand_qa(alist)
            if (len(items[2].strip().split('_'))==101) and (len(items[3].strip().split('_'))==101):
                x_train_1.append(encode_sent(vocab, items[2], 200))
                x_train_2.append(encode_sent(vocab, items[3], 200))
                x_train_3.append(encode_sent(vocab, nega, 200))
                i+=1
            else:
                print('item[2] len:{}'.format(len(items[2].split('_'))))
                print('item[3] len:{}'.format(len(items[3].split('_'))))
        except Exception as e:
            print(e)
            print('load train data error: '+str(items))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)


def load_data_val_6(testList, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    true_index=index
    i=int(0)
    while i<batch:
        try:          
            if (true_index >= len(testList)):
                true_index = len(testList) - 1
            items = testList[true_index].split(' ')
            if (len(items[2].strip().split('_'))==101) and (len(items[3].strip().split('_'))==101):
                x_train_1.append(encode_sent(vocab, items[2], 100))
                x_train_2.append(encode_sent(vocab, items[3], 100))
                x_train_3.append(encode_sent(vocab, items[3], 200))
                #idlist.append(items[1])
                i+=1
            else:
                print('item[2] len:{}'.format(len(items[2].split('_'))))
                print('item[3] len:{}'.format(len(items[3].split('_'))))
        except Exception as e:
            print(e)
            print('load test data error: '+str(items))
        true_index += 1
    print("test data load....")
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_9(trainList, vectors, size):
    x_train_1 = []
    x_train_2 = []
    y_train = []
    for i in range(0, size):
        pos = trainList[random.randint(0, len(trainList) - 1)]
        posItems = pos.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], posItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, posItems[3], posItems[2], 200))
        y_train.append([1, 0])
        neg = trainList[random.randint(0, len(trainList) - 1)]
        negItems = neg.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], negItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, negItems[3], posItems[2], 200))
        y_train.append([0, 1])
    return np.array(x_train_1), np.array(x_train_2), np.array(y_train)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
			
# def read_array():
    # file="./train_test_id_embedding_vocaball.vector"
    # d=pandas.read_csv(file,sep=' ')
    # d=d.dropna(axis=1)
    # a=d.as_matrix()
    # a1=a[:,::-1].T
    # a2=np.lexsort(a1)
    # a3=a[a2]
    # a4=a3[:,1:]
    # return a4
    
if __name__ == '__main__': 
    #build_vocab();
    vocab=read_vocab()
    print(len(vocab))
    #raw=read_raw()
    #alist=read_alist()
    #x1,x2,x3=load_data_6(vocab,alist,raw,100)
    #print(x1.shape)
    #testlist=load_test_and_vectors()
    #y1,y2,y3=load_data_val_6(testlist,vocab,0,100)
    #print(y1.shape)