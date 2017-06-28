#coding:utf-8
import tensorflow as tf
import datahelper
import codecs
import operator
import os
import numpy as np
import datetime
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
embedding_size=200
sequence_length=100
batch_size=256
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)

start=datetime.datetime.now()
vocab=datahelper.read_vocab()
featurelist=datahelper.load_featurelist()
qapairs=datahelper.load_qapairs()
end_load=datetime.datetime.now()
print('load data time'+str((end_load-start).seconds))
checkpoint_dir='./1496370939/checkpoints/'

ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
saver=tf.train.import_meta_graph(checkpoint_dir+'model-40000.meta')
graph=tf.get_default_graph()
input1=graph.get_operation_by_name('input_x_1').outputs[0]
dropout_keep_prob=graph.get_operation_by_name("dropout_keep_prob").outputs[0]

pool1=graph.get_operation_by_name('pooled_reshape_1').outputs[0]
filter_sizes=[1,2,3,5]
with tf.Session(config=session_conf) as sess:
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        pass
    num_filters_total = 500 * 4 
    pooled_reshape_2=graph.get_operation_by_name('pooled_reshape_2').outputs[0]
    #pooled_reshape_2=tf.placeholder(tf.float32,[1, num_filters_total],name="question")
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pool1, pool1), 1)) #计算向量长度Batch模式
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_reshape_2, pooled_reshape_2), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(pool1, pooled_reshape_2), 1) #计算向量的点乘Batch模式
    cos_12 = tf.divide(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") #计算向量夹角Batch模式
    scoreList = {}
    i=int(0)
    query=raw_input("大声告诉我你的疑问: ").decode('utf-8')
    start_pro=datetime.datetime.now()
    print('load cnn time: '+str((start_pro-end_load).seconds))
    fen=datahelper.cut(query)
    pad=datahelper.pad(fen)
    x=datahelper.load_query_data(pad,vocab,i,batch_size)
    start_cal=datetime.datetime.now()
    print('preprocess query time:'+str((start_cal-start_pro).seconds))
    while True:
        ques,idlist=datahelper.load_question_feature(featurelist,i,batch_size)
        #print('len of idlist:'+str(len(idlist)))
        feed_dict={input1:x,pooled_reshape_2:ques,dropout_keep_prob:1}
        scores=sess.run(cos_12,feed_dict)
        #print(scores)
        for j in range(0,len(idlist)):
            id=idlist[j]
            score=scores[j]
            if not id in scoreList:
                scoreList[id]=[]
            #print('score id'+str(id))
            #print('score'+str(score))	
            scoreList[id].append(score)			
        # for score in scores :
            # scoreList.append(score)
        i+=batch_size
        #if i>=5:
        if i>=len(featurelist):
            break
    end_cal=datetime.datetime.now()
    print('calculate time: '+str((end_cal-start_cal).seconds))    
    dict=sorted(scoreList.items(),key=lambda d:d[1],reverse=True)
    #print('score top1:'+str(dict[0][1][0][0]))
    #print('score top2:'+str(dict[1][1][0][0]))
    #print('score top3:'+str(dict[2][1][0][0]))
    start_find=datetime.datetime.now()
    top1_id=dict[0][0]
    top1_score=dict[0][1][0]
    print('score'+str(top1_score))
    #print('top1_id:'+str(top1_id))
    top1_question=qapairs[top1_id][0][0].encode('utf-8')
    top1_answer=qapairs[top1_id][0][1].encode('utf-8')
    #print('top1 answer: '+str(qapairs[top1_id][0][1]))
    if top1_score<0.60:
        print('我装作听不懂的样子,可能这是你要的答案：{}'.format(top1_answer))
    else:
        print(str(top1_answer))
    print('top1 question: '+str(top1_question))
    end=datetime.datetime.now()
    print('find answer time: '+str((end-start_find).seconds))
    print('total time: '+str((end-start).seconds))
    bingo=0
    k=1
    while bingo==0:
        feedback=raw_input('问题解决了吗？解决请按1，没解决请按2：')
        if feedback=='1':
            print('真是机智的孩纸，一点就着')
            bingo=1
        elif feedback=='2':
            if k>=3:
                print('恕在下无知，你的问题超纲了')
                break
            else:
                id=dict[k][0]
                print('这条回答也许会让你满意：'+str(qapairs[id][0][1].encode('utf-8')))
                print('question found: '+str(qapairs[id][0][0].encode('utf-8')))
                k+=1
        else:
            feedback=raw_input('你真淘气,皮皮虾请求你认真作答 ')
    # #wei.close()
    # #n_wei.close()
    # sessdict = {}
    # index = int(0) 
    # start_find=datetime.datetime.now()
    # file=codecs.open(qa_file,'r',encoding='utf-8')
    # line=file.readline() 
    # while line!='':
        # if (line=='\n')or (line=='\r')or (line==' ' ):
            # print('tab exist line')
            # line=file.readline()
            # continue
        # items = line.strip().split(' ')
        # if len(items)==2:
            # if len(items[1].split('#'))==2:
                # qid = items[0].split(':')[1]
                # if not qid in sessdict:
                    # sessdict[qid]=[]
                # sessdict[qid].append((scoreList[index],items[1]))
                # index += 1
            # else:
                # print('not complete line: '+str(line))
        # else:
            # print('not complete line：'+str(line))
        # if index>=100:
        # #if index >= len(featurelist):
            # print('index out of featurelist: {}\n'.format(index))
            # break
        # if index >=len(scoreList):
            # print('index out of scorelist: {}\n'.format(len(scoreList)))
            # break
        # line=file.readline()
    # dict=sorted(sessdict.items(),key=lambda d:d[1],reverse=True)
    # k=dict[0]
    # end=datetime.datetime.now()
    # print('find answer time: '+str((end-start_find).seconds))
    # print('total time: '+str((end-start).seconds))
    # if k[1][0][0][0]<0.60:
        # print('我装作听不懂的样子,可能这是你要的答案：{}'.format(str(k[1][0][1].split('#')[1])))
    # else:
        # print(str(k[1][0][1].split('#')[1]))
    # print('top1 question: '+str(k[1]))
    # bingo=0
    # i=1
    # while bingo==0:
        # feedback=input('问题解决了吗？解决请按1，没解决请按2：')
        # if feedback=='1':
            # print('很高兴为你解答')
            # bingo=1
        # elif feedback=='2':
            # if i>=3:
                # print('恕在下无知，你的问题超纲了')
                # break
            # else:
                # print('这条回答也许会让你满意：'+str(dict[i][1][0][1].split('#')[1]))
                # i+=1
        # else:
            # feedback==input('你真淘气,皮皮虾请求你认真作答 ')
    #print(dict)
