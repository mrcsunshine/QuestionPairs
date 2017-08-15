import tensorflow as tf 
import datahelper

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)

# checkpoint_dir='./runs/1495545740/checkpoints/'
#vocab=datahelper.read_vocab()
#questionlist=datahelper.load_question()
#ques_file='../../data/test/qid_question_head.txt'
checkpoint_dir='./cnnrun/cnnrun/runs/1496370939/checkpoints/'
#checkpoint_dir='./checkpoints/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
saver=tf.train.import_meta_graph(checkpoint_dir+'model-40000.meta')
graph=tf.get_default_graph()
# print(graph.get_all_collection_keys())
# print(graph.get_collection_ref('train_op'))
# print(graph.get_collection_ref('summaries'))
#print(graph.get_operations())
# loss=graph.get_operation_by_name('loss_1')
# accuracy=graph.get_operation_by_name('accuracy_1')
# train_op = graph.get_operation_by_name('Adam')
# global_step = graph.get_tensor_by_name('global_step:0')
# grad_summaries_merged=graph.get_operation_by_name('Merge/MergeSummary')
# train_summary_op=graph.get_operation_by_name('Merge_1/MergeSummary')
input1=graph.get_operation_by_name('input_x_1').outputs[0]
dropout_keep_prob=graph.get_operation_by_name("dropout_keep_prob").outputs[0]
train_op=graph.get_operation_by_name("Adam").outputs[0]
pool1=graph.get_operation_by_name('pooled_reshape_1').outputs[0]
#print(graph.get_operations())
print(train_op)
print(input1)
print(pool1)
with tf.session(config=session_conf) as sess:
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        pass
    try:
        i=int(0)
        while true:
            ques=datahelper.load_question_data(questionlist,vocab,i,batch_size)
            feed_dict = {input1:ques,dropout_keep_prob:1}
            features=sess.run(pool1,feed_dict)
            print("index: {}\n".format(i))
            for feature in features :
                value=[]
                for j in range(0,2000):
                    value.append(feature[j])
                featurelist.append(value)
            i+=batch_size
            if i>=len(questionlist):
                print('out of questionlist')
                break
        sessdict = {}
        index = int(0)
        resname = "./qid_question_head_feature" + ".txt"     ####存放question vector的目录
        if os.path.exists(resname):
            os.remove(resname)
        result = codecs.open(resname, 'w', 'utf-8')  
        file=codecs.open(ques_file,'r',encoding='utf-8')
        line=file.readline() 
        while line!='':
          #for line in open(val_file):
            items = line.strip().split(' ')
            if len(items)==2:
                if len(items[1].strip().split('_'))==101:
                    qid = items[0].split(':')[1]
                    qid=int(qid)
                    if not qid in sessdict:
                        sessdict[qid] = []
                    sessdict[qid].append((featurelist[index]))
                    index += 1
                    print('score index:'+str(index))
                #print("line:{}".format(line))
                #if index>=4000:
                    if index >= len(questionlist):
                        print('index out of testlist: {}\n'.format(index))
                        break
                    if index >=len(featurelist):
                        print('index out of scorelist: {}\n'.format(len(featurelist)))
                        break
                else:
                    print('not complete line: '+str(line))
            else:
                print('not complete line: '+str(line))
            line=file.readline()
        for k, v in sessdict.items():
        #v.sort(key=operator.itemgetter(0), reverse=true)
            result.write('qid:'+str(k)+'_'+str(v)+'\n')
        #print('qid'+str(k))
        print('done')
        result.close()
    except exception as e:
        print(e)
        traceback.print_exc()
