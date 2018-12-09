#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
#from six.moves import xrange
import os.path
#import time
import scipy.io as scio

import inference
import pymysql.cursors

FLAGS = None


def main(_):
    Samples_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,89],name='X_input')
    Labels_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,5],name='Y_input')
    
    global_step = tf.Variable(0,trainable=False)
        
    logits = inference.inference(Samples_placeholder)#网络结构
    
    loss = inference.loss(logits,Labels_placeholder)#误差函数
    
    train_op = inference.train(loss,global_step)    #训练方式
    
    evaluation = inference.evaluation(logits, Labels_placeholder)#预测结果和实际结果评估函数
    
    saver = tf.train.Saver()


    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
        

        model_dir = './model/train.ckpt-99999'#tf.train.latest_checkpoint('./logfile')

        saver.restore(sess, model_dir)
        
        test_data_file = os.path.join(FLAGS.data_dir,'Data_o.mat' )
        if not tf.gfile.Exists(test_data_file):
            raise ValueError('Failed to find file: ' + test_data_file)
        
        
        data = scio.loadmat(test_data_file)
        #samples = np.array(data['Train_in'],dtype=np.float32)
        labels = np.array(data['Train_out'],dtype=np.float32)
        #新增
        # Connect to the database
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='qq5099689439',
                                     db='weibodata',
                                     charset='utf8mb4',
                                     )

        try:
            #   with connection.cursor() as cursor:
            # Create a new record
            #      sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"

            #      cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            #  connection.commit()

            with connection.cursor() as cursor:
                # Read a single record
                sql = "SELECT * FROM `向量结果` WHERE `微博昵称`=%s"

                cursor.execute(sql, (sys.argv[1],))
                #sql = "SELECT `微博昵称`, `严谨性` FROM `sheet1`ORDER BY `微博昵称` "
                #cursor.execute(sql)
                result = cursor.fetchone()
                arr_ys = list(result)
                index = 0
                arr = []

                for arr_y in arr_ys:
                    if (index not in [102,101,100,99,93,92,89,80,77,15,14,13,10,0] ) :
                        arr.append(arr_y)
                    index +=1
            #print(arr)



        finally:
            connection.close()

        samples = np.array(arr, dtype=np.float32)
        samples = samples.reshape((1,89))
        #新增结束
        try:

            eval_value,predict = sess.run([evaluation,logits], feed_dict={Samples_placeholder:samples,Labels_placeholder:labels})#打印loss
            print("Loss of testing NN" )
            print(eval_value)
            test_file = os.path.join(FLAGS.data_dir,'test_result.mat' )
            predict=predict.astype(np.int32)
            scio.savemat(test_file,{'Predict':predict})
                   
        except tf.errors.OutOfRangeError:
            print('Done testing --epoch limit reached')
            
        finally:
            coord.request_stop()
            coord.join(threads)
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data',help='Path to the train/test data ')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
