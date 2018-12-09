# -*- coding: utf-8 -*-

# This model is used for user behavior prediction
#  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
from six.moves import xrange
import os.path
import time
#import matplotlib.pyplot as plt
import inference
import scipy.io as scio

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
#LEARNING_RATE_BASE = 0.01
#LEARNING_RATE_DECAY_FACTOR = 0.98

FLAGS = None

def main(_):
    Samples_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,89],name='X_input')
    Labels_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,5],name='Y_input')
    
    global_step = tf.Variable(0,trainable=False)
        
    logits = inference.inference(Samples_placeholder)
    
    loss = inference.loss(logits,Labels_placeholder)
    
    train_op = inference.train(loss,global_step)    
    
    evaluation = inference.evaluation(logits, Labels_placeholder)
  
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        sess.run(init)
        
        
        # load data
        train_file = os.path.join(FLAGS.data_dir,'Data_o.mat' )
        if not tf.gfile.Exists(train_file):
            raise ValueError('Failed to find file: ' + train_file)
        
        
        data = scio.loadmat(train_file)

        
        samples = np.array(data['Train_data'],dtype=np.float32)
        labels = np.array(data['Train_label'],dtype=np.float32)
        
        train_samples = samples
        train_labels = labels
        test_samples = samples
        test_labels = labels


        #Para = data['Data_PS']

        for i in xrange(FLAGS.max_steps):
            start_time = time.time()            
            loss_value,_=sess.run([loss,train_op],feed_dict={Samples_placeholder:samples,Labels_placeholder:labels})
            duration=time.time()-start_time
            if (i+1) % 10 == 0:                
                print("Loss of training NN, step: %d, loss: %f, and time:%f" % (i+1,loss_value,duration))
            #if (i+1) % 100 == 0 :
                eval_value = sess.run(evaluation, feed_dict={Samples_placeholder:train_samples,Labels_placeholder:train_labels})
                
                #print("Evaluation of test samples, step: %d, loss: %f." % (i+1,eval_value))
                #summary_writer.add_summary(summary_str, i)
                
            if i%1000 == 0 or (i+1) == FLAGS.max_steps:
                eval_value = sess.run(evaluation, feed_dict={Samples_placeholder:samples,Labels_placeholder:labels})
                print(eval_value)
                
                
        checkpoint_path = os.path.join(FLAGS.train_dir, 'train.ckpt')
        saver.save(sess, checkpoint_path, global_step=i)        
        predict = sess.run(logits, feed_dict={Samples_placeholder:test_samples,Labels_placeholder:test_labels})        
        test_file = os.path.join(FLAGS.data_dir,'predict.mat' )
        scio.savemat(test_file,{'Predict':predict})

    
    
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data',help='Path to the train/test data ')
    #parser.add_argument('--batch_size',type=int,default=50,help='The size of minibatch')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='The initial learning rate')
    parser.add_argument('--max_steps',type=int,default=100000,help='The max steps for training')
    parser.add_argument('--train_dir',type=str,default='logfile',help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)