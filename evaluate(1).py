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


    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
        

        model_dir = './model/train.ckpt-99999'#tf.train.latest_checkpoint('./logfile')

        saver.restore(sess, model_dir)
        
        test_data_file = os.path.join(FLAGS.data_dir,'Data_o.mat' )
        if not tf.gfile.Exists(test_data_file):
            raise ValueError('Failed to find file: ' + test_data_file)
        
        
        data = scio.loadmat(test_data_file)
        samples = np.array(data['Train_in'],dtype=np.float32)
        labels = np.array(data['Train_out'], dtype=np.float32)
        
        try:

            eval_value,predict = sess.run([evaluation,logits], feed_dict={Samples_placeholder:samples,Labels_placeholder:labels})
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
