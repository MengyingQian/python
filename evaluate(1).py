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
import matplotlib.pyplot as plt
import inference

FLAGS = None
lossList = []
trainsteps = []

def main(_):
    Samples_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,89],name='X_input')
    Labels_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,5],name='Y_input')
    
    global_step = tf.Variable(0,trainable=False)
        
    logits = inference.inference(Samples_placeholder)
    
    loss = inference.loss(logits,Labels_placeholder)
    
    train_op = inference.train(loss,global_step)    
    
    evaluation = inference.evaluation(logits, Labels_placeholder)
    
    saver = tf.train.Saver()
    nowstep=999

    for i in range(100):
        model_dir = f'./logfile/train.ckpt-{nowstep:d}'  # tf.train.latest_checkpoint('./logfile')

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)




            saver.restore(sess, model_dir)

            test_data_file = os.path.join(FLAGS.data_dir,'Data_o.mat' )
            if not tf.gfile.Exists(test_data_file):
                raise ValueError('Failed to find file: ' + test_data_file)


            data = scio.loadmat(test_data_file)
            samples = np.array(data['Train_in'],dtype=np.float32)
            labels = np.array(data['Train_out'], dtype=np.float32)

            try:

                eval_value,predict = sess.run([evaluation,logits], feed_dict={Samples_placeholder:samples,Labels_placeholder:labels})
                loss_value, _ = sess.run([loss, train_op],feed_dict={Samples_placeholder: samples, Labels_placeholder: labels})
                print("Loss of testing NN" )
                print(eval_value)
                print(loss_value)
                lossList.append(loss_value)
                test_file = os.path.join(FLAGS.data_dir,'test_result.mat' )
                predict=predict.astype(np.int32)
                scio.savemat(test_file,{'Predict':predict})

            except tf.errors.OutOfRangeError:
                print('Done testing --epoch limit reached')

            finally:
                coord.request_stop()
                coord.join(threads)
        trainsteps.append(nowstep + 1)
        nowstep=nowstep+1000
        print(nowstep)
    plt.plot(trainsteps, lossList, label='First Line')
    plt.show()
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data',help='Path to the train/test data ')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
