#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The model of feature extraction, the model is based on CNN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io as scio
#import data
#import numpy as np

import tensorflow as tf
#from six.moves import xrange


MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.衰减呈阶梯函数，控制衰减周期（阶梯宽度）
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.学习率衰减因子
INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.初始学习率，过小收敛过慢，过大不收敛，0.1
LAMBDA = 0.001   #正则化项在误差函数中所占比重，超参数
#0.005,0.002
LAYER1 = 80
LAYER2 = 40
LAYER3 = 20

def inference(datasets):
    # Define the archtecture of cnn for feature extraction


    with tf.variable_scope('layer_1') as scope:
        weights = tf.get_variable('weights', shape=[89,LAYER1],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))#初始化权重，高斯搜一下
        biases = tf.get_variable('biases', shape=[LAYER1],initializer=tf.constant_initializer(0.0))
        result1=tf.nn.relu(tf.matmul(datasets,weights)+biases)#激活函数，主要是把线性变为非线性
        tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(LAMBDA)(weights))#如果不加限制，权重在叠加过程中会变得特别大，跟对不同特征权值如果不加限制数量级会不同，会造成模型不收敛
        
    with tf.variable_scope('layer_2') as scope:
        weights = tf.get_variable('weights', shape=[LAYER1,LAYER2],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[LAYER2],initializer=tf.constant_initializer(0.0))
        result2=tf.nn.relu(tf.matmul(result1,weights)+biases)
        tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(LAMBDA)(weights))
        
    with tf.variable_scope('layer_3') as scope:
        weights = tf.get_variable('weights', shape=[LAYER2,LAYER3],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[LAYER3],initializer=tf.constant_initializer(0.0))
        result3=tf.nn.relu(tf.matmul(result2,weights)+biases)    
        tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(LAMBDA)(weights))
        
    with tf.variable_scope('linear_reg') as scope:
        weights = tf.get_variable('weights', shape=[LAYER3,5],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[5],initializer=tf.constant_initializer(0.0))
        result4=tf.matmul(result3,weights)+biases
        tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(LAMBDA)(weights))
        
    return result4

def loss(logits,labels):
    
    labels = tf.cast(labels,tf.float32)
    loss = tf.reduce_mean(tf.square(labels - logits))  
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    #cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',loss) #把张量cross_entropy_loss添加到字典集合中key='losses'的子集中

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')  #返回字典集合中key='losses'的子集中元素之和


def train(total_loss, global_step):#将最后输出结果的误差，（反向传播）分配到前向

    decay_steps = 100#每经过decay_step步训练，衰减lr 
 
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)#搜这个函数可得初始值，随着训练增加学习率变动
    
    opt = tf.train.GradientDescentOptimizer(lr)
    train_op = opt.minimize(total_loss)
    
    return train_op  
    
def evaluation(logits, labels):#评估函数在训练过程重打印输出
    
    error = tf.reduce_mean(tf.abs(labels-logits),0)
    return error
    #return tf.reduce_mean(error)


def read_and_decode(filename , shuffle_batch=True):
    data = scio.loadmat(filename)
    samples = data['Train_data']
    labels  = data['Train_label']
    Para = data['Data_PS']
        
        
        
        
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _,serialized_example = reader.read(filename_queue)
    features =tf.parse_single_example(serialized_example,features={'sample':tf.FixedLenFeature([],tf.string),'label':tf.FixedLenFeature([],tf.int64)})
    
    sample = tf.decode_raw(features['sample'],tf.uint8)
    sample = tf.cast(sample, tf.float32)
    sample = tf.reshape(sample, [1024])
    
    label = features['label']
    
#    if shuffle_batch:
#        samples, labels = tf.train.shuffle_batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4,min_after_dequeue=2000)
#    else:
#        samples, labels = tf.train.batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4)
    
    return samples, labels, Para