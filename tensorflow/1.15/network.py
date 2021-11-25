import sys
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from utils import tftools
from utils import tfutils 
from utils import tflosses
from utils import tfwatcher
from functools import partial

class Network:
	def __init__(self):
		self.graph = tf.Graph()
		gpu_options = tf.GPUOptions(allow_growth=True)
		tf_config = tf.ConfigProto(gpu_options=gpu_options,
				allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(graph=self.graph, config=tf_config)
			
	def initialize(self, config, num_classes):
		'''
			Initialize the graph from scratch according config.
		'''
		with self.graph.as_default():
			with self.sess.as_default():
				# Set up placeholders
				h, w = config.image_size
				channels = config.channels
				image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
				label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
				learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
				keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
				phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
				global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

				image_splits = tf.split(image_batch_placeholder, config.num_gpus)
				label_splits = tf.split(label_batch_placeholder, config.num_gpus)
				grads_splits = []
				split_dict = {}
				def insert_dict(k,v):
					if k in split_dict: split_dict[k].append(v)
					else: split_dict[k] = [v]
						
				for i in range(config.num_gpus):
					scope_name = '' if i==0 else 'gpu_%d' % i
					with tf.name_scope(scope_name):
						with tf.variable_scope('', reuse=i>0):
							with tf.device('/gpu:%d' % i):
								images = tf.identity(image_splits[i], name='inputs')
								labels = tf.identity(label_splits[i], name='labels')
								# Save the first channel for testing
								if i == 0:
									self.inputs = images

								
								

								network = imp.load_source('network', config.network)
								net, end_points = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
														bottleneck_layer_size = config.embedding_size, 
														weight_decay = config.weight_decay)

								prelogits = net
								

								# Build all losses
								loss_list = []

								softmax_loss = tflosses.softmax_loss(prelogits, labels, 64,
													scope='AuxSoftmax', **config.losses['softmax'])
								loss_list.append(softmax_loss)
								insert_dict('aux_loss', softmax_loss)

								
								reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
								loss_list.append(reg_loss)
								insert_dict('reg_loss', reg_loss)

								total_loss = tf.add_n(loss_list, name='total_loss')
				
				


				# Merge the splits
				# grads = tfutils.average_grads(grads_splits)
				for k,v in split_dict.items():
					v = tfutils.average_tensors(v)
					tfwatcher.insert(k, v)
					if 'loss' in k:
						tf.summary.scalar('losses/' + k, v)
					else:
						tf.summary.scalar(k, v)
					
				fn_vars = tf.trainable_variables()
				
				self.train_op = tf.train.AdamOptimizer().minimize(total_loss)
				self.update_global_step_op = tf.assign_add(global_step, 1)

				
				tf.summary.scalar('learning_rate', learning_rate_placeholder)
				summary_op = tf.summary.merge_all()

				# Initialize variables
				self.sess.run(tf.local_variables_initializer())
				self.sess.run(tf.global_variables_initializer())
				self.saver = tf.train.Saver(fn_vars, max_to_keep=None)

				

				# Keep useful tensors
				self.image_batch_placeholder = image_batch_placeholder
				self.label_batch_placeholder = label_batch_placeholder 
				self.learning_rate_placeholder = learning_rate_placeholder
				self.keep_prob_placeholder = keep_prob_placeholder 
				self.phase_train_placeholder = phase_train_placeholder 
				self.global_step = global_step
				
				self.summary_op = summary_op
				

	def train(self, image_batch, label_batch, learning_rate, keep_prob):
		feed_dict = {self.image_batch_placeholder: image_batch,
					self.label_batch_placeholder: label_batch,
					self.learning_rate_placeholder: learning_rate,
					self.keep_prob_placeholder: keep_prob,
					self.phase_train_placeholder: True,}

		_, wl, sm, step = self.sess.run([self.train_op,
			tfwatcher.get_watchlist(), self.summary_op, self.update_global_step_op], feed_dict = feed_dict)
		
		return wl, sm, step
