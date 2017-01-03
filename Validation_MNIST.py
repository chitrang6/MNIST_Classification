# Author:- Chitrang Talaviya
# This file contains the code for the evluation of the trained LeNet model. 


import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from model_definition import *
from inputToMNISTModel import *

#train_dir = '/Users/chitrangtalaviya/Desktop/planettest/log/train'
#log_dir = '/Users/chitrangtalaviya/Desktop/planettest/log/eval'

cwd = os.getcwd()
train_dir = cwd + '/log/train/'
log_dir = cwd + 'log/eval'

def main():
	tf.logging.set_verbosity(tf.logging.DEBUG)
	images , labels = InputsToMNISTModel('TEST.tfrecords' , 64 , 1000 , one_hot_labels = False)
	predictions , _ = LeNet_Model(images)
	predictions = tf.to_int32(tf.argmax(predictions, 1))
	tf.scalar_summary('accuracy', slim.metrics.accuracy(predictions, labels))
	metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        "streaming_mse": slim.metrics.streaming_mean_squared_error(predictions, labels)})
	names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })

	print('Running evaluation Loop...')
	checkpoint_path = tf.train.latest_checkpoint(train_dir)
	metric_values = slim.evaluation.evaluate_once(
        master='',
        checkpoint_path=checkpoint_path,
        logdir=log_dir,
        num_evals=1000,
        eval_op=names_to_updates.values(),
        final_op=names_to_values.values(),
        summary_op=tf.merge_all_summaries())
	names_to_values = dict(zip(names_to_values.keys(), metric_values))
	for name in names_to_values:
		print('%s: %f' % (name, names_to_values[name]))


if __name__ == '__main__':
	main()
