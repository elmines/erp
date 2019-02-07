import tensorflow as tf
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .models import Model
from .metrics import *
from .data import Dataset

import os

def logRegression(shape):
	#TRAINING
	#Graph
	inputs = tf.placeholder(tf.float32, (None,) + tuple(shape), name="inputs")
	flatInputs = tf.layers.flatten(inputs)

	labels = tf.placeholder(tf.float32, None)
	logit = tf.squeeze(tf.layers.dense(flatInputs, 1))
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)


	m = Model(inputs=inputs, labels=labels, loss=loss, train_op=train_op,
		predictions=predictions)
	return m

def convolution(shape, classes=1):
	"""
	:param shape: The shape of the inputs (should be [numChannels, samplesPerChannel])
	:param int classes: The number of different output classes
	"""
	inputs = tf.placeholder(tf.float32, (None,) + tuple(shape), name="inputs")

	if classes > 1:
		labels = tf.placeholder(tf.float32, (None, classes), name="labels")
	else:
		labels = tf.placeholder(tf.float32, (None,), name="labels")


	convolved = tf.layers.conv1d(inputs, filters=6, kernel_size=96, strides=1)
	pooled = tf.layers.max_pooling1d(convolved, pool_size=3, strides=2)
	logit = tf.squeeze( tf.layers.dense(tf.layers.flatten(pooled), classes) )
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32, name="predictions")

	m = Model(inputs=inputs, labels=labels, loss=loss, train_op=train_op,
		predictions=predictions)

	return  m

	trainSession(trainSet, testSet, inputs, labels, loss, predictions,
		model_dir=model_dir)
	print("inputs =", inputs)
	print("predictions =", predictions)

def trainSession(trainSet, testSet, model, model_dir=None):

	inputs = model.inputs
	labels = model.labels
	loss = model.loss
	train_op = model.train_op
	predictions = model.predictions

	trainInputs = trainSet.inputs
	testInputs = testSet.inputs
	trainLabels = trainSet.labels
	testLabels = testSet.labels

	numEpochs = 20
	print("Training model for {} epochs".format(numEpochs))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		bestAccuracy = -1
		for i in range(numEpochs):
			[lossValue, _] = sess.run([loss, train_op], {inputs: trainInputs, labels:trainLabels})
			print("Epoch {}: loss = {}".format(i, lossValue), flush=True)
			print("\tTesting on {} samples".format(len(testInputs)))

			[predictionValues] = sess.run([predictions], {inputs: testInputs} )
			testAccuracy       = accuracy(testLabels, predictionValues)
			testPrecision      = precision(testLabels, predictionValues)
			testRecall         = recall(testLabels, predictionValues)
			testF1             = f1(testLabels, predictionValues)
			print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
				testAccuracy, testPrecision, testRecall, testF1))
			if testAccuracy >= bestAccuracy:
				if model_dir:
					bestAccuracy = testAccuracy
					save_path = saver.save(sess, os.path.join(model_dir, "model.ckpt"))
					print("\tSaved new best model to",save_path)
			else:
				pass #break
			print()

def lda(trainSet, testSet):
	(trainInputs, trainLabels) = (trainSet.inputs, trainSet.labels)
	(testInputs, testLabels) = (testSet.inputs, testSet.labels)

	ldaModel = LinearDiscriminantAnalysis()
	ldaModel.fit(trainInputs, trainLabels)
	ldaPredictions = ldaModel.predict(testInputs)
	ldaAccuracy    = accuracy(ldaPredictions, testLabels) #np.mean( testingLabels == ldaPredictions)
	ldaPrecision   = precision(ldaPredictions, testLabels)
	ldaRecall      = recall(ldaPredictions, testLabels)
	ldaF1          = f1(ldaPredictions, testLabels)
	print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
		ldaAccuracy, ldaPrecision, ldaRecall, ldaF1))
