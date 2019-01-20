import tensorflow as tf
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from metrics import *

import sys

def logRegression(trainSet, testSet):
	#TRAINING
	#Graph
	inputs = tf.placeholder(tf.float32, (None, trainSet.inputs.shape[-1]) )
	labels = tf.placeholder(tf.float32, None)
	logit = tf.squeeze(tf.layers.dense(inputs, 1))
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)

	trainSession(trainSet, testSet, inputs, labels, loss, predictions)

def convolution(trainSet, testSet):
	inputs = tf.placeholder(tf.float32, (None,) + trainSet.inputs.shape[1:])
	print("input.shape =", inputs.shape)
	labels = tf.placeholder(tf.float32, None)

	convolved = tf.layers.conv1d(inputs, 1, 64)
	print("convolved.shape =", convolved.shape)
	pooled = tf.layers.max_pooling1d(convolved, 3, 1)
	print("pooled.shape =", pooled.shape)


	projection = tf.layers.dense( tf.layers.flatten(pooled), 1)
	print("projection.shape =", projection)
	logit = tf.squeeze(projection)
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)

	#sys.exit(0)
	trainSession(trainSet, testSet, inputs, labels, loss, predictions)

def trainSession(trainSet, testSet, inputs, labels, loss, predictions):
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)


	trainInputs = trainSet.inputs
	testInputs = testSet.inputs
	trainLabels = trainSet.labels
	testLabels = testSet.labels


	numEpochs = 20
	print("Training model for {} epochs".format(numEpochs))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		bestAccuracy = -1
		for i in range(numEpochs):
			[lossValue, _] = sess.run([loss, train_op], {inputs: trainInputs, labels:trainLabels})
			print("Epoch {}: loss = {}".format(i, lossValue), flush=True)
			print("Testing on {} samples".format(len(testInputs)))

			[predictionValues] = sess.run([predictions], {inputs: testInputs} )
			testAccuracy       = accuracy(testLabels, predictionValues)
			testPrecision      = precision(testLabels, predictionValues)
			testRecall         = recall(testLabels, predictionValues)
			testF1             = f1(testLabels, predictionValues)
			print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
				testAccuracy, testPrecision, testRecall, testF1))
			print()
			if testAccuracy >= bestAccuracy: bestAccuracy = testAccuracy
			else:                        pass #break

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

