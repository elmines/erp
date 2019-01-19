import tensorflow as tf
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from metrics import *

def logRegression(trainSet, testSet):
	#TRAINING
	#Graph
	inputs = tf.placeholder(tf.float32, (None, trainSet.inputs.shape[-1]) )
	labels = tf.placeholder(tf.float32, None)
	logit = tf.squeeze(tf.layers.dense(inputs, 1))
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)

	trainSession(trainSet, testSet, inputs, labels, loss, train_op, predictions)

def trainSession(trainSet, testSet, inputs, labels, loss, train_op, predictions):
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

