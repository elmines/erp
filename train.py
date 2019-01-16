import argparse
import tensorflow as tf
import numpy as np

import sys

from tools import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
tf.set_random_seed(0)
np.random.seed(0)


def main(args):
	padLength = 256
	trainPercent = 0.75
	rawDataset = readJson(args.data)
	(rawPos, rawNeg) = json2numpy(rawDataset)
	(pos, neg) = (resampleBatch(rawPos, padLength), resampleBatch(rawNeg, padLength))
	taper(pos)
	taper(neg)

	data = np.concatenate( (pos, neg), axis=0 )
	data = data.reshape( (data.shape[0], np.prod(data.shape[1:])) ) #Flatten latter axes
	labels = np.array(len(pos) * [1] + len(neg) * [0])

	#Shuffle
	randomIndices = np.random.permutation(len(data))
	data = data[randomIndices]
	labels = labels[randomIndices]
	(trainInputs, testInputs) = partition(data, trainPercent)
	(trainLabels, testLabels) = partition(labels, trainPercent)

	numNegTrain = sum(trainLabels == 1)
	numPosTrain = len(trainLabels) - numNegTrain
	numNegTest = sum(testLabels == 1)
	numPosTest = len(testLabels) - numNegTest
	print("Training on {} samples".format(len(trainInputs)))
	print("\t{} negative training cases, {} positive training cases".format(numNegTrain, numPosTrain))
	print("\t{} negative testing  cases, {} positive testing  cases".format(numNegTest, numPosTest))


	if args.regression:
		print("Logistic Regression Model")
		logRegression(trainInputs, trainLabels, testInputs, testLabels)
	if args.lda:
		print("LDA Model", flush=True)
		lda(trainInputs, trainLabels, testInputs, testLabels)


def logRegression(trainInputs, trainLabels, testInputs, testLabels):
	numEpochs = 20
	#TRAINING
	#Graph
	inputs = tf.placeholder(tf.float32, (None, trainInputs.shape[-1]) )
	labels = tf.placeholder(tf.float32, None)
	logit = tf.squeeze(tf.layers.dense(inputs, 1))
	loss = tf.losses.sigmoid_cross_entropy(labels, logit)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)

	print("Training logistic regression model for {} epochs".format(numEpochs))
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

def lda(trainInputs, trainLabels, testInputs, testLabels):
	ldaModel = LinearDiscriminantAnalysis()
	ldaModel.fit(trainInputs, trainLabels)
	ldaPredictions = ldaModel.predict(testInputs)
	ldaAccuracy    = accuracy(ldaPredictions, testLabels) #np.mean( testingLabels == ldaPredictions)
	ldaPrecision   = precision(ldaPredictions, testLabels)
	ldaRecall      = recall(ldaPredictions, testLabels)
	ldaF1          = f1(ldaPredictions, testLabels)
	print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
		ldaAccuracy, ldaPrecision, ldaRecall, ldaF1))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Detect ERPs in EEG")
	parser.add_argument("--data", metavar="data.json", required=True, help="Path to EDF-style JSON file")
	parser.add_argument("--regression", action="store_true", help="Train logistic regression model")
	parser.add_argument("--lda", action="store_true", help="Train linear discriminant model")
	args = parser.parse_args()

	main(args)
