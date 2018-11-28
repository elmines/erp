import argparse
import tensorflow as tf
import numpy as np

from tools import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
tf.set_random_seed(0)
np.random.seed(0)


def main(args):
	trainingPercent = 0.75
	paths = readJson("paths.json")	
	neg = filterPackets(readJson(paths["negative"]))
	pos = filterPackets(readJson(paths["positive"]))
	
	(inputData, labelData) = prepare(neg, pos)
	#Shuffle
	randomIndices = np.random.permutation(len(inputData))
	inputData = inputData[randomIndices]
	labelData = labelData[randomIndices]
	(trainingInputs, testingInputs) = partition(inputData, trainingPercent)
	(trainingLabels, testingLabels) = partition(labelData, trainingPercent)

	numNegTraining = sum(trainingLabels == 1)
	numPosTraining = len(trainingLabels) - numNegTraining
	numNegTesting = sum(testingLabels == 1)
	numPosTesting = len(testingLabels) - numNegTesting
	print("Training on {} samples".format(len(trainingInputs)))
	print("\t{} negative training cases, {} positive training cases".format(numNegTraining, numPosTraining))
	print("\t{} negative testing  cases, {} positive testing  cases".format(numNegTesting, numPosTesting))
	

	if args.regression:
		numEpochs = 20
		#TRAINING
		#Graph
		inputs = tf.placeholder(tf.float32, (None, trainingInputs.shape[-1]) )
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
				[lossValue, _] = sess.run([loss, train_op], {inputs: trainingInputs, labels:trainingLabels})
				print("Epoch {}: loss = {}".format(i, lossValue), flush=True)
				print("Testing on {} samples".format(len(testingInputs)))

				[predictionValues] = sess.run([predictions], {inputs: testingInputs} )
				testAccuracy       = accuracy(testingLabels, predictionValues)
				testPrecision      = precision(testingLabels, predictionValues)
				testRecall         = recall(testingLabels, predictionValues)
				testF1             = f1(testingLabels, predictionValues)
				print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
					testAccuracy, testPrecision, testRecall, testF1))
				print()
				if testAccuracy >= bestAccuracy: bestAccuracy = testAccuracy
				else:                        pass #break

	if args.lda:
		print("LDA Model", flush=True)
		ldaModel = LinearDiscriminantAnalysis()
		ldaModel.fit(trainingInputs, trainingLabels)
		ldaPredictions = ldaModel.predict(testingInputs)
		ldaAccuracy    = accuracy(ldaPredictions, testingLabels) #np.mean( testingLabels == ldaPredictions)
		ldaPrecision   = precision(ldaPredictions, testingLabels)
		ldaRecall      = recall(ldaPredictions, testingLabels)
		ldaF1          = f1(ldaPredictions, testingLabels)
		print("\t   accuracy={0}, precision={1}, recall={2}, f1={3}".format(
			ldaAccuracy, ldaPrecision, ldaRecall, ldaF1))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Detect ERPs in EEG")
	parser.add_argument("--regression", action="store_true", help="Train logistic regression model")
	parser.add_argument("--lda", action="store_true", help="Train linear discriminant model")
	args = parser.parse_args()

	main(args)

