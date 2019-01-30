"""
Script for performing training via the command-line, rather than programmatically.
"""
import sys
import argparse
import tensorflow as tf
import numpy as np

tf.set_random_seed(0)
np.random.seed(0)


sys.path.append("../..")
#Local modules
from erp.tools import *
from erp import ml

def main(args):

	models = [args.lda, args.regression, args.conv]
	if np.sum(models) != 1:
		print("Specify one of --lda, --regression, or --conv")

	packetLength = 256
	maxReading = 80
	trainPercent = 0.75

	rawDataset = readJson(args.data)
	(rawPos, rawNeg) = json2numpy(rawDataset, packetLength)
	pos = cleanBatch(rawPos, maxReading, packetLength)
	neg = cleanBatch(rawNeg, maxReading, packetLength)

	data = np.concatenate( (pos, neg), axis=0 )
	labels = np.array(len(pos) * [1] + len(neg) * [0])

	#Shuffle
	randomIndices = np.random.permutation(len(data))
	data = data[randomIndices]
	labels = labels[randomIndices]

	(trainSet, testSet) = partition(trainPercent, data, labels)


	numNegTrain = sum(trainSet.labels == 1)
	numPosTrain = len(trainSet.labels) - numNegTrain
	numNegTest = sum(testSet.labels == 1)
	numPosTest = len(testSet.labels) - numNegTest
	print("Training on {} samples".format(len(trainSet.inputs)))
	print("\t{} negative training cases, {} positive training cases".format(numNegTrain, numPosTrain))
	print("\t{} negative testing  cases, {} positive testing  cases".format(numNegTest, numPosTest))


	inputShape = trainSet.inputs.shape[1:]
	if args.regression:
		print("Logistic Regression Model")
		model = ml.logRegression(inputShape)
		ml.trainSession(trainSet, testSet, model, args.model_dir)
	if args.lda:
		print("LDA Model", flush=True)
		ml.lda(trainSet, testSet)
	if args.conv:
		print("Convolutional Model")
		model = ml.convolution(inputShape)
		ml.trainSession(trainSet, testSet, model, args.model_dir)




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Detect ERPs in EEG")
	parser.add_argument("--data", metavar="data.json", required=True, help="Path to EDF-style JSON file")
	parser.add_argument("--regression", action="store_true", help="Train logistic regression model")
	parser.add_argument("--lda", action="store_true", help="Train linear discriminant model")
	parser.add_argument("--conv", action="store_true", help="Train a convolutional neural network")

	parser.add_argument("--model_dir", default="ckpts/", help="Directory to save model checkpoints")
	args = parser.parse_args()

	main(args)
