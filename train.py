import argparse
import tensorflow as tf
import numpy as np

tf.set_random_seed(0)
np.random.seed(0)

#Local modules
from tools import *
import ml

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

	(trainSet, testSet) = partition(trainPercent, data, labels)


	numNegTrain = sum(trainSet.labels == 1)
	numPosTrain = len(trainSet.labels) - numNegTrain
	numNegTest = sum(testSet.labels == 1)
	numPosTest = len(testSet.labels) - numNegTest
	print("Training on {} samples".format(len(trainSet.inputs)))
	print("\t{} negative training cases, {} positive training cases".format(numNegTrain, numPosTrain))
	print("\t{} negative testing  cases, {} positive testing  cases".format(numNegTest, numPosTest))


	if args.regression:
		print("Logistic Regression Model")
		ml.logRegression(trainSet, testSet)
	if args.lda:
		print("LDA Model", flush=True)
		ml.lda(trainSet, testSet)




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Detect ERPs in EEG")
	parser.add_argument("--data", metavar="data.json", required=True, help="Path to EDF-style JSON file")
	parser.add_argument("--regression", action="store_true", help="Train logistic regression model")
	parser.add_argument("--lda", action="store_true", help="Train linear discriminant model")
	args = parser.parse_args()

	main(args)
