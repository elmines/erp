import tensorflow as tf

class Model(object):

	def __init__(self):
		self.trainInputs = None
		self.trainLabels = None
		self.testInputs = None
		self.testLabels = None
		pass

	def loadTraining(self, inputs, labels):
		self.trainInputs = inputs
		self.trainLabels = labels
	def loadTesting(self, inputs, labels):
		self.testInputs = inputs
		self.testLabels = labels

	@property
	def epochs

