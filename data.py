from collections import namedtuple

class Dataset(namedtuple("Dataset", ["inputs", "labels"])):
	pass

class DataPartition(namedtuple("DataPartition", ["training", "testing", "validation"])):
	pass
