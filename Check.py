import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

def load_sca_model(model_file):
	check_file_exists(model_file)
	try:
		model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model

# Compute the rank of the real key for a give set of predictions
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
	# Compute the rank
	if len(last_key_bytes_proba) == 0:
		# If this is the first rank we compute, initialize all the estimates to zero
		key_bytes_proba = np.zeros(256)
	else:
		# This is not the first rank we compute: we optimize things by using the
		# previous computations to save time!
		key_bytes_proba = last_key_bytes_proba

	for p in range(0, max_trace_idx-min_trace_idx):
		# Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
		plaintext = metadata[min_trace_idx + p]['plaintext'][0]
		for i in range(0, 256):
			# Our candidate key byte probability is the sum of the predictions logs
			proba = predictions[p][plaintext ^ i]
			if proba != 0:
				key_bytes_proba[i] += np.log(proba)
			else:
				# We do not want an -inf here, put a very small epsilon
				# that correspondis to a power of our min non zero proba
				min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
				if len(min_proba_predictions) == 0:
					print("Error: got a prediction with only zeroes ... this should not happen!")
					sys.exit(-1)
				min_proba = min(min_proba_predictions)
				key_bytes_proba[i] += np.log(min_proba**2)
	# Now we find where our real key candidate lies in the estimation.
	# We do this by sorting our estimates and find the rank in the sorted array.
	sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
	real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
	return (real_key_rank, key_bytes_proba)

def full_ranks(model, dataset, metadata, min_trace_idx, max_trace_idx, rank_step):
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	real_key = metadata[0]['key'][0]
	# Check for overflow
	if max_trace_idx > dataset.shape[0]:
		print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
		sys.exit(-1)
	# Get the input layer shape
	input_layer_shape = model.get_layer(index=0).input_shape
	# Sanity check
	if input_layer_shape[1] != len(dataset[0, :]):
		print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(dataset[0, :])))
		sys.exit(-1)
	# Adapt the data shape according our model input
	if len(input_layer_shape) == 2:
		# This is a MLP
		input_data = dataset[min_trace_idx:max_trace_idx, :]
	elif len(input_layer_shape) == 3:
		# This is a CNN: reshape the data
		input_data = dataset[min_trace_idx:max_trace_idx, :]
		input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
	else:
		print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
		sys.exit(-1)

	# Predict our probabilities
	predictions = model.predict(input_data)

	index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
	f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
	key_bytes_proba = []
	for t, i in zip(index, range(0, len(index))):
		real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, real_key, t-rank_step, t, key_bytes_proba)
		f_ranks[i] = [t - min_trace_idx, real_key_rank]
	return f_ranks


#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

# Check a saved model against one of the ASCAD databases Attack traces
def check_model(model_file, ascad_database, num_traces=2000):
	check_file_exists(model_file)
	check_file_exists(ascad_database)
	# Load profiling and attack data and metadata from the ASCAD database
	(X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(ascad_database, load_metadata=True)
	# Load model
	model = load_sca_model(model_file)
	# We test the rank over traces of the Attack dataset, with a step of 10 traces
	ranks = full_ranks(model, X_attack, Metadata_attack, 0, num_traces, 10)
	# We plot the results
	x = [ranks[i][0] for i in range(0, ranks.shape[0])]
	y = [ranks[i][1] for i in range(0, ranks.shape[0])]
	plt.title('Performance of '+model_file+' against '+ascad_database)
	plt.xlabel('number of traces')
	plt.ylabel('rank')
	plt.grid(True)
	plt.plot(x, y)
	plt.show(block=False)
	plt.savefig(model_file[:-3]+'_check.png')
	plt.figure()