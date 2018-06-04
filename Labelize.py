import os.path
import sys
import h5py
import numpy as np
import random
from sklearn.model_selection import train_test_split

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

# Our labelization function:
# It is as simple as the computation of the result of Sbox(p[3] + k[3]) (see the White Paper)
# Note: you can of course adapt the labelization here (say if you want to attack the first byte Sbox(p[0] + k[0])
# or if you want to attack another round of the algorithm).


def labelize(plaintexts, keys):
    return keys[:, 0] ^ plaintexts[:, 0]


def extract_traces(traces_file, labeled_traces_file, test_size=0.1, target_points=[n for n in range(3000, 4000)],
                   desync=0):
    check_file_exists(traces_file)
    check_file_exists(os.path.dirname(labeled_traces_file))
    # Open the raw traces HDF5 for reading
    try:
        in_file = h5py.File(traces_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % traces_file)
        sys.exit(-1)

    raw_traces = in_file['traces']
    raw_plaintexts = in_file['texts']
    raw_ciphertexts = in_file['ciphertexts']
    raw_keys = in_file['keys']

    # Extract a larger set of points to handle desynchronization
    min_target_point = min(target_points)
    max_target_point = max(target_points)
    desync_raw_traces = raw_traces[:, range(min_target_point, max_target_point + 1 + desync)]

    # Apply a desynchronization on the traces if asked to
    # See White Paper
    # In order to desynchronize our traces, we only shift the
    # target_points by the given amount
    # Select the profiling and attack traces and zoom in on the points of interest
    desync_traces = np.zeros([len(raw_traces), len(target_points)], desync_raw_traces.dtype)
    desync_metadata = np.zeros(len(raw_traces), np.uint32)
    curr_trace = 0
    for trace in range(len(raw_traces)):
        # Desynchronize the profiling traces with the asked amount
        r_desync = random.randint(0, desync)
        desync_metadata[curr_trace] = r_desync
        curr_point = 0
        for point in map(int.__sub__, target_points, [min_target_point] * len(target_points)):
            desync_traces[curr_trace, curr_point] = desync_raw_traces[trace, point + r_desync]
            curr_point += 1
        curr_trace += 1

    # Compute our labels
    labels = labelize(raw_plaintexts, raw_keys)

    #     print(desync_traces.shape, labels.shape, desync_metadata.shape, raw_keys.shape, raw_plaintexts.shape, desync_traces.shape, )

    i_train, i_test = train_test_split(range(len(desync_traces)), test_size=test_size)

    i_train = sorted(i_train)
    i_test = sorted(i_test)

    # Open the output labeled file for writing
    try:
        out_file = h5py.File(labeled_traces_file, "w")
    except:
        print("Error: can't open HDF5 file '%s' for writing ..." % labeled_traces_file)
        sys.exit(-1)
    # Create our HDF5 hierarchy in the output file:
    # 	- Profilinging traces with their labels
    #	- Attack traces with their labels

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")
    # Datasets in the groups
    profiling_traces_group.create_dataset(name="traces", data=desync_traces[i_train], dtype=desync_traces.dtype)
    attack_traces_group.create_dataset(name="traces", data=desync_traces[i_test], dtype=desync_traces.dtype)
    # Labels in the groups
    profiling_traces_group.create_dataset(name="labels", data=labels[i_train], dtype=labels.dtype)
    attack_traces_group.create_dataset(name="labels", data=labels[i_test], dtype=labels.dtype)
    # Put the metadata (plaintexts, keys, ...) so that one can check the key rank
    metadata_type = np.dtype([
            ("plaintext", raw_plaintexts.dtype, (16,)),
            ("key", raw_keys.dtype, (16,)),
            ("desync", np.uint32, (1,)),
           ])
    profiling_metadata = np.array([(raw_plaintexts[i], raw_keys[i], desync_metadata[i]) for i in i_train], dtype=metadata_type)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)
    attack_metadata = np.array([(raw_plaintexts[i], raw_keys[i], desync_metadata[i]) for i in i_test], dtype=metadata_type)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type)

    out_file.flush()
    out_file.close()
    
#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_labelized(ascad_database_file, load_metadata=False):
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