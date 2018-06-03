import os.path
import sys
import h5py
import numpy as np
import random


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


# TODO: sanity checks on the parameters
def extract_traces(traces_file, labeled_traces_file, profiling_index = [n for n in range(0, 2560)], attack_index = [n for n in range(0, 2560, 10)], target_points=[n for n in range(3000, 4000)], profiling_desync=0, attack_desync=0):
    check_file_exists(traces_file)
    check_file_exists(os.path.dirname(labeled_traces_file))
    # Open the raw traces HDF5 for reading
    try:
        in_file  = h5py.File(traces_file, "r")
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
    desync_raw_traces = raw_traces[:, range(min_target_point, max_target_point + 1 + max(profiling_desync, attack_desync))]

    # Apply a desynchronization on the traces if asked to
    # See White Paper
    # In order to desynchronize our traces, we only shift the
    # target_points by the given amount
    # Select the profiling and attack traces and zoom in on the points of interest
    raw_traces_profiling = np.zeros([len(profiling_index), len(target_points)], desync_raw_traces.dtype)
    profiling_desync_metadata = np.zeros(len(profiling_index), np.uint32)
    curr_trace = 0
    for trace in profiling_index:
        # Desynchronize the profiling traces with the asked amount
        r_desync = random.randint(0, profiling_desync)
        profiling_desync_metadata[curr_trace] = r_desync
        curr_point = 0
        for point in map(int.__sub__, target_points, [min_target_point] * len(target_points)):
          raw_traces_profiling[curr_trace, curr_point] = desync_raw_traces[trace, point+r_desync]
          curr_point += 1
        curr_trace += 1

    raw_traces_attack = np.zeros([len(attack_index), len(target_points)], desync_raw_traces.dtype)
    attack_desync_metadata = np.zeros(len(attack_index), np.uint32)
    curr_trace = 0
    for trace in attack_index:
        # Desynchronize the profiling traces with the asked amount
        r_desync = random.randint(0, attack_desync)
        attack_desync_metadata[curr_trace] = r_desync
        curr_point = 0
        for point in map(int.__sub__, target_points, [min_target_point] * len(target_points)):
          raw_traces_attack[curr_trace, curr_point] = desync_raw_traces[trace, point+r_desync]
          curr_point += 1
        curr_trace += 1

    # Compute our labels
    labels_profiling = labelize(raw_plaintexts[profiling_index], raw_keys[profiling_index])
    labels_attack  = labelize(raw_plaintexts[attack_index], raw_keys[attack_index])
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
    profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling, dtype=raw_traces_profiling.dtype)
    attack_traces_group.create_dataset(name="traces", data=raw_traces_attack, dtype=raw_traces_attack.dtype)
    # Labels in the groups
    profiling_traces_group.create_dataset(name="labels", data=labels_profiling, dtype=labels_profiling.dtype)
    attack_traces_group.create_dataset(name="labels", data=labels_attack, dtype=labels_attack.dtype)
    # Put the metadata (plaintexts, keys, ...) so that one can check the key rank
    metadata_type = np.dtype([
            ("plaintext", raw_plaintexts.dtype, (16,)),
            ("key", raw_keys.dtype, (16,)),
            ("desync", np.uint32, (1,)),
           ])
    profiling_metadata = np.array([(raw_plaintexts[n], raw_keys[n], profiling_desync_metadata[k]) for n, k  in zip(profiling_index, range(0, len(profiling_desync_metadata)))], dtype=metadata_type)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)
    attack_metadata = np.array([(raw_plaintexts[n], raw_keys[n], attack_desync_metadata[k]) for n, k in zip(attack_index, range(0, len(attack_desync_metadata)))], dtype=metadata_type)
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