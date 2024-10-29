""" Once a dataset is generated using `snp_generation.py` regroup time-series 
into batch files to make loading faster. 
Once generated, it is recommended to call `snp_generation_batched.py` to store 
data by large batches in order to reduce the number of files to load.
"""
import os 
from pathlib import Path
from tqdm import tqdm
import numpy as np


def batch_npy_files(input_directory, batch_size, output_directory):
    """ Group .npy files in the input directory into batches.

    Args:
        input_directory (str): Path to the directory containing .npy files.
        batch_size (int): Number of .npy files per batch.
        output_directory (str): Path to the directory where batches will be saved.
    """
    # ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    npy_files.sort()

    # group the files into batches
    for i in tqdm(range(0, len(npy_files), batch_size)):
        batch_files = npy_files[i:i+batch_size]
        batch_data = []

        # load each .npy file in the batch
        for file in batch_files:
            data = np.load(input_directory/file)
            batch_data.append(data)

        # save the batch as a new .npy file
        batch_array = np.concatenate(batch_data)  # Convert list to numpy array
        batch_path = output_directory / f'batch{i//batch_size+1:04}.npy'
        np.save(batch_path, batch_array)

if __name__ == "__main__":

    # script arguments 
    batch_size = 256
    input_dir = Path(__file__).parents[1] / '_cache' / 'snp_generation'
    output_dir = Path(__file__).parents[1] / '_cache' / 'snp_generation_batched'    # Change this to your output directory

    batch_npy_files(input_dir, batch_size, output_dir)

    print("FINISHED")