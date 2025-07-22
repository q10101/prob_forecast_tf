import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io
import os

# Global variables that will be set after data preparation
data_train = None
data_test = None
num_channels = None
train_dataset = None
test_dataset = None
sigmaX = None
muX = None
sigmaT = None
muT = None

# --- 1. Load Data (Synthetic data generation or .mat file) ---
def generate_synthetic_data(num_observations=1000, min_timesteps=50, max_timesteps=150, num_channels=3):
    """Generates synthetic waveform data for demonstration.
    Each sequence is numTimeSteps-by-numChannels.
    """
    data = []
    for _ in range(num_observations):
        num_time_steps = np.random.randint(min_timesteps, max_timesteps + 1)
        time = np.linspace(0, 2 * np.pi, num_time_steps)
        sequence = np.zeros((num_time_steps, num_channels))
        for i in range(num_channels):
            amplitude = np.random.rand() * 2 + 1
            frequency = np.random.rand() * 2 + 0.5
            phase = np.random.rand() * 2 * np.pi
            noise = np.random.randn(num_time_steps) * 0.1
            sequence[:, i] = amplitude * np.sin(frequency * time + phase) + noise
        data.append(sequence.astype(np.float32))
    return data

def read_waveform_data_from_mat(file_path='WaveformData.mat'):
    """
    Reads waveform data from a .mat file.
    Args:
        file_path (str): The path to the WaveformData.mat file.
    Returns:
        list: A list of NumPy arrays, where each array represents a sequence
              of waveform data (numTimeSteps-by-numChannels).
              Returns an empty list if the file cannot be loaded or
              the expected data key is not found.
    """
    try:
        mat_contents = scipy.io.loadmat(file_path)
        if 'data' in mat_contents:
            loaded_data = mat_contents['data']
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                if loaded_data.ndim > 1:
                    loaded_data = loaded_data.flatten()
                return [seq.astype(np.float32) for seq in loaded_data if isinstance(seq, np.ndarray)]
            else:
                print(f"Warning: Data found under 'data' key is not in the expected format (NumPy object array).")
                if isinstance(loaded_data, np.ndarray):
                    return [loaded_data.astype(np.float32)]
                return []
        else:
            print(f"Error: 'data' key not found in {file_path}. Please check the .mat file structure.")
            return []
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the .mat file: {e}")
        return []

# --- 2. Prepare Data for Training ---
def prepare_data_for_training(data_set):
    X_set = []
    T_set = []
    for sequence in data_set:
        X_set.append(sequence[:-1, :])
        T_set.append(sequence[1:, :])
    return X_set, T_set

# Function to create a TensorFlow Dataset, handling padding.
def create_tf_dataset(X_data, T_data, batch_size, shuffle=True):
    def pad_sequence_left(sequence, max_len, padding_value=0.0):
        current_len = tf.shape(sequence)[0]
        padding_needed = max_len - current_len
        padded_sequence = tf.pad(sequence, [[padding_needed, 0], [0, 0]], "CONSTANT", constant_values=padding_value)
        return padded_sequence

    dataset = tf.data.Dataset.from_generator(
        lambda: ((X, T) for X, T in zip(X_data, T_data)),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, num_channels]), tf.TensorShape([None, num_channels]))
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X_data))

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(tf.TensorShape([None, num_channels]), tf.TensorShape([None, num_channels])),
        padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # for X_batch, T_batch in dataset:
    #     num_batches += 1

    return dataset

def prepare_data():
    """Main function to prepare all data and set global variables."""
    global data_train, data_test, num_channels, train_dataset, test_dataset, sigmaT, muT, sigmaX, muX
    
    # Use synthetic data if WaveformData.mat is not found, otherwise try to load it
    mat_file_path = 'WaveformData.mat'
    if os.path.exists(mat_file_path):
        data = read_waveform_data_from_mat(mat_file_path)
        if not data:
            print(f"Could not load data from {mat_file_path}, generating synthetic data instead.")
            data = generate_synthetic_data(num_observations=1000, num_channels=3)
    else:
        print(f"'{mat_file_path}' not found, generating synthetic data.")
        data = generate_synthetic_data(num_observations=1000, num_channels=3)

    if not data:
        raise RuntimeError("No data available for processing. Exiting.")

    num_channels = data[0].shape[1]
    print(f"Loaded {len(data)} observations with {num_channels} channels.")

    # Visualize the first few sequences
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(min(4, len(data))):
        ax = axes[i]
        for channel in range(num_channels):
            ax.plot(data[i][:, channel], label=f'Channel {channel + 1}')
        ax.set_xlabel("Time Step")
        ax.set_title(f"Sequence {i + 1}")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.suptitle("Example Waveform Sequences", y=1.02)
    #plt.show(block=False)
    plt.savefig("output/1-example-waveform.png")
    plt.pause(0.1)

    # Partition data into training and test sets
    data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)

    print(f"Training observations: {len(data_train)}")
    print(f"Test observations: {len(data_test)}")

    XTrain_raw, TTrain_raw = prepare_data_for_training(data_train)
    XTest_raw, TTest_raw = prepare_data_for_training(data_test)

    # Normalize the predictors and targets.
    all_XTrain_concatenated = np.vstack(XTrain_raw)
    all_TTrain_concatenated = np.vstack(TTrain_raw)

    muX = np.mean(all_XTrain_concatenated, axis=0)
    sigmaX = np.std(all_XTrain_concatenated, axis=0)
    sigmaX[sigmaX == 0] = 1.0

    muT = np.mean(all_TTrain_concatenated, axis=0)
    sigmaT = np.std(all_TTrain_concatenated, axis=0)
    sigmaT[sigmaT == 0] = 1.0

    XTrain = [(seq - muX) / sigmaX for seq in XTrain_raw]
    TTrain = [(seq - muT) / sigmaT for seq in TTrain_raw]
    XTest = [(seq - muX) / sigmaX for seq in XTest_raw]
    TTest = [(seq - muT) / sigmaT for seq in TTest_raw]

    train_dataset = create_tf_dataset(XTrain, TTrain, batch_size=32, shuffle=True)
    test_dataset = create_tf_dataset(XTest, TTest, batch_size=1, shuffle=False)
    
    print("Data preparation completed successfully!")