import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 6. Test Recurrent Neural Network ---
def test_lstm(model, test_dataset, num_channels, sigmaT, muT, sigmaX, muX):
    print("\n--- Testing LSTM Neural Network ---")
    test_rmse_errors = []
    all_test_predictions_denorm = []
    all_test_targets_denorm = []
    all_test_inputs_original = []

    for X_batch_padded, T_batch_padded in test_dataset:
        Y_pred_padded, _ = model(X_batch_padded, training=False)
        actual_sequence_lengths = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(X_batch_padded), axis=-1) != 0, tf.int32), axis=-1)
        sequence_length = actual_sequence_lengths[0].numpy()
        Y_pred_norm = Y_pred_padded[0, :sequence_length, :].numpy()
        T_true_norm = T_batch_padded[0, :sequence_length, :].numpy()
        Y_pred_denormalized = Y_pred_norm * sigmaT + muT
        T_true_denormalized = T_true_norm * sigmaT + muT
        X_input_original_norm = X_batch_padded[0, :sequence_length, :].numpy()
        X_input_original_denorm = X_input_original_norm * sigmaX + muX
        rmse_val = np.sqrt(np.mean((Y_pred_denormalized - T_true_denormalized) ** 2))
        test_rmse_errors.append(rmse_val)
        all_test_predictions_denorm.append(Y_pred_denormalized)
        all_test_targets_denorm.append(T_true_denormalized)
        all_test_inputs_original.append(X_input_original_denorm)

    plt.figure(figsize=(8, 6))
    plt.hist(test_rmse_errors, bins=20, edgecolor='black')
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.title("Test RMSE Errors Histogram")
    plt.grid(True)
    #plt.show(block=False)
    plt.savefig("output/4-rmse-errors.png")
    plt.pause(0.1)

    mean_rmse = np.mean(test_rmse_errors)
    print(f"Mean RMSE on test data: {mean_rmse:.4f}")

    # return test_rmse_errors, all_test_predictions_denorm, all_test_targets_denorm, all_test_inputs_original