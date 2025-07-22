import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 7. Forecast Future Time Steps ---
def forecast_lstm(model, data_test, idx_to_forecast, num_channels, sigmaT, muT, sigmaX, muX, open_loop_offset=75, num_closed_loop_steps=200):
    full_test_sequence_raw = data_test[idx_to_forecast]
    X_initial_raw = full_test_sequence_raw[:-1, :]
    T_true_future_raw = full_test_sequence_raw[1:, :]
    X_initial_norm = (X_initial_raw - muX) / sigmaX
    X_initial_tensor = tf.constant(X_initial_norm, dtype=tf.float32)[tf.newaxis, :, :]

    print(f"\n--- Forecasting using sequence at index {idx_to_forecast} from test data ---")

    # Open Loop Forecasting
    print("\nPerforming Open Loop Forecasting with explicit state update (normalized values in loop)...")
    X_prime_raw = X_initial_raw[:open_loop_offset, :]
    X_prime_norm = (X_prime_raw - muX) / sigmaX
    X_prime_tensor = tf.constant(X_prime_norm, dtype=tf.float32)[tf.newaxis, :, :]
    predictions_prime_norm_tensor, (h_n, c_n) = model(X_prime_tensor, training=False)
    open_loop_current_states = [h_n, c_n]
    Y_open_loop_norm = [predictions_prime_norm_tensor[0, -1:, :].numpy()]
    remaining_inputs_raw = X_initial_raw[open_loop_offset:, :]

    for t in range(remaining_inputs_raw.shape[0]):
        Xt_raw = remaining_inputs_raw[t, :]
        Xt_norm = (Xt_raw - muX) / sigmaX
        Xt_tensor = tf.constant(Xt_norm, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        next_pred_norm_tensor, (h_n, c_n) = model(Xt_tensor, training=False, initial_state=open_loop_current_states)
        open_loop_current_states = [h_n, c_n]
        Y_open_loop_norm.append(next_pred_norm_tensor[0, -1:, :].numpy())

    Y_open_loop_norm = np.vstack(Y_open_loop_norm)
    Y_open_loop_new_denorm = Y_open_loop_norm * sigmaT + muT

    plt.figure(figsize=(16, 12))
    num_rows = (num_channels + 1) // 2
    if num_channels == 1: num_rows = 1

    for channel_idx in range(num_channels):
        plt.subplot(num_rows, 2, channel_idx + 1)
        plt.plot(np.arange(X_initial_raw.shape[0]), X_initial_raw[:, channel_idx], 'b-', label='Observed (Input)')
        true_future_segment = full_test_sequence_raw[open_loop_offset:open_loop_offset + Y_open_loop_new_denorm.shape[0], channel_idx]
        plt.plot(np.arange(open_loop_offset, open_loop_offset + Y_open_loop_new_denorm.shape[0]), true_future_segment, 'g-', label='True Future')
        plt.plot(np.arange(open_loop_offset, open_loop_offset + Y_open_loop_new_denorm.shape[0]), Y_open_loop_new_denorm[:, channel_idx], 'c--', label='Open-Loop Prediction (Stateful)')
        plt.axvline(x=open_loop_offset - 1, color='k', linestyle=':', label='Last Priming Time Step')
        plt.title(f"Open-Loop Forecasting (Channel {channel_idx + 1}) for Test Sequence {idx_to_forecast + 1}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Open-Loop Forecasting for All Channels (Stateful)", y=1.02, fontsize=16)
    #plt.show(block=False)
    plt.savefig("output/5-openloop-forecast.png")
    plt.pause(0.1)

    # Closed Loop Forecasting
    print("\nPerforming Closed Loop Forecasting...")
    _, (h_n_initial, c_n_initial) = model(X_initial_tensor, training=False)
    last_states = [h_n_initial, c_n_initial]
    current_input_denorm = X_initial_raw[-1:, :]
    current_input_norm = (current_input_denorm - muX) / sigmaX
    current_input_tensor = tf.constant(current_input_norm, dtype=tf.float32)[tf.newaxis, :, :]
    first_prediction_norm_tensor, (h_n_next, c_n_next) = model(current_input_tensor, training=False, initial_state=last_states)
    Y_closed_loop_norm = [first_prediction_norm_tensor[0, -1:, :].numpy()]
    current_states = [h_n_next, c_n_next]
    current_input_tensor = first_prediction_norm_tensor

    for t in range(1, num_closed_loop_steps):
        next_prediction_norm_tensor, (h_n_loop, c_n_loop) = model(current_input_tensor, training=False, initial_state=current_states)
        current_states = [h_n_loop, c_n_loop]
        Y_closed_loop_norm.append(next_prediction_norm_tensor[0, -1:, :].numpy())
        current_input_tensor = next_prediction_norm_tensor

    Y_closed_loop_norm = np.vstack(Y_closed_loop_norm)
    Y_closed_loop_denorm = Y_closed_loop_norm * sigmaT + muT

    plt.figure(figsize=(16, 12))
    num_rows = (num_channels + 1) // 2
    if num_channels == 1: num_rows = 1

    for channel_idx in range(num_channels):
        plt.subplot(num_rows, 2, channel_idx + 1)
        plt.plot(np.arange(X_initial_raw.shape[0]), X_initial_raw[:, channel_idx], 'b-', label='Observed (Input)')
        true_future_start_idx = X_initial_raw.shape[0]
        true_future_end_idx = min(true_future_start_idx + num_closed_loop_steps, full_test_sequence_raw.shape[0])
        if true_future_end_idx > true_future_start_idx:
            plt.plot(np.arange(true_future_start_idx, true_future_end_idx), full_test_sequence_raw[true_future_start_idx:true_future_end_idx, channel_idx], 'g-', label='True Future (if available)')
        plt.plot(np.arange(X_initial_raw.shape[0], X_initial_raw.shape[0] + num_closed_loop_steps), Y_closed_loop_denorm[:, channel_idx], 'm--', label='Closed-Loop Prediction')
        plt.axvline(x=X_initial_raw.shape[0] - 1, color='k', linestyle=':', label='Last Observed Time Step')
        plt.title(f"Closed-Loop Forecasting (Channel {channel_idx + 1}) for Test Sequence {idx_to_forecast + 1}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Closed-Loop Forecasting for All Channels", y=1.02, fontsize=16)
    #plt.show(block=False)
    plt.savefig("output/5-closedloop-forecast.png")
    plt.pause(0.1)