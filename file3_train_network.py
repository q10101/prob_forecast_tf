import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# --- 5. Train Recurrent Neural Network ---
def train_network(model, optimizer, criterion, train_dataset_, num_epochs=200):
    train_losses = []
    all_iteration_losses = []
    
    print("\n--- Training LSTM Neural Network ---")
    train_start_time = time.time()
    train_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at: {train_start_datetime}")

    @tf.function
    def train_step(x_batch_, t_batch_):
        with tf.GradientTape() as tape:
            predictions, _ = model(x_batch_, training=True)
            per_element_loss = tf.square(t_batch_ - predictions)
            mask = tf.cast(tf.reduce_sum(tf.abs(t_batch_), axis=-1) != 0, dtype=tf.float32)
            expanded_mask = tf.expand_dims(mask, axis=-1)
            masked_loss = per_element_loss * expanded_mask
            loss_ = tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-8)
        gradients = tape.gradient(loss_, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_

    for epoch in range(num_epochs):
        epoch_loss_sum = 0
        num_batches = 0
        for X_batch, T_batch in train_dataset_:
            num_batches += 1
            loss = train_step(X_batch, T_batch)
            all_iteration_losses.append(loss.numpy())
            epoch_loss_sum += loss.numpy()
        avg_epoch_loss = epoch_loss_sum / num_batches
        train_losses.append(avg_epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Batch Loss: {avg_epoch_loss:.4f}')

    train_end_time = time.time()
    training_elapsed_time = train_end_time - train_start_time
    print(f"Training finished in: {training_elapsed_time:.2f} seconds")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title(f"Training Loss per Epoch\nStart: {train_start_datetime}, Elapsed: {training_elapsed_time:.2f}s")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

    # Plot all iteration losses
    plt.figure(figsize=(12, 6))
    plt.plot(all_iteration_losses, label='Iteration Loss', alpha=0.7)
    plt.title(f"Training Loss per Iteration\nStart: {train_start_datetime}, Elapsed: {training_elapsed_time:.2f}s")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
    
    return train_losses, all_iteration_losses

# Example usage (assuming you have the necessary imports and data from file1 and file2):
# from file1_data_preparation import train_dataset_, num_channels
# from file2_network_definition import create_model_and_optimizer
# 
# input_size = num_channels
# hidden_size = 128
# output_size = num_channels
# 
# model, optimizer, criterion = create_model_and_optimizer(input_size, hidden_size, output_size)
# model.summary()
# 
# train_losses, all_iteration_losses = train_network(model, optimizer, criterion, train_dataset_, num_epochs=200)