import tensorflow as tf
# from file1_data_prep import prepare_data, num_channels, train_dataset, test_dataset, sigmaT, muT, sigmaX, muX
import file1_data_prep as gd
from file2_network_def import create_model_and_optimizer
from file3_train_network import train_network
from file4_test_lstm import test_lstm
from file5_forecast_lstm import forecast_lstm

# num_channels, train_dataset, test_dataset, sigmaT, muT, sigmaX, muX

# Set device to GPU if available, otherwise CPU
print(f"TensorFlow Version: {tf.__version__}")
# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and being used.")
else:
    print("GPU is not available, using CPU.")

# --- Main Execution ---
def main():
    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    gd.prepare_data()

    # Step 2: Define network
    print("\nStep 2: Defining network...")
    input_size = gd.num_channels
    hidden_size = 128
    output_size = gd.num_channels

    model, optimizer, criterion = create_model_and_optimizer(input_size, hidden_size, output_size)
    model.summary()
    
    # Step 3: Train network
    print("\nStep 3: Training network...")
    train_losses, all_iteration_losses = train_network(model, optimizer, criterion, gd.train_dataset, num_epochs=200)
    
    print("\nTraining completed successfully!")

    # Step4: Test network
    test_lstm(model, gd.test_dataset, gd.num_channels, gd.sigmaT, gd.muT, gd.sigmaX, gd.muX)

    # Step5: Forcast openloop and closedloop
    idx_to_forecast = 1
    open_loop_offset = 75
    num_closed_loop_steps = 200
    # data_test <> test_dataset
    forecast_lstm(model, gd.data_test, idx_to_forecast, gd.num_channels, gd.sigmaT, gd.muT, gd.sigmaX, gd.muX,
                  open_loop_offset, num_closed_loop_steps)

    return model, train_losses, all_iteration_losses

if __name__ == "__main__":
    model, train_losses, all_iteration_losses = main()