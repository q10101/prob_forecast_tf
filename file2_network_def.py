import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# --- 3. Define LSTM Neural Network Architecture ---
class LSTMForecaster(models.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMForecaster, self).__init__()
        self.masking = layers.Masking(mask_value=0.0, input_shape=(None, input_size))
        self.lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.fc = layers.Dense(output_size)

    def call(self, inputs, training=False, initial_state=None):
        masked_inputs = self.masking(inputs)
        lstm_out, h_n, c_n = self.lstm(masked_inputs, initial_state=initial_state)
        output = self.fc(lstm_out)
        return output, (h_n, c_n)

# --- 4. Specify Training Options ---
def create_model_and_optimizer(input_size, hidden_size, output_size):
    model = LSTMForecaster(input_size, hidden_size, output_size)
    model.build(input_shape=(None, None, input_size))
    
    optimizer = optimizers.Adam()
    criterion = losses.MeanSquaredError(reduction=losses.Reduction.NONE)
    
    return model, optimizer, criterion