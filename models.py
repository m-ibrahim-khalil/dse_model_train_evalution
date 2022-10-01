import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout


class BiLSTM(Model):
  def __init__(self, input_shape, num_classes):
    super(BiLSTM, self).__init__(name='BiLSTM')
    self.input_layer = Bidirectional(LSTM(50, return_sequences=True, input_shape = input_shape))
    self.bi_lstm = Bidirectional(LSTM(50, return_sequences=False))
    self.hidden_layer = Dense(25, activation= 'relu')
    self.output_layer = Dense(num_classes, activation= 'relu')
  
  def call(self, input_tensor, training=False):
    x = self.input_layer(input_tensor)
    x = Dropout(0.5)
    x = self.bi_lstm(x)
    x = Dropout(0.5)
    x = self.hidden_layer(x)
    x = self.output_layer(x)
    return x
