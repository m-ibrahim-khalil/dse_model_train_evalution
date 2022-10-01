import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

import config
import models
from dataprep import generate_train_dataset

def get_model():
 
    if config.model_name == "LSTM":
        model = models.BiLSTM(config.input_shape, config.num_classes)

    # model.summary()
    if config.optimizer_fn == 'sgd':
        optimizer = tf.keras.optimizers.SGD(config.learning_rate, config.momentum)
    elif config.optimizer_fn == 'adam':
        optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.momentum)
    elif config.optimizer_fn == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(config.learning_rate, config.momentum)
    else:
        print("add another optimizer")
    
    model.compile(optimizer= optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  

    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU found: {gpu}")

    # load configs and resolve paths
    result_save_path = os.path.join(config.result_dir, config.model)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    log_dir = os.path.join(result_save_path, "logs_{}".format(config.version))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # get the original_dataset
    train_dataset, valid_dataset = generate_train_dataset()
    model = get_model()
    
    # set the callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    callback_list = [rlrop, tensorboard_callback]

    # start training
    model.fit(train_dataset,
                epochs= config.EPOCHS,
                validation_data=valid_dataset,
                batch_size = config.BATCH_SIZE,
                callbacks=callback_list,
                verbose=1)

    # save model
    model_name="{}_{}_{}".format(config.model, config.version, config.data_file.split()[0])
    model_save_path = os.path.join(result_save_path, model_name)
    model.save(model_save_path, save_format='tf')