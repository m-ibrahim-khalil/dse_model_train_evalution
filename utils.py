import os
import numpy as np
import tensorflow as tf
from config import *
from dataprep import generate_test_dataset

def evaluate():
    # get the test_dataset
    test_data_config = (test_dir)
    X_test, Y_test = generate_test_dataset(test_data_config)     

    result_save_path = os.path.join(result_dir, model_name)
    model_name = "{}_{}_{}".format(model_name, version, data_file.split()[0])
    model_save_path = os.path.join(result_save_path, model_name)
   
    loaded_model = tf.keras.models.load_model(model_save_path)
    
    predictions = loaded_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean((predictions - Y_test)**2))
    accuracy = loaded_model.evaluate((X_test, Y_test))

    return rmse, accuracy