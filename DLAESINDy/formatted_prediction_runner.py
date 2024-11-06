# import sys
# import os
# import pandas as pd
# import math
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# from aesindy.solvers import DatasetConstructorMultiEquity, scale_dataset
# from aesindy.training import TrainModel
# from best_params import default_params as params
# from aesindy.helper_functions import call_polygon

# print(params)


# datasets = []

# raw_data = call_polygon("SPY",'2010-01-01','2024-01-01','minute',15)
# raw_data.rename(columns={'c':'x'}, inplace=True)
# raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
# raw_data['time'] = raw_data['date'].dt.time
# # scaled_data = scale_dataset(raw_data, params)
# # datasets.append(scaled_data)


# data_dict = {
#     'x':[raw_data],
#     'dt': 900
#     }

# data_builder = DatasetConstructorMultiEquity(params=params)
# data_builder.build_solution(data_dict)
# train_data = data_builder.get_data()
# print(train_data['x'].shape)    
# trainer = TrainModel(train_data, params)
# trainer.fit() 

# test_raw_data = call_polygon("SPY",'202024-01-01','2024-11-01','minute',15)
# test_raw_data.rename(columns={'c':'x'}, inplace=True)
# test_raw_data['date'] = pd.to_datetime(test_raw_data['date']).dt.tz_localize(None)
# test_raw_data['time'] = test_raw_data['date'].dt.time

# test_data_dict = {
#     'x':[test_raw_data],
#     'dt': 900
# }


import sys
import os
import pandas as pd
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from aesindy.solvers import DatasetConstructorMultiEquity, scale_dataset
from aesindy.training import TrainModel
from best_params import default_params as params
from aesindy.helper_functions import call_polygon

def train_model(train_data_raw):
    """
    Train the model on historical data
    """
    data_dict = {
        'x': [train_data_raw],
        'dt': 900
    }

    data_builder = DatasetConstructorMultiEquity(params=params)
    data_builder.build_solution(data_dict)
    train_data = data_builder.get_data()
    print("Training data shape:", train_data['x'].shape)    
    
    trainer = TrainModel(train_data, params)
    prediction = trainer.fit()
    
    return trainer.model, data_builder

def prepare_prediction_data(test_data_raw, data_builder):
    """
    Prepare new data for prediction using the same preprocessing as training data
    """
    test_data_dict = {
        'x': [test_data_raw],
        'dt': 900
    }
    
    # Use the same data builder to process test data
    data_builder.build_solution(test_data_dict)
    test_data = data_builder.get_data()
    
    # Format data for prediction
    test_x = test_data['x'].T
    test_dx = test_data['dx'].T
    
    if params['svd_dim'] is not None:
        test_xorig = test_data['xorig'].T
        formatted_test_data = [test_xorig, test_x, test_dx]
    else:
        formatted_test_data = [test_x, test_dx]
        
    return formatted_test_data

def generate_predictions(model, test_data, window_size=params['input_dim']):
    """
    Generate predictions using sliding windows
    """
    predictions = []
    
    # Slide through the test data with the specified window size
    for i in range(0, len(test_data[0]) - window_size - params['future_steps'], window_size):
        # Extract window of data
        window_data = [x[i:i+window_size] for x in test_data]
        
        # Make prediction
        prediction = model.predict_future(window_data[0], params['future_steps'])
        predictions.append(prediction)
        
    return np.array(predictions)

if __name__ == "__main__":
    # Training phase
    print("Loading and preparing training data...")
    train_raw_data = call_polygon("SPY", '2010-01-01', '2024-01-01', 'minute', 15)
    train_raw_data.rename(columns={'c':'x'}, inplace=True)
    train_raw_data['date'] = pd.to_datetime(train_raw_data['date']).dt.tz_localize(None)
    train_raw_data['time'] = train_raw_data['date'].dt.time
    
    # Train model
    print("Training model...")
    model, data_builder = train_model(train_raw_data)
    
    # Prediction phase
    print("Loading and preparing test data...")
    test_raw_data = call_polygon("SPY", '2024-01-01', '2024-11-01', 'minute', 15)
    test_raw_data.rename(columns={'c':'x'}, inplace=True)
    test_raw_data['date'] = pd.to_datetime(test_raw_data['date']).dt.tz_localize(None)
    test_raw_data['time'] = test_raw_data['date'].dt.time
    
    test_time_index = test_raw_data['date']
    # Prepare test data using same preprocessing as training
    formatted_test_data = prepare_prediction_data(test_raw_data, data_builder)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, formatted_test_data)
    
    # Save predictions
    prediction_df = pd.DataFrame(predictions)
    prediction_df['date'] = test_time_index
    prediction_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
