
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from default_params import default_params
import numpy as np
from aesindy.solvers import DatasetConstructorMultiEquity, scale_dataset
from aesindy.training import TrainModel
from aesindy.helper_functions import call_polygon
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import gc
import warnings 
import pandas as pd
warnings.filterwarnings("ignore")


def update_params_from_wandb(params, wandb_config):
    """
    Update the default parameters with values from the wandb config.
    
    Args:
    default_params (dict): The default parameter dictionary
    wandb_config (wandb.config): The wandb config object
    
    Returns:
    dict: Updated parameter dictionary
    """
    updated_params = params.copy()
    
    for key, value in wandb_config.items():
        if key in updated_params:
            updated_params[key] = value
        else:
            print(f"Warning: '{key}' found in wandb config but not in default params. Ignoring.")
    
    return updated_params



def model_runner(wandb_params, input_data):
    params = update_params_from_wandb(default_params, wandb_params)
    params['model'] = 'multi'
    params['case'] = 'hyp'
    params['use_wandb'] = True
    params['prediciton_mode'] = 'max'
    params['future_steps'] = 24
    params['open_close'] = 'open'
    
    # scaled_datasets = []
    # for data in input_data:
    data = input_data[int(params['data_length']*len(input_data)):]
        # scaled_data = scale_dataset(data, params)
        # scaled_datasets.append(scaled_data)
    ## slice the data based on a fractional proportion, must remain in sequential order
    # print(scaled_datasets[0])
    data_dict = {
        'x':data,
        'dt': 900
        }
    data_builder = DatasetConstructorMultiEquity(
                    params=params,
                    )
    data_builder.build_solution(data_dict)
    train_data = data_builder.get_data()
    trainer = TrainModel(train_data, params)
    trainer.fit() 

    del trainer.model
    del trainer

    tf.keras.backend.clear_session()
    gc.collect()

def wandb_sweep(data):


    # Hyperparameters
    sweep_config = {
        'method': 'bayes', 
        'bayes': {
            'bayes_initial_samples': 75,
            'exploration_factor': 0.2,
        },
        'metric': {
            'name': 'current_best_val_prediction_loss',
            'goal': 'minimize'
        },
        "parameters": {
            "learning_rate": {'values': [0.001,.0001]},
            "latent_dim": {'values': [2,3]},
            "input_dim": {'values': [64,128,256]},
            "poly_order": {'values': [2,3]},
            "n_frequencies": {'values': [2,3,4]},
            "loss_weight_layer_l2": {'values': [0.05,0.1]},
            "loss_weight_layer_l1": {'values': [0.05,0.1]},
            "loss_weight_x0": {'values': [0.03,0.05]},
            "loss_weight_integral": {'values': [0.01,0.05,0.1]},
            "loss_weight_sindy_regularization": {'values': [1e-5,1e-3,1e-1]},
            "loss_weight_prediction": {'values': [0.5,1.0]},
            "loss_weight_rec": {'values': [0.3,0.6,0.9]},
            "loss_weight_sindy_z": {'values': [0.001,0.0001]},
            "loss_weight_sindy_x": {'values': [0.0003,0.0001]},
            "batch_size": {'values': [32,64]},
            "data_length": {'values': [0,0.25,0.5,0.75]},
            "widths_ratios": {'values': [[0.5,0.25],[0.75,0.5,0.25],[0.8,0.6,0.4,0.2]]},
            "sindy_threshold": {'values': [0.05,0.1,0.2]},
            "sindy_init_scale": {'values': [3.0,5.0]},
            "threshold_frequency": {'values': [50,100,500]},
            "coefficient_threshold": {'values': [1,2]},
            "sindycall_freq": {'values': [250,500,1500]},
            "smoothing_steps": {'values': [4,8]},
            "smoothed_diff": {'values': [False]},
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="DLAESINDy")

    # Define the objective function for the sweep
    def sweep_train():

        wandb.init(project="DLAESINDy", config=wandb.config)
        model_runner(wandb.config, data)

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=1000) 


if __name__ == '__main__':
    datasets = []
    # idxbig = ["QQQ","SPY","NVDA","IWM","AAPL","TSLA","MSFT","AMD","AMZN","META","GOOGL"]
    # for symbol in idxbig:
    raw_data = call_polygon("SPY",'2010-01-01','2024-09-01','minute',15)
    # datasets.append(raw_data)
    # scaled_data = raw_data.copy()
    # scaled_data = scale_dataset(scaled_data, params={'input_dim': 256,'future_steps': 10})
    raw_data.rename(columns={'c':'x'}, inplace=True)
    raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
    raw_data['time'] = raw_data['date'].dt.time
    wandb_sweep([raw_data])