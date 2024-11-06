
import sys
import os
import pandas as pd
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from aesindy.solvers import DatasetConstructorMultiEquity, scale_dataset
from aesindy.training import TrainModel
from default_params import default_params as params
from aesindy.helper_functions import call_polygon

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


params['model'] = 'spy'
params['case'] = '1hr_3rd_dim64_ld3_sine_x001'
params['poly_order'] = 2
params['include_fourier'] = True
params['n_frequencies'] = 4
params['fix_coefs'] = False
params['svd_dim'] = None
params['latent_dim'] = 2
params['scale'] = False
params['input_dim'] = 64
params['save_checkpoints'] = True 
params['save_freq'] = 5 
params['print_progress'] = True
params['print_frequency'] = 10
# data preperation
params['train_ratio'] = 0.8
params['test_ratio'] = 0.15
# training time cutoffs
params['max_epochs'] = 1500
params['patience'] = 100
# loss function weighting
params['loss_weight_rec'] = 0.3
params['loss_weight_sindy_z'] = 0.001
params['loss_weight_sindy_x'] = 0.0001
params['loss_weight_sindy_regularization'] = 1e-05
params['loss_weight_integral'] = 0.05
params['loss_weight_x0'] = 0.03
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 
params['loss_weight_prediction'] = 1.0
params['widths_ratios'] = [0.75, 0.5, 0.25]
params['use_wandb'] = False
params['coefficient_threshold'] = 1 ## set to none for turning off RFE
params['sindy_threshold'] = 0.3
params['threshold_frequency'] = 80
params['use_sindycall'] = True
params['sindy_init_scale'] = 3.0
params['sindycall_freq'] = 55
params['future_steps'] = 8  # Number of future steps to predict
params['loss_weight_prediction'] = 1.0  # Weight for future prediction loss
params['prediction_mode'] = 'max'
params['batch_size'] = 64
params['smoothing_steps'] = 4
params['smoothed_diff'] = False
params['open_close'] = 'open'

print(params)


datasets = []

raw_data = call_polygon("SPY",'2016-01-01','2024-09-01','minute',15)
raw_data.rename(columns={'c':'x'}, inplace=True)
raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
raw_data['time'] = raw_data['date'].dt.time
# scaled_data = scale_dataset(raw_data, params)
# datasets.append(scaled_data)


data_dict = {
    'x':[raw_data],
    'dt': 900
    }

data_builder = DatasetConstructorMultiEquity(params=params)
data_builder.build_solution(data_dict)
train_data = data_builder.get_data()
# print(train_data['x'].shape)    
trainer = TrainModel(train_data, params)
trainer.fit() 