import os
default_params = {}

default_params['tau'] = None # skip 
default_params['tend'] = 20
default_params['dt'] = 0.001

default_params['system_coefficients'] = None 
default_params['normalization'] = None 
default_params['noise'] = 0.0
default_params['interpolate'] = False
default_params['interp_dt'] = 0.01
default_params['interp_kind'] = 'cubic'
default_params['interp_coefs'] = [21, 3]

default_params['n_ics'] = 5

default_params['train_ratio'] = 0.8
default_params['include_fourier'] = True
default_params['exact_features'] = False # Overrides poly_order

default_params['svd_dim'] = None
default_params['scale'] = False 

default_params['ode_net'] = False
default_params['ode_net_widths'] = [1.5, 3]

# sequential thresholding parameters
default_params['coefficient_initialization'] = 'random_normal'
default_params['fixed_coefficient_mask'] = False
default_params['fix_coefs'] = False
default_params['trainable_auto'] = True 
default_params['sindy_pert'] = 0.0

default_params['activation'] = 'relu'
default_params['use_bias'] = True 

default_params['learning_rate'] = 1e-3
default_params['learning_rate_sched'] = False 

default_params['save_checkpoints'] = False 
default_params['save_freq'] = 1

default_params['print_progress'] = True
default_params['print_frequency'] = 10 
default_params['use_wandb'] = False

# training time cutoffs
default_params['max_epochs'] = 1500
default_params['patience'] = 75
default_params['sparse_weighting'] = None

default_params['use_sindycall'] = True
default_params['sindy_threshold'] = 0.4
default_params['sindy_init_scale'] = 7.0
default_params['future_steps'] = 8  # Number of future steps to predict

default_params['n_time_series'] = 1 # Weight for future prediction loss
default_params['variable_weights'] = [1.0] # Weight for future prediction loss
default_params['smoothing_steps'] = 4
default_params['smoothed_diff'] = False

# PARAMS THAT ARE MODIFIED
default_params['data_length'] = 0
default_params['open_close'] = "open" 
default_params['batch_size'] = 32 
default_params['coefficient_threshold'] = 1e-6 ## set to none for turning off RFE
default_params['input_dim'] = 256 
default_params['latent_dim'] = 3
default_params['learning_rate'] = .0001
# loss function weighting
default_params['loss_weight_rec'] = 0.6
default_params['loss_weight_sindy_z'] = 0.001 
default_params['loss_weight_sindy_x'] = 0.0001
default_params['loss_weight_sindy_regularization'] = 0.1
default_params['loss_weight_integral'] = 0.1
default_params['loss_weight_x0'] = 0.03
default_params['loss_weight_layer_l2'] = 0.1
default_params['loss_weight_layer_l1'] = 0.1
default_params['loss_weight_prediction'] = 1.0
default_params['loss_weight_prediction'] = 0.5
default_params['prediction_mode'] = 'close'
default_params['n_frequencies'] = 2
default_params['smoothing_steps'] = 4
default_params['smoothed_diff'] = False
default_params['sindycall_freq'] = 1500
default_params['poly_order'] = 2
default_params['threshold_frequency'] = 50
default_params['widths_ratios'] = [0.8, 0.6, 0.4, 0.2]