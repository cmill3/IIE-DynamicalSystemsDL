import numpy as np
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from scipy import interpolate
from scipy.signal import savgol_filter
from .dynamical_models import get_model
from .helper_functions import get_hankel
from tqdm import tqdm
import pdb
from pysindy.differentiation import SmoothedFiniteDifference    
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DatasetConstructor:
    def __init__(self, 
                input_dim=128,
                interp_dt=0.01,
                savgol_interp_coefs=[21, 3],
                interp_kind='cubic',
                future_steps=10,
                smoothing_steps=4):

        self.input_dim = input_dim
        self.interp_dt = interp_dt 
        self.savgol_interp_coefs = savgol_interp_coefs
        self.interp_kind = interp_kind
        self.future_steps = future_steps
        self.smoothing_steps = smoothing_steps

    def get_data(self):
        return {
            't': self.t,
            'x': self.x,
            'dx': self.dx,
            'z': self.z,
            'dz': self.dz,
            'sindy_coefficients': self.sindy_coefficients
        }
    
    def build_solution(self, data):
        dt = data['dt']
        if 'time' in data.keys():
            times = data['time']
        elif 'dt' in data.keys():
            times = []
            for xr in data['x']:
                times.append(np.linspace(0, dt*len(xr), len(xr), endpoint=False))
        
        x = data['x'][0]
        print(f"len(x): {len(x)}")
        if 'dx' in data.keys():
            dx = data['dx']
        else:
            sfd = SmoothedFiniteDifference(smoother_kws={'window_length': self.smoothing_steps})
            dx = [sfd._differentiate(xr, dt) for xr in x]
        
                    
        n = self.input_dim 
        if self.future_steps > 0:
            n += self.future_steps
        n_delays = n
        xic = []
        dxic = []
        for j, xr in enumerate(x):
            print(j)
            print(len(xr))
            print(xr)
            n_steps = len(xr) - n
            print(f"n_steps: {n_steps}")
            xj = np.zeros((n_steps, n_delays))
            dxj = np.zeros((n_steps, n_delays))
            for k in range(n_steps):
                xj[k, :] = xr[k:n_delays+k]
                dxj[k, :] = dx[j][k:n_delays+k]
            xic.append(xj)
            dxic.append(dxj)
        H = np.vstack(xic)
        dH = np.vstack(dxic)
        
        self.t = np.hstack(times)
        self.x = H.T
        self.dx = dH.T
        self.z = np.hstack(x) 
        self.dz = np.hstack(dx)
        self.sindy_coefficients = None # unused

class DatasetConstructorMultiEquity:
    def __init__(self, 
                params):

        self.params = params

    def get_data(self):
        return {
            't': self.t,
            'x': self.x,
            'dx': self.dx,
            'z': self.z,
            'dz': self.dz,
            'sindy_coefficients': self.sindy_coefficients
        }
    
    def build_solution(self, data):
        dt = data['dt']
        if 'time' in data.keys():
            times = data['time']
        elif 'dt' in data.keys():
            times = []
            for xr in data['x']:
                times.append(np.linspace(0, dt*len(xr), len(xr), endpoint=False))
        
        xic = []
        dxic = []
        datasets = data['x']
        for dataset in datasets:
            x = dataset['x'].values
            if 'dx' in dataset.keys():
                dx = dataset['dx']
            else:
                if self.params['smoothed_diff'] is False:
                    # Ensure x is properly shaped for gradient
                    x_arr = np.array(x)
                    dx = np.gradient(x_arr, dt)

                else:
                    sfd = SmoothedFiniteDifference(smoother_kws={'window_length': self.params['smoothing_steps']})
                    # Reshape 1D array to 2D column vector
                    x_reshaped = x.reshape(-1, 1)
                    # Perform differentiation
                    dx = sfd._differentiate(x_reshaped, dt)
                    # Reshape back to original shape if needed
                    dx = dx.reshape(x.shape)
                
                dataset['dx'] = dx

            # dataset['dx'] = dx
            
            dataset["time_str"] = dataset['time'].apply(lambda x: x.strftime('%H:%M:%S'))
            dx_values = dataset['dx'].values
            for index, row in dataset.iterrows():
                if index < (self.params['input_dim'] + self.params['future_steps']):
                    continue
                if self.params['open_close'] == 'open':
                    if row['time_str'] in ['09:45:00','10:00:00','10:15:00','10:30:00','10:45:00']:
                        # x_valuesx[index-self.input_dim:index+self.future_steps])
                        xic.append(x[index-self.params['input_dim']:index+self.params['future_steps']])
                        dxic.append(dx_values[index-self.params['input_dim']:index+self.params['future_steps']])
                elif self.params['open_close'] == 'close':
                    if row['time_str'] in ['14:45:00','15:00:00','15:15:00','15:30:00','15:45:00']:
                        # x_valuesx[index-self.input_dim:index+self.future_steps])
                        xic.append(x[index-self.params['input_dim']:index+self.params['future_steps']])
                        dxic.append(dx_values[index-self.params['input_dim']:index+self.params['future_steps']])
        # print(xic[0])
        # print(len(xic))
        H = np.vstack(xic)
        dH = np.vstack(dxic)
        self.t = np.hstack(times)
        self.x = H.T
        self.dx = dH.T
        self.z = np.hstack(x) 
        self.dz = np.hstack(dx)
        self.sindy_coefficients = None # unused
        print(f"self.x.shape: {self.x.shape}")
        print(f"self.dx.shape: {self.dx.shape}")

                
class TemporalFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=100, min_periods=10, scaling_method='rolling', prediction_horizon=8):
        self.window_size = window_size
        self.min_periods = min_periods
        self.scaling_method = scaling_method
        self.prediction_horizon = prediction_horizon
        self.feature_scalers = {}

    def fit(self, X, y=None):
        # X should be a pandas DataFrame
        for column in X.columns:
            if self.scaling_method == 'expanding':
                self.feature_scalers[column] = {
                    'mean': X[column].expanding(min_periods=self.min_periods).mean(),
                    'std': X[column].expanding(min_periods=self.min_periods).std()
                }
            elif self.scaling_method == 'rolling':
                ## double check if this is correct
                self.feature_scalers[column] = {
                    'mean': X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean(),
                    'std': X[column].rolling(window=self.window_size, min_periods=self.min_periods).std()
                }
        return self

    def transform(self, X):
        X_scaled = X.copy()
        for column in X.columns:
            if column in self.feature_scalers:
                mean = self.feature_scalers[column]['mean']
                std = self.feature_scalers[column]['std']
                
                # Update rolling statistics
                # if self.scaling_method == 'rolling':
                #     mean = mean.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean())
                #     std = std.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).std())
                #     self.feature_scalers[column]['mean'] = mean
                #     self.feature_scalers[column]['std'] = std
                
                # Apply normalization using the updated statistics
                X_scaled[column] = (X[column] - mean) / (std + 1e-8)
                
                # Handle NaNs
                X_scaled[column] = X_scaled[column].fillna(method='ffill').fillna(0)
        
        return X_scaled


    def inverse_transform(self, X, evaluation_column, alert_index, model_hyperparameters):
        X_inverse = X.cpu().detach().numpy()
        if evaluation_column == 'h':
            target = X_inverse[:, 1]
        elif evaluation_column == 'l':
            target = X_inverse[:, 2]

        # try:
        mean = self.feature_scalers[evaluation_column]['mean'].reset_index(drop=True)
        std = self.feature_scalers[evaluation_column]['std'].reset_index(drop=True)
        # instance_start = alert_index - model_hyperparameters['context_length']
        instance_std = std[alert_index:alert_index + model_hyperparameters['prediction_horizon']]
        instance_mean = mean[alert_index:alert_index + model_hyperparameters['prediction_horizon']]

        unscaled_target = (target * instance_std) + instance_mean
        # except:
        #     print(f"Error in inverse transform for column {column}")
        return unscaled_target
    

def scale_dataset(data, params):
    temporal_scaler = TemporalFeatureScaler(window_size=params['input_dim'], min_periods=10, scaling_method='rolling', prediction_horizon=params['future_steps'])
    scaled_data = temporal_scaler.fit_transform(data[['c']])
    scaled_data['date'] = data['date']
    scaled_data['date'] = pd.to_datetime(scaled_data['date']).dt.tz_localize(None)
    scaled_data['time'] = scaled_data['date'].dt.time
    scaled_data.rename(columns={'c': 'x'}, inplace=True)
    return scaled_data