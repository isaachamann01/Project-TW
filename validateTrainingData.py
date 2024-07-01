import pandas as pd
import numpy as np
import tensorflow as tf

# RNN uses previous 50 close prices to predict the average over the next 5 close prices.
previous_timesteps = 50

def import_csv(filename):
    df = pd.read_csv(filename)
    selected_columns = ['Open time', 'Close']
    df = df[selected_columns]
    return df

def generate_rolling_averages(df):
    df['5_unit_avg'] = df['Close'].rolling(window=5).mean()
    df['60_unit_avg'] = df['Close'].rolling(window=60).mean()
    df['1440_unit_avg'] = df['Close'].rolling(window=1440).mean()
    df['43800_unit_avg'] = df['Close'].rolling(window=43800).mean()
    return df

def generate_numpy_array(df, timefrom, timeto):

    # We take the df provided and give a numpy array with shape (dflength, 50, 6)
    # Df Length is the length of the dataframe, and the 50 is the 50 entries before
    # 1st column in the df is the time and the next 5 are the close + rolling averages.

    # Find index of time from and time to
    timefrom = pd.to_datetime(timefrom)
    timeto = pd.to_datetime(timeto)
    timefrom_index = df[df['Open time'] == timefrom].index[0]
    timeto_index = df[df['Open time'] == timeto].index[0]

    # Assuming df is your DataFrame and it has 6 columns
    dfLength = len(df[timefrom_index:timeto_index])

    # Initialize the 3D array
    array_3d = np.zeros((dfLength, 50, 6))

    # Fill the 3D array
    for i in range(timefrom_index, timeto_index):
        # Determine the start index for slicing the DataFrame, ensuring it's not negative
        start_idx = max(i - 49, 0)
        # Slice the DataFrame to get up to 50 rows ending with the current row
        temp_df = df.iloc[start_idx:i+1]
        # If there are fewer than 50 rows, pad the beginning with zeros
        if len(temp_df) < 50:
            padding = np.zeros((50 - len(temp_df), 6))
            temp_array = np.vstack((padding, temp_df.to_numpy()))
        else:
            temp_array = temp_df.to_numpy()
            # Assign this 2D array to the corresponding "page" in the 3D array
            array_3d[i] = temp_array

def remove_time(numpy_array):
    return numpy_array[:, :, 1:]

def normalise_numpy_array(array_3d):
    for i in range(5):
        array_3d[:, :, i] = (array_3d[:, :, i] - array_3d[:, :, i].mean()) / array_3d[:, :, i].std()
    return array_3d

def fit_to_model(array_2d, model):
    # 2D array is the array of the last 50 close prices
    # We need to fit this to the model and get the prediction
    prediction = model.predict(array_2d)
    return prediction

def check_prediction(current, prediction, position):
    # Get percentage difference between current and prediction
    percentage_difference = (prediction - current) / current

    # If percentage difference is greater than 0.002, return buy
    # Else return close
    if percentage_difference > 0.002:
        if position == 'buy':
            return 'hold'
        else:
            return 'buy'
    else:
        return 'close'
    
def check_ROI(array_3d):
    currentAmount = 0
    position = 'close'
    for i in range(array_3d.shape[0]):
        prediction = check_prediction(array_3d[i, 0, 0], array_3d[i, 0, 5], position)
        prediction
    
def main():

    # Import BTCUSDT.h5 current folder.
    model = tf.load_model('BTCUSDT.h5')

    # Import the CSV file
    df = import_csv('BTCUSDT.csv')
    
    # Generate rolling averages
    df = generate_rolling_averages(df)
    
    # Generate numpy array
    array_3d = generate_numpy_array(df, '2021-01-01 00:00:00', '2021-01-01 00:00:00')
    
    # Remove time from numpy array
    array_3d = remove_time(array_3d)
    
    # Normalise the numpy array
    array_3d = normalise_numpy_array(array_3d)
    
    starting_cost = 100

    # get the 2d array from each 3d array
    for i in range(array_3d.shape[0]):
        # the prediction
        array_3d[:,:,6] = fit_to_model(array_3d[i,:,:], model)
    
    # Check prediction
    check_prediction(1, prediction, 'buy')


            



