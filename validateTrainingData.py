import pandas as pd

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

def generate_numpy_array(df, timesteps, timefrom, timeto):
    # We take the df provided and give a numpy array with shape (dflength, 50, 5)
    # Df Length is the length of the dataframe, and the 50 is the 50 entries before

    
