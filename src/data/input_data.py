#%% LIB
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'household_power_consumption.txt'
#%% REPOSITORY
def fetch_data(data_path=DATA_PATH):
    df = pd.read_csv(DATA_PATH, sep=";")
    return df

#%% TRANSFORM
def trasnform_data(df):
    df = df.copy()
    df = (
        df
        .pipe(format_time)
        .pipe(remove_missing_rows)
        .pipe(format_numeric_field)
        .pipe(compute_additional_energy_consumption)
        )
    return df

def format_time(df):
    df = df.copy()
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.to_period('W-SUN').dt.start_time
    return df

def remove_missing_rows(df):
    df = df.copy()
    df = df[df['Global_active_power']!='?']
    return df

def format_numeric_field(df):
    df = df.copy()
    df[['Global_active_power',
        'Global_reactive_power', 
        'Voltage', 
        'Global_intensity', 
        'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3']] = df[[
            'Global_active_power',
            'Global_reactive_power', 
            'Voltage', 
            'Global_intensity', 
            'Sub_metering_1', 
            'Sub_metering_2', 
            'Sub_metering_3']].astype(float)
    return df

def compute_additional_energy_consumption(df):
    df = df.copy()
    df['Active_power_consumption'] = (
        df['Global_active_power'] * 1000 * 1/60
        )
    df['Sub_metering_0'] = (
        df['Active_power_consumption']
        - df['Sub_metering_1'] 
        - df['Sub_metering_2'] 
        - df['Sub_metering_3']
        )
    df['Reactive_power_consumption'] = (
        df['Global_reactive_power'] * 1000 * 1/60
        )
    df['Total_power_consumption'] = (
        df['Active_power_consumption'] + df['Reactive_power_consumption']
        )
    return df

#%% SERVICE
def fetch_and_transform_data():
    df = fetch_data()
    df = trasnform_data(df)
    return df