"""
Information from source: 
1.date: Date in format dd/mm/yyyy
2.time: time in format hh:mm:ss
3.global_active_power: household global minute-averaged active power (in kilowatt)
4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
5.voltage: minute-averaged voltage (in volt)
6.global_intensity: household global minute-averaged current intensity (in ampere)
7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

Notes: 
1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.
"""

#%% LIB
import pandas as pd
pd.set_option('display.max_columns', None)

#%% LOAD & TRANSFORM  DATA
df = pd.read_csv("data/household_power_consumption.txt", sep=";")
df['datetime'] = df['Date'] + ' ' + df['Time']
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['Global_active_power']!='?']
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
        
df['Sub_metering_0'] = df['Global_active_power']*1000/60 - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
    
df.head()
df.info()
