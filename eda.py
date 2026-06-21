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

Ref: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
"""

#%% LIB
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from src.service.input_data import fetch_and_transform_data
from src.service.transform_data import (
    get_daily_data,
    get_weekly_data
    )
from src.plot import (
    plot_consumption_from_different_sources_series,
    plot_consumption_daily,
    plot_consumption_weekly
    )

pd.set_option('display.max_columns', None)

#%% LOAD & TRANSFORM  DATA
df = fetch_and_transform_data()

df.head()
df.info()
len(df)

df_daily = get_daily_data(df)
df_weekly = get_weekly_data(df)

#%% PLOT
plot_consumption_from_different_sources_series(
    df, 
    datetime_from='2010-11-01',
    datetime_to='2010-11-26',
    mode='absolute'
    )

plot_consumption_daily(df_daily)

plot_consumption_weekly(df, mode='absolute')

#%% ANALYSIS

plot_acf(df_daily['Daily_power_consumption'], lags=400)
plt.show()

plot_acf(df_weekly['Weekly_energy_consumption'], lags=60)
plt.show()
