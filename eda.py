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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'
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
len(df)


#%% PLOT
# Transform before plot
df_plot = df[[
    'datetime', 
    'Sub_metering_0', 
    'Sub_metering_1', 
    'Sub_metering_2', 
    'Sub_metering_3', 
    'Global_reactive_power'
    ]]
df_plot = df_plot.melt(
    id_vars='datetime',
    value_vars=[
        'Sub_metering_0', 
        'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3', 
        'Global_reactive_power'
        ],
    var_name='metering',
    value_name='energy_in_kwh'
    )


def plot_usage_over_time(
        df, 
        datetime_from,
        datetime_to,
        mode='absolute',
        ):
    # CAT_COLOR = {
    #     'nums_unique_not_ai_user': "#696FC7",
    #     'nums_unique_ai_user': "#A1BC98", 
    #     }
    df = df[
        (df['datetime']>=datetime_from)
        &(df['datetime']<=datetime_to)
         ]
    
    fig = px.area(
        df,
        x="datetime",
        y="energy_in_kwh",
        color="metering",
        # color_discrete_map=CAT_COLOR,
        title="Số HS học trên hệ thống mỗi ngày",
        # labels={
        #     "metering": "",
        #     "energy_in_kwh": (
        #         "Số HS" if mode == 'absolute' else
        #         'Tỉ lệ HS (%)'
        #         ),
        #     "datetime": "Ngày",
        # },
    )
    if mode == 'relative':
        fig.update_traces(groupnorm='percent')
    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation='h',
            y=-0.5,
            x=0.5,
            xanchor='center',
            yanchor='bottom'
            )
        )
    fig.for_each_trace(
        lambda t: t.update(
            name={
                'Sub_metering_1': 'kitchen',
                'Sub_metering_2': 'laundry_room',
                'Sub_metering_3': 'water_heater_and_air_conditioner',
                'Sub_metering_0': 'other_rooms',
                'Global_reactive_power': 'reactive_power_source'
                }[t.name]
            )
        )
    fig.show()


plot_usage_over_time(
    df_plot, 
    datetime_from='2010-11-20',
    datetime_to='2010-11-26',
    mode='absolute'
    )

