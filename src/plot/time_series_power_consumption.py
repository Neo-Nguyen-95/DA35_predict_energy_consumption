#%% LIB
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#%% PLOT 1: ALL TIME STEP
def plot_consumption_from_different_sources_series(
        df, 
        datetime_from,
        datetime_to,
        mode='absolute',
        ):
    # Transform before plot
    df_plot = df[[
        'datetime', 
        'Sub_metering_0', 
        'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3', 
        'Reactive_power_consumption'
        ]]
    df_plot = df_plot.melt(
        id_vars='datetime',
        value_vars=[
            'Sub_metering_0', 
            'Sub_metering_1', 
            'Sub_metering_2', 
            'Sub_metering_3', 
            'Reactive_power_consumption'
            ],
        var_name='metering',
        value_name='energy_in_wh'
        )
    
    df_plot = df_plot[
        (df_plot['datetime']>=datetime_from)
        &(df_plot['datetime']<=datetime_to)
         ]
    
    fig = px.area(
        df_plot,
        x="datetime",
        y="energy_in_wh",
        color="metering",
        title="Energy consumption of a household (measured each minute)",
        labels={
            "metering": "Meters",
            "energy_in_kwh": (
                "Energy Consumption (Wh)" if mode == 'absolute' else
                'Energy Proportion (%)'
                ),
            "datetime": "Date",
        },
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
                'Reactive_power_consumption': 'reactive_power_source'
                }[t.name]
            )
        )
    fig.show()
    
#%% PLOT 2: DAILY
def plot_consumption_daily(df_daily):
    fig = px.line(
        df_daily,
        x='Date',
        y='Daily_power_consumption'
        )
    fig.show()
    
#%% PLOT 3: WEEKLY
def process_weekly_energy_consumption(df, col):
    df_result = (
        df
        .groupby('week')[col]
        .sum()
        .reset_index(name='energy_consumption')
        )
    df_result['metering'] = col
    return df_result


def plot_consumption_weekly(df, mode='absolute'):
    df_sub_metering_0 = process_weekly_energy_consumption(df, 'Sub_metering_0')
    df_sub_metering_1 = process_weekly_energy_consumption(df, 'Sub_metering_1')
    df_sub_metering_2 = process_weekly_energy_consumption(df, 'Sub_metering_2')
    df_sub_metering_3 = process_weekly_energy_consumption(df, 'Sub_metering_3')
    df_reactive_power_consumption = process_weekly_energy_consumption(
        df, 'Reactive_power_consumption'
        )
    df_plot = pd.concat([
        df_sub_metering_0,
        df_sub_metering_1,
        df_sub_metering_2,
        df_sub_metering_3,
        df_reactive_power_consumption
        ], axis='rows')
    fig = px.area(
        df_plot,
        x="week",
        y="energy_consumption",
        color="metering",
        title="Weeky energy consumption of a household (Wh)",
        labels={
            "metering": "Meters",
            "energy_consumption": (
                "Energy Consumption (Wh)" if mode == 'absolute' else
                'Energy Proportion (%)'
                ),
            "week": "Week",
        },
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
                'Reactive_power_consumption': 'reactive_power_source'
                }[t.name]
            )
        )
    fig.show()