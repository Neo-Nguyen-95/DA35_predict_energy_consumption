def get_daily_data(df):
    return (
        df
        .groupby('Date')['Total_power_consumption']
        .sum()
        .reset_index(name='Daily_power_consumption')
        )

def get_weekly_data(df):
    return (
    df
    .groupby('week')['Total_power_consumption']
    .sum()
    .reset_index(name='Weekly_energy_consumption')
    )