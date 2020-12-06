import numpy as np
import pandas as pd
import holidays
from datetime import datetime
import matplotlib.pyplot as plt


def quick_line_plot(dataframe, show=True):
    if not show:
        return
    ax = dataframe.plot(x='Date', y='Trips', kind='line', figsize=(10, 5),
                        style=['black'], marker='', linewidth=1.5, fontsize=10)
    ax.set(xlabel='date', ylabel='number of trips', title='New York FHV in 2015-2018')


def data_manipulation(dataframe):
    """
    :param dataframe:
    :return:
    a- mark datetime with holiday and extreme weather events
    b- shift one day for prediction
    c- take log scale of total trips
    """
    # mark datetime with holiday and extreme weather events
    rec_weather = [datetime.strptime(date, "%Y-%m-%d")
                   for date in ['2018-01-04', '2018-03-21', '2017-03-14', '2017-02-09', '2016-01-23']]
    rec_holidays = [date for year in range(2015, 2019)
                    for date, _ in sorted(holidays.US(years=year).items())]
    dataframe['holiday'] = np.where(dataframe['Date'].isin(rec_holidays), 1, 0)
    dataframe['bad_weather'] = np.where(dataframe['Date'].isin(rec_weather), 1, 0)

    # shift one day for prediction
    dataframe['next_holiday'] = dataframe['holiday'].shift(-1)
    dataframe['next_bad_weather'] = dataframe['bad_weather'].shift(-1)
    dataframe['next_holiday'].fillna(0, inplace=True)
    dataframe['next_bad_weather'].fillna(0, inplace=True)

    # take log scale of total trips
    dataframe['Trips'] = np.log(dataframe['Trips'])
    return dataframe, rec_weather, rec_holidays


def line_plot(dataframe, rec_weather, rec_holidays, show=True):
    if not show:
        return

    ax = dataframe.plot(x='Date', y='Trips', kind='line', figsize=(10, 5),
                        style=['black'], marker='', linewidth=1.5, fontsize=10)
    # add extreme weather
    for rec in rec_weather:
        ax.axvline(rec, color='green', linestyle='--')
    # add holidays
    for rec in rec_holidays:
        ax.axvline(rec, color='gray', linestyle='--')
    ax.set(xlabel='date', ylabel='number of trips', title='New York FHV in 2015-2018')
    plt.show()


if __name__ == "__main__":
    # csv read
    df = pd.read_csv('FHV_NY_tripdata_2015to2018.csv')
    print('size of FHV data', df.shape)
    # eliminate two spikes
    df = df.iloc[:-125]

    # manipulate data type
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # quick view of dataset
    quick_line_plot(df, False)

    # data processing
    df, record_weather, record_holidays = data_manipulation(df)
    line_plot(df, record_weather, record_holidays, show=True)
