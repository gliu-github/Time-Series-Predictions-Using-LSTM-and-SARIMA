import pandas as pd


def prepare_daily_data(target_year):
    """
    :param year: target year
    :return:
    NOTE: column names are different after year 2017
    """
    for dd in range(1, 13):
        # take out 'pick up' data
        file = 'fhv_tripdata_' + str(target_year) + '-' + ('%02d' % dd) + '.csv'
        print("current filename: ", file)
        df = pd.read_csv(file, usecols=['Pickup_date'])
        print(" " * 4 + "data record ", df.shape[0])

        # convert to daily base
        df['Pickup_date'] = pd.to_datetime(df['Pickup_date'])
        df['time'] = df['Pickup_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_count = pd.DataFrame(df['time'].value_counts()).sort_index()
        print(" " * 4 + "days in current file", df_count.shape[0], "days")

        # write to csv file, mode: append
        df_count = df_count.rename({'time': 'trips'}, axis=1)
        df_count.to_csv('FHV_NY_tripdata_2015to2018.csv', mode='a', header=['Trips'],
                        index=df_count.index.tolist(), index_label='Date')
        return


if __name__ == "__main__":
    # e.g., load 2015 data
    target_year = 2015
    prepare_daily_data(target_year)

