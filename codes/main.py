import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import time
from utils import train, predict, evaluate_metrics, dickey_fuller, plot_acf_pacf
from EDA_of_prepared_data import data_manipulation
from model_lstm import Model
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 2000)


filename = 'FHV_NY_tripdata_2015to2018.csv'
LOOK_BACK = 28
FORECAST_HORIZON = 1
BATCH_SIZE = 1
N_EPOCHS = 30
LEARNING_RATE = 0.001
config = dict(features=3, forecast_horizon=FORECAST_HORIZON)


def plot_train_performance(records_dict):
    train_true_y = np.array(records_dict['train_true_y'])
    train_pred_y = np.array(records_dict['train_pred_y'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.array(train_true_y[:150]).reshape(-1), c='black', label='truth')
    ax.plot(train_pred_y[:150], label='pred', c='red', linestyle='--', alpha=0.7)
    ax.set(title="train one step ahead forecast", ylabel="Number of trips (log)",
           xlabel="Days", ylim=(11.0, 12.5))
    ax.legend()
    plt.show()


def plot_predict_lstm(true_y, predict_y):
    true_y = np.array(true_y)
    predict_y = np.array(predict_y)
    mean = np.mean(predict_y, axis=0).reshape(-1)
    std = np.std(predict_y, axis=0).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_y.reshape(-1), c='black', linewidth=2, label='truth')
    ax.plot(mean, label='pred', c='blue', linestyle='--', alpha=0.7)
    ax.fill_between([*range(len(true_y.reshape(-1)))], mean - 2 * std, mean + 2 * std,
                    label='99%', color='red', alpha=.7)
    ax.fill_between([*range(len(true_y.reshape(-1)))], mean - 3 * std, mean + 3 * std,
                    label='95%', color='grey', alpha=.5)
    ax.set(xlabel="Days", ylabel="Number of trips (scaled)", ylim=(13.0, 13.8))
    ax.legend()
    plt.show()


def run_lstm(df):
    # train model
    model = Model(config)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    since = time.time()
    model, records = train(model, df, optimizer, N_EPOCHS, LOOK_BACK, FORECAST_HORIZON, BATCH_SIZE)
    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # check train performance
    plot_train_performance(records)
    print(records['epoch_loss'])

    # predict with test data
    test_true_y, test_pred_y = predict(model, df, N_EPOCHS, LOOK_BACK, FORECAST_HORIZON, BATCH_SIZE)

    # show test result
    plot_predict_lstm(test_true_y, test_pred_y)
    eval_lstm = evaluate_metrics(np.exp(test_true_y), np.mean(np.exp(test_pred_y), axis=0).reshape(-1))
    return eval_lstm


def run_sarima(df_):
    # take rolling mean to make data stationary
    data = df_[['Date', 'Trips']]
    data.set_index('Date', inplace=True)
    data['rolling'] = data['Trips'].rolling(7).mean()
    data['raw_ma'] = data['Trips'] - data['rolling']
    dickey_fuller(data['raw_ma'].dropna())
    plot_acf_pacf(data['raw_ma'].dropna())

    # split data into train and test using 2018-01-01
    split_date = datetime.strptime('2018-01-01', '%Y-%m-%d')
    train = data['raw_ma'].iloc[data.index < split_date]
    test = data['raw_ma'].iloc[data.index >= split_date]
    train = train.dropna()
    test = test.dropna()

    # check 1-differencing
    x = data['raw_ma'].dropna() - data['raw_ma'].dropna().shift(7)
    plot_acf_pacf(x.dropna())

    # define p,d,q as 2,1,1 AIC and GridSearch can be used for batch processing
    model = sm.tsa.statespace.SARIMAX(train, order=(2, 1, 1), seasonal_order=(2, 1, 1, 7),
                                      enforce_invertibility=False, enforce_stationarity=False)
    model_fit = model.fit()

    # prediction test data and prepare values for plotting
    pred = model_fit.get_prediction(start=test.index[0].strftime('%Y-%m-%d'),
                                    end=test.index[-1].strftime('%Y-%m-%d'))
    idx = test.shape[0] + 7
    rolling_back = df['Trips'].iloc[-idx:-1].rolling(7).mean()
    fc = pred.predicted_mean + rolling_back.dropna().values
    conf = pred.conf_int(alpha=0.05)
    lower_bounds = conf.iloc[:, 0] + rolling_back.dropna().values
    upper_bounds = conf.iloc[:, 1] + rolling_back.dropna().values
    rmse_ = np.sqrt(np.mean(np.square(test.values - pred.predicted_mean.values)))

    # plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data.loc[data.index >= datetime.strptime('2017-10-01', '%Y-%m-%d'), 'Trips'].index,
            data.loc[data.index >= datetime.strptime('2017-10-01', '%Y-%m-%d'), 'Trips'].values,
            color='black')
    ax.plot(test.index, test.values + rolling_back.dropna().values,
            color='blue', label='truth')
    ax.plot(test.index, fc, color='red', linestyle='--',
            label="prediction (RMSE={:0.2f})".format(rmse_))
    ax.fill_between(test.index, lower_bounds, upper_bounds,
                    color='orange', alpha=0.4, label="confidence interval (95%)")
    ax.legend()
    ax.set_title("SARIMA")

    # evaluate performance
    evaluation = evaluate_metrics(test.values + rolling_back.dropna().values, fc)
    return evaluation


if __name__ == "__main__":
    # csv read and quick process data
    df = pd.read_csv(filename)
    df = df.iloc[:-125]  # eliminate two spikes
    print('size of FHV data', df.shape)

    # manipulate data type
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df, _, _ = data_manipulation(df)

    # run LSTM
    eval_lstm = run_lstm(df)

    # run SARIMA
    df['Trips'] = np.exp(df['Trips'])
    eval_sarima = run_sarima(df)

    # print result
    print('------' * 15)
    print(eval_lstm.head())
    print('------' * 15)
    print(eval_sarima.head())



