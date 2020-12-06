import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import collections
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def data_transform(df, look_back=1, forecast_horizon=1, batch_size=1):
    df_x, df_y = df[['Trips', 'next_holiday', 'next_bad_weather']], df['Trips']
    batch_x, batch_y, batch_offset = [], [], []
    for i in range(0, len(df) - look_back - forecast_horizon - batch_size + 1, batch_size):
        # put to small batch
        for j in range(batch_size):
            x = df_x.values[i + j:i + j + look_back, :]
            y = df_y.values[i + j + look_back:i + j + look_back + forecast_horizon]
            offset = x[0, 0]
            batch_x.append(np.array(x).reshape(look_back, -1))
            batch_y.append(np.array(y))
            batch_offset.append(np.array(offset))

        # format to ndarray
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_offset = np.array(batch_offset)

        # substract the first value of each batch
        offsets = batch_offset.reshape(-1, 1)
        batch_x[:, :, 0] -= offsets
        batch_y -= offsets

        # return and reset
        yield batch_x, batch_y, batch_offset
        batch_x, batch_y, batch_offset = [], [], []


def train(model, df, optimizer, n_epochs=10, look_back=1, forecast_horizon=1, batch_size=1):
    # no validation here
    model.train()
    train_true_y, train_pred_y = [], []
    records = collections.defaultdict(list)

    for epoch in tqdm(range(n_epochs)):
        # before 2018-01-01 as training
        train_data = data_transform(df[df.Date < datetime.strptime('2018-01-01', "%Y-%m-%d")],
                                    look_back, forecast_horizon, batch_size)
        epoch_loss = []
        for i, batch in enumerate(train_data):
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break

            # output and loss
            out = model.forward(batch[0].float(), batch_size)
            loss = model.loss(out, batch[1].float())

            # write records
            if epoch == n_epochs - 1:
                train_true_y.append((batch[1] + batch[2]).detach().numpy().reshape(-1))
                train_pred_y.append((out + batch[2]).detach().numpy().reshape(-1))
                records['train_true_y'] = train_true_y
                records['train_pred_y'] = train_pred_y

            # backtracking
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        records['epoch_loss'].append(np.mean(epoch_loss))
        print('Epoch {}/{}: loss={:0.4f}'.format(epoch, n_epochs, np.mean(epoch_loss), end='\r'))
    return model, records


def predict(model, df, n_epochs, look_back, forecast_horizon, batch_size):
    test_true_y, test_pred_y = [], []
    for epoch in range(n_epochs):
        epoch_loss = []
        preds = []
        test_data = data_transform(df[df.Date >= datetime.strptime('2018-01-01', "%Y-%m-%d")],
                                   look_back, forecast_horizon, batch_size)
        for i, batch in enumerate(test_data):
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].float(), batch_size)
            loss = model.loss(out, batch[1].float())
            epoch_loss.append(loss.item())
            if epoch == 0:
                test_true_y.append((batch[1] + batch[2]).detach().numpy().reshape(-1))
            preds.append((out + batch[2]).detach().numpy().reshape(-1))
        print('Epoch {}/{}: loss={:0.4f}'.format(epoch, n_epochs, np.mean(epoch_loss), end='\r'))
        test_pred_y.append(preds)
    return test_true_y, test_pred_y


def evaluate_metrics(y, pred):
    results = pd.DataFrame({'r2_score': r2_score(y, pred)}, index=[0])
    results['Mean_Absolute_Error'] = mean_absolute_error(y, pred)
    results['Median_Absolute_Error'] = median_absolute_error(y, pred)
    results['MSE'] = mean_squared_error(y, pred)
    results['MSLE'] = mean_squared_log_error(y, pred)
    results['MAPE'] = np.mean(np.abs((y - pred) / y)) * 100
    results['RMSE'] = np.sqrt(results['MSE'])
    return results


def dickey_fuller(timeseries):
    print("Results of Dickey-Fuller Test:")
    test = adfuller(timeseries, autolag='AIC')
    output = pd.Series(test[0:4],
                       index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    for k, v in test[4].items():
        output['Critical Value {}'.format(k)] = v
        output['stationary {}'.format(k)] = 'False' if v < test[0] else 'True'
    print(output)
    return


def decompose(timeseries):
    """
    decomposition of seasonal time series
    """
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig, ax = plt.subplots(3, figsize=(9, 9))
    ax[0].plot(trend, label='Trend')
    ax[0].legend(loc='best')
    ax[1].plot(seasonal, label='Seasonality')
    ax[1].legend(loc='best')
    ax[2].plot(residual, label='Residuals')
    ax[2].legend(loc='best')
    plt.tight_layout()
    fig.autofmt_xdate()
    return trend, seasonal, residual


def plot_acf_pacf(data):
    fig, ax = plt.subplots(3, figsize=(9,9))
    ax[0].plot(data)
    ax[1] = plot_acf(data, ax = ax[1], lags = 20)
    ax[2] = plot_pacf(data, ax = ax[2], lags = 20)
    plt.tight_layout()