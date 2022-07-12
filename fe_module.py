import pickle
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data_file_path = './studies/flash_crashes/omx30/data/omxs30c1_td_raw.csv'
file_path = './studies/flash_crashes/omx30/features/'


def tick_gradient(start_tick, end_tick, start_prx, end_prx):
    return (end_prx - start_prx)/(end_tick - start_tick)


def read_de_output():
    df = pd.read_csv(data_file_path)
    df = df.sort_values(by='Timestamp')
    df = df[df['Timestamp'] >= '2022-05-02 07:00:00.000000+00:00']
    df['SEQNUM'] = range(1, df.shape[0] + 1)

    trade_tick_data = df[df['EVENT_TYPE'] == 'trade']
    trade_tick_data = trade_tick_data[['Timestamp',
                                       'SEQNUM',
                                       'TRDPRC_1',
                                       'TRDVOL_1']]
    trade_tick_data['TNO'] = range(1, trade_tick_data.shape[0] + 1)
    trade_tick_data['TPT'] = trade_tick_data['TNO'] / trade_tick_data['SEQNUM']

    price_gradient = [0]
    for i in range(1, trade_tick_data.shape[0]):
        price_gradient.append(np.abs(tick_gradient(trade_tick_data.iloc[i-1, 1],
                                            trade_tick_data.iloc[i, 1],
                                            trade_tick_data.iloc[i-1, 2],
                                            trade_tick_data.iloc[i, 2])))

    trade_tick_data['PRICE_GRADIENT'] = price_gradient
    trade_tick_data = trade_tick_data.set_index('Timestamp')
    trade_tick_data.to_csv('./studies/flash_crashes/omx30/data/omxs30c1_td_clean.csv')

    return trade_tick_data


def transform_real_sample(df):
    scaler_file = file_path + 'scaler.pickle'
    pca_file = file_path + 'pca.pickle'

    X = df.to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_model = PCA(n_components=2)
    x_pca = pca_model.fit_transform(X_scaled)

    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pca_file, 'wb') as f:
        pickle.dump(pca_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    c = np.ones(df.shape[0]).reshape(df.shape[0], 1)

    return x_pca, c


def feature_engineering_phase():
    features_file = file_path + 'x_c.pickle'

    df = read_de_output()
    x_pca, c = transform_real_sample(df)

    features_dict = {'x': x_pca, 'c': c}

    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



