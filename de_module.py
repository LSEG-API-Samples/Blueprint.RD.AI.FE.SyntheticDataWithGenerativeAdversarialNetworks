import refinitiv.data as rd
import pandas as pd

from os.path import exists
from dateutil import parser
from datetime import datetime

file_path_tick = './studies/flash_crashes/omx30/data/omx30_tick_data.csv'


def get_tick_data(instrument, start_date, end_date):
    df = pd.DataFrame()
    end_date = parser.parse(end_date).replace(tzinfo=None)
    start_date = parser.parse(start_date).replace(tzinfo=None)
    print(f'Requesting tick data for {instrument} for the period {start_date} to {end_date}')
    while end_date >= start_date:
        try:
            temp_df = rd.get_history(universe=[instrument],
                                     end=end_date, count=5000,
                                     fields=["BID", "ASK", "EVENT_TYPE", "BIDSIZE", "ASKSIZE", "TRDPRC_1",
                                             "TRDVOL_1", "VWAP", "ACVOL_UNS", "RTL", "SEQNUM"],
                                     interval='tick')
            end_date = temp_df.index.min().replace(tzinfo=None).strftime('%Y/%m/%d %H:%M:%S.%f')
            end_date = datetime.strptime(end_date, '%Y/%m/%d %H:%M:%S.%f')
            if len(df):
                df = pd.concat([df, temp_df], axis=0)
            else:
                df = temp_df
        except:
            continue

    df.reset_index(inplace=True)
    df['BID'] = pd.to_numeric(df['BID'])
    df['ASK'] = pd.to_numeric(df['ASK'])
    df['BIDSIZE'] = pd.to_numeric(df['BIDSIZE'])
    df['ASKSIZE'] = pd.to_numeric(df['ASKSIZE'])
    df['TRDVOL_1'] = pd.to_numeric(df['TRDVOL_1'])
    df['TRDPRC_1'] = pd.to_numeric(df['TRDPRC_1'])
    df['VWAP'] = pd.to_numeric(df['VWAP'])
    df['ACVOL_UNS'] = pd.to_numeric(df['ACVOL_UNS'])
    df['RTL'] = pd.to_numeric(df['RTL'])
    df['SEQNUM'] = pd.to_numeric(df['SEQNUM'])

    print(f'{df.shape[0]} datapoints for {instrument} are created and stored')
    return df


def get_tick_data_market_sample():
    rd.open_session(app_key='')
    print('session opened')
    df = get_tick_data('OMXS30c1', '2022-05-02T09:00:00Z', '2022-05-02T17:00:00Z')
    rd.close_session()

    df.to_csv(file_path_tick, index=False)


def data_engineering_phase():
    if not exists(file_path_tick):
        get_tick_data_market_sample()
