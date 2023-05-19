import pandas as pd
import os
from tqdm import trange
import re


def is_csv(file_name):
    if len(file_name) > 4 and file_name[-4:] == ".csv":
        return True
    return False


def get_df_from_path(data_path):
    file_path_list = os.listdir(data_path)
    file_path_list = list(filter(is_csv, file_path_list))
    frames = []
    for file_path in file_path_list:
        df = pd.read_csv(f'{data_path}/{file_path}')
        df = df.drop(columns=df.columns[0])
        df['account_id'] = re.findall(r"(.*)_week*", file_path)[0]
        frames.append(df)
    return pd.concat(frames)


def get_ordered_chunks_path_list():
    # files in Data directory
    data_path = "Data"
    file_path_list = os.listdir(data_path)
    file_path_list = list(filter(is_csv, file_path_list))

    def sort_accounts(acc):
        return int(acc.replace('account_', '').replace('.csv', ''))
    file_path_list.sort(reverse=False, key=sort_accounts)

    return file_path_list


def save_account_weeks_csv(account, df):
    for i in range(53):
        try:
            filtered = df[df['date'].dt.isocalendar().week == i]
        except:
            mask = pd.to_datetime(df['date']).dt.isocalendar().week == i
            filtered = df[mask]
        if filtered.shape[0] > 0:
            filtered.to_csv('Weekly/' + account + '_week_' +
                            str(i) + ".csv", index=False)


def save_account_weeks(file_path):
    for i in trange(len(file_path)):
        account = file_path[i].replace('.csv', '')
        df = pd.read_csv("Simplr/Data/" + file_path[i])
        df = df.drop(columns=['Unnamed: 0'])
        df['date'] = pd.to_datetime(df['created_at'].astype('datetime64[ns]'))
        account = df.iloc[0]["account_id"]
        save_account_weeks_csv(account, df)


def get_accounts_week(week=0, data_path='Data/Weekly'):
    '''
    Get list of account files for given week
    '''
    file_path_list = os.listdir(data_path)
    filtered = list(filter(lambda x: "week_" +
                    str(week) + "." in x, file_path_list))
    return filtered


def list_files(path):
    file_path_list = os.listdir(path)
    print(file_path_list)
    file_path_list = list(filter(is_csv, file_path_list))

    return file_path_list
