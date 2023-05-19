#############################################
# To get metrics on outputs from each model #
#############################################

import pandas as pd
import os
import pickle as pkl

def analyse_account(account_number, chunks_directory = '../data/chunks/'):
    df = pd.read_csv(chunks_directory + "chunks3_account_" + str(account_number) + "_week_18.csv")
    noise_by_account = pkl.load(open('../data/junk/junk_by_account_10000.pkl', 'rb'))
    noise = noise_by_account['account_' + str(account_number)]
    print("Noise predicted by frequency model:")
    display(noise)
    print("Noise of the account (chunks):")
    display(df[df.signal == False])

    print("False positives: (noise labelled as signal):")
    false_positives = df[(df.signal == False) & ~df.chunk.isin(noise)]
    display(false_positives)
    print("False negatives (signal labelled as noise):")
    false_negatives = df[(df.signal == True) & df.chunk.isin(noise)]
    display(false_negatives.head())

    return {
        'df': df,
        'fp': false_positives,
        'fn': false_negatives,
        'noise': noise
    }
    