# utils for ED analysis

import pandas as pd
import os
import sys


# show progress
def show_progress(total,current):
    percent = current*100/total
    progress = '▉' * int(percent/10)
    sys.stdout.write(f"\rProgress:[{progress:<10}] {percent:.1f}%")
    sys.stdout.flush()


# merge dfs
def merge_dfs(meta_df,right_df):
    dataset_info = pd.merge(meta_df, right_df, on='SEQN', how='left')
    return dataset_info


