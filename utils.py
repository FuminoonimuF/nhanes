# utils for ED analysis

import pandas as pd
import os
import sys


# show progress
def show_progress(total, current):
    percent = current * 100 / total
    progress = '▉' * int(percent / 10)
    sys.stdout.write(f"\rProgress:[{progress:<10}] {percent:.1f}%")
    sys.stdout.flush()


# merge dfs
def merge_dfs(meta_df, right_df):
    dataset_info = pd.merge(meta_df, right_df, on='SEQN', how='left')
    return dataset_info


# preprocess
def preprocess(dir):
    ed_path = '%s/ED_%s.csv' % (dir, dir)
    lbd_path = '%s/LBD_%s.csv' % (dir, dir)
    d_path = '%s/D_%s.csv' % (dir, dir)
    vit_path = '%s/L45VIT_C_%s.csv' % (dir, dir)

    bmi_path = '%s/BMI_%s.csv' % (dir, dir)
    info_path = '%s/DEMO_B_%s.csv' % (dir, dir)
    sst_path = '%s/SST_%s.csv' % (dir, dir)
    diq_path = '%s/DIQ_%s.csv' % (dir, dir)
    alq_path = '%s/ALQ_%s.csv' % (dir, dir)
    smoke_path = '%s/SMOKE_%s.csv' % (dir, dir)
    sports_path = '%s/SPORTS_%s.csv' % (dir, dir)
    mcq_path = '%s/MCQ_C_%s.csv' % (dir, dir)
    bpq_path = '%s/BPQ_B_%s.csv' % (dir, dir)

    # load datasets
    dataset_ed = pd.read_csv(ed_path)
    dataset_lbd = pd.read_csv(lbd_path)
    dataset_d = pd.read_csv(d_path)
    dataset_vit = pd.read_csv(vit_path)

    dataset_info = pd.read_csv(info_path)
    dataset_bmi = pd.read_csv(bmi_path)
    dataset_sst = pd.read_csv(sst_path)
    dataset_diq = pd.read_csv(diq_path)
    dataset_alq = pd.read_csv(alq_path)
    dataset_smoke = pd.read_csv(smoke_path)
    dataset_sports = pd.read_csv(sports_path)
    dataset_mcq = pd.read_csv(mcq_path)
    dataset_bpq = pd.read_csv(bpq_path)

    # except dataset_info
    info_dataset_list = [dataset_bmi, dataset_diq, dataset_alq, dataset_bpq,
                         dataset_smoke, dataset_sports, dataset_sst, dataset_mcq]
    # except dataset_ed
    target_dataset_list = [dataset_d, dataset_lbd, dataset_vit]

    # create interviewee info data
    info_dataset = dataset_info
    for i in info_dataset_list:
        info_dataset = merge_dfs(info_dataset, i)

    '''
    info_dataset
    ----------------------------------------
    Description  Note
    SEQN     序列号
    DMDMARTL 婚姻状态排除77 99    Marital Status,%    分组：Married=1/Widowed=2/Divorced=3/Separated=4/Never married=5/Living with partner=6
    DMDEDUC  教育水平 排除7 9     Education,%         分组：Less Than High School=1/High School Diploma=2/More Than High School=3
    RIDAGEEX 年龄                 Age,year            分组：240-480/481-720/721-   (表里按月份算)
    WTMEC2YR 权重 叠加/2          Weight
    RIAGENDR 性别 排除2           Gender
    RIDRETH1 种族                 Race,%              分组：Mexican American=1/Other Hispanic=2/Non-Hispanic White=3/Non-Hispanic Black=4/Other Race=5
  
    ----------------------------------------
    SSTESTO  睾酮                                Testostosterone,%     分组：-3/3-10/10-
    PAD320   过去30天能否进行中等活动 排除7 9    Physical Activity,%   分组：Y/N/Unable
    PAD480   每日电脑使用时间 排除77 99          Daily hours of Computer use,%   分组：Less than 1 hour=0/1 hour=1/2-5 hours=2+3+4+5/None=6
    BMXBMI   BMI                                 BMI,kg/m2,IQR
    SMQ020   吸烟史 排除7 9                      Smoke,%                分组：Y/N
    ALQ101   饮酒史 排除7 9                      Alcohol,%              分组：Y/N
    BPQ040A  服用高血压药 排除7 9 1              Hypertention,%         分组：Y/N
    BPQ090D  服用降脂药 排除 7 9 1               cholesterol,%          分组：Y/N
    DIQ010   糖尿病 排除 7 9 1                   Diabetes,%             分组：Y/N
    MCQ220   癌症 排除 7 9 1                     Cancer,%               分组：Y/N
    ----------------------------------------
    '''

    info_dataset = info_dataset.loc[:, ['SEQN', 'DMDMARTL', 'DMDEDUC', 'RIAGENDR', 'RIDAGEEX', 'RIDRETH1', 'WTMEC2YR',
                                        'SSTESTO', 'PAD320', 'PAD480', 'BMXBMI', 'SMQ020', 'ALQ101', 'DIQ010',
                                        'BPQ040A', 'BPQ090D', 'MCQ220']]

    print(info_dataset.head(5))

    # preprocess info dataset
    # 婚姻状态 排除77 99
    drop_index = info_dataset[info_dataset['DMDMARTL'].isin([77, 99])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['DMDMARTL'])

    # 教育水平 排除 7 9
    drop_index = info_dataset[info_dataset['DMDEDUC'].isin([7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['DMDEDUC'])

    # WTMEC2YR权重/2
    info_dataset['WTMEC2YR'] = info_dataset['WTMEC2YR'] / 2

    # 排除女性
    drop_index = info_dataset[info_dataset['RIAGENDR'] == 2].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['RIAGENDR'])

    # 排除睾酮<3
    drop_index = info_dataset[info_dataset['SSTESTO'] < 3].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['SSTESTO'])

    # 排除糖尿病
    drop_index = info_dataset[info_dataset['DIQ010'].isin([1, 7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['DIQ010'])

    # 过去30天能否进行中等活动 排除 7 9
    drop_index = info_dataset[info_dataset['PAD320'].isin([7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    # info_dataset = info_dataset.drop(columns=['PAD320'])

    # 每日电脑使用时间 排除77 99
    drop_index = info_dataset[info_dataset['PAD480'].isin([77, 99])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    # info_dataset = info_dataset.drop(columns=['PAD480'])

    # 吸烟史 排除7 9
    drop_index = info_dataset[info_dataset['SMQ020'].isin([7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    # info_dataset = info_dataset.drop(columns=['SMQ020'])

    # 饮酒史 排除7 9
    drop_index = info_dataset[info_dataset['ALQ101'].isin([7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    # info_dataset = info_dataset.drop(columns=['ALQ101'])

    # 癌症 排除7 9 1
    drop_index = info_dataset[info_dataset['MCQ220'].isin([1, 7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['MCQ220'])

    # 服用高血压药 排除7 9 1
    drop_index = info_dataset[info_dataset['BPQ040A'].isin([1, 7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['BPQ040A'])

    # 服用降压药 排除7 9 1
    drop_index = info_dataset[info_dataset['BPQ090D'].isin([1, 7, 9])].index
    print(drop_index, 'deleted')
    info_dataset.drop(drop_index, inplace=True)
    info_dataset = info_dataset.drop(columns=['BPQ090D'])

    '''
    merged_dataset
    ----------------------------------------
    Description  Note
    LBDCRYSI β隐黄素   b-cryptoxanthin(ug/dL)
    LBDLUZSI 叶黄素    Combined Lutein/zeaxanthin(ug/dL)
    LBDB12SI 维生素12  Vitamin B12, serum (pmol/L)
    LBDFOLSI 叶酸      Folate, serum (nmol/L)
    LBDVIDMS 维生素D   Vitamin D(nmol/L)
    ----------------------------------------
    KIQ400   ed自评 排除 7 9    ED(Erectile Dysfunction),%, 分组：Y=1+2/N=3+4
    ----------------------------------------
    '''
    # merge datasets and remove SQEN col, reset index, drop unused cols
    merged_dataset = dataset_ed
    for i in target_dataset_list:
        merged_dataset = merge_dfs(merged_dataset, i)

    merged_dataset = merged_dataset.loc[:,
                     ['KIQ400', 'LBDFOLSI', 'LBDB12SI', 'SEQN', 'LBDVIDMS', 'LBDCRYSI', 'LBDLUZSI']]

    print(merged_dataset.head(5))

    # merge info dataset and target dataset
    merged_dataset = merge_dfs(info_dataset, merged_dataset)
    merged_dataset = merged_dataset.drop(columns=['SEQN'])

    print(merged_dataset)

    # delete rows including na
    merged_dataset = merged_dataset.dropna()
    print(merged_dataset)

    # delete incorrect data
    drop_index = merged_dataset[merged_dataset['KIQ400'].isin([7, 9])].index
    print(drop_index, 'deleted.')
    merged_dataset.drop(drop_index, inplace=True)

    drop_index = merged_dataset[merged_dataset['LBDB12SI'] > 5000].index
    print(drop_index, 'deleted')
    merged_dataset.drop(drop_index, inplace=True)

    drop_index = merged_dataset[merged_dataset['LBDFOLSI'] > 500].index
    print(drop_index, 'deleted')
    merged_dataset.drop(drop_index, inplace=True)

    merged_dataset = merged_dataset.reset_index(drop=True)

    print(merged_dataset.head(5))
    return merged_dataset


# RIDAGEEX 年龄                 Age,year            分组：240-480/481-720/721-   (表里按月份算)
def spilt_by_age(dataset):
    low_age = dataset.drop(dataset[(dataset['RIDAGEEX'] <= 240) | (dataset['RIDAGEEX'] > 480)].index)
    low_age = low_age.reset_index(drop=True)
    mid_age = dataset.drop(dataset[(dataset['RIDAGEEX'] <= 480) | (dataset['RIDAGEEX'] > 720)].index)
    mid_age = mid_age.reset_index(drop=True)
    high_age = dataset.drop(dataset[(dataset['RIDAGEEX'] <= 720)].index)
    high_age = high_age.reset_index(drop=True)
    print(low_age, mid_age, high_age)
    return {
        "low_age_dataset": low_age,
        "mid_age_dataset": mid_age,
        "high_age_dataset": high_age
    }
