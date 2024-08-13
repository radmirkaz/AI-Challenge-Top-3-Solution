import numpy as np
import polars as pl
from scipy import stats
from golden_features import golden_features, whynotcols, whynotstats

import warnings
warnings.filterwarnings("ignore")


def preprocess_df(cur_df):
    cur_df['НРП'] = cur_df['НРП'].astype(np.int8)

    cur_df['до_поставки>=длительность'] = (cur_df['До поставки'] >= cur_df['Длительность']).astype('int')
    cur_df['до_поставки==длительность'] = (cur_df['До поставки'] == cur_df['Длительность']).astype('int')
    cur_df['diff'] = (cur_df['До поставки'] - cur_df['Длительность']).apply(abs)
    cur_df['diff_divide'] = cur_df['До поставки'] / cur_df['Длительность']
    cur_df['diff2'] = cur_df['diff'].apply(lambda x: 1 if x < 2 else 2 if x < 7 else 3 if x < 17 else 4)

    cur_df['0_1_norm'] = cur_df['Дней между 0_1'] / cur_df['Длительность']
    cur_df['1_2_norm'] = cur_df['Дней между 1_2'] / cur_df['Длительность']
    cur_df['2_3_norm'] = cur_df['Дней между 2_3'] / cur_df['Длительность']
    cur_df['3_4_norm'] = cur_df['Дней между 3_4'] / cur_df['Длительность']
    cur_df['4_5_norm'] = cur_df['Дней между 4_5'] / cur_df['Длительность']
    cur_df['5_6_norm'] = cur_df['Дней между 5_6'] / cur_df['Длительность']
    cur_df['6_7_norm'] = cur_df['Дней между 6_7'] / cur_df['Длительность']
    cur_df['7_8_norm'] = cur_df['Дней между 7_8'] / cur_df['Длительность']

    cur_df['0_9_norm'] = cur_df[['0_1_norm','1_2_norm','2_3_norm','3_4_norm','4_5_norm','5_6_norm','6_7_norm','7_8_norm']].sum(axis=1)
    cur_df['сумма обработчиков'] = cur_df[['Количество обработчиков 7', 'Количество обработчиков 15', 'Количество обработчиков 30']].sum(axis = 1) # skipna = True
    cur_df['сумма месяцев'] = cur_df[['Месяц1', 'Месяц2', 'Месяц3']].sum(axis=1)
    cur_df['сумма согласований'] = cur_df[['Согласование заказа 1','Согласование заказа 2','Согласование заказа 3']].sum(axis=1)
    cur_df['сумма изменений даты поставки'] = cur_df[['Изменение даты поставки 7','Изменение даты поставки 15','Изменение даты поставки 30']].sum(axis=1)

    cur_df['вся_инфа'] = cur_df['Поставщик'].astype(str) + '_' + cur_df['Группа материалов'].astype(str) + '_' + cur_df['Завод'].astype(str) + '_' + cur_df['Вариант поставки'].astype(str) + '_' + cur_df['НРП'].astype(str) + '_' + cur_df['Операционный менеджер'].astype(str) + '_' + cur_df['Категорийный менеджер'].astype('str') + '_' + cur_df['Материал'].astype(str) + '_' + cur_df['Закупочная организация'].astype(str) + '_' + cur_df['ЕИ'].astype(str) + '_' + cur_df['Группа закупок'].astype(str) + cur_df['Балансовая единица'].astype(str) + cur_df['Месяц1'].astype(str)
    cur_df['м1-м2'] = cur_df['Месяц2'] - cur_df['Месяц1']
    cur_df['м2-м3'] = cur_df['Месяц3'] - cur_df['Месяц2']

    return cur_df


def make_gold_fts(df, recipy=golden_features):
    start_columns = df.columns.to_list()
    for features in recipy['new_features']:
        col1 = features['feature1']
        col2 = features['feature2']
        operation = features['operation']

        if operation == 'multiply':
            res = df[col1]*df[col2]
        elif operation == 'ratio':
            res = df[col1]/df[col2]
        elif operation == 'sum':
            res = df[col1]+df[col2]
        elif operation == 'diff':
            res = df[col1]-df[col2]
        else:
            print(operation)

        df = pd.concat([df, res], axis=1)
    df.columns = start_columns + recipy['new_columns']
    return df


def make_fts(df, sub_df):
    for col in whynotcols:
        s = pd.concat([df, sub_df]).groupby(by='вся_инфа')[col].agg(['mean'])
        s.columns = [f'{col}_{i}' for i in ['mean']]
        df = df.merge(s, on='вся_инфа', how='left')
    return df