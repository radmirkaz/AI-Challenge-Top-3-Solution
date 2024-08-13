import numpy as np
import polars as pl
from scipy import stats
from golden_features import golden_features, whynotcols, whynotstats

import warnings
warnings.filterwarnings("ignore")


def preprocess_df(cur_df: pl.DataFrame) -> pl.DataFrame:
    cur_df = cur_df.with_columns([
        pl.col('НРП').cast(pl.Int8),
        (pl.col('До поставки') >= pl.col('Длительность')).cast(pl.Int8).alias('до_поставки>=длительность'),
        (pl.col('До поставки') == pl.col('Длительность')).cast(pl.Int8).alias('до_поставки==длительность'),
        (pl.col('До поставки') - pl.col('Длительность')).abs().alias('diff'),
        (pl.col('До поставки') / pl.col('Длительность')).alias('diff_divide'),
        (pl.col('diff').apply(lambda x: 1 if x < 2 else 2 if x < 7 else 3 if x < 17 else 4)).alias('diff2'),
        (pl.col('Дней между 0_1') / pl.col('Длительность')).alias('0_1_norm'),
        (pl.col('Дней между 1_2') / pl.col('Длительность')).alias('1_2_norm'),
        (pl.col('Дней между 2_3') / pl.col('Длительность')).alias('2_3_norm'),
        (pl.col('Дней между 3_4') / pl.col('Длительность')).alias('3_4_norm'),
        (pl.col('Дней между 4_5') / pl.col('Длительность')).alias('4_5_norm'),
        (pl.col('Дней между 5_6') / pl.col('Длительность')).alias('5_6_norm'),
        (pl.col('Дней между 6_7') / pl.col('Длительность')).alias('6_7_norm'),
        (pl.col('Дней между 7_8') / pl.col('Длительность')).alias('7_8_norm'),
        pl.sum(["0_1_norm", "1_2_norm", "2_3_norm", "3_4_norm", "4_5_norm", "5_6_norm", "6_7_norm", "7_8_norm"]).alias('0_9_norm'),
        pl.sum(["Количество обработчиков 7", "Количество обработчиков 15", "Количество обработчиков 30"]).alias('сумма обработчиков'),
        pl.sum(["Месяц1", "Месяц2", "Месяц3"]).alias('сумма месяцев'),
        pl.sum(["Согласование заказа 1", "Согласование заказа 2", "Согласование заказа 3"]).alias('сумма согласований'),
        pl.sum(["Изменение даты поставки 7", "Изменение даты поставки 15", "Изменение даты поставки 30"]).alias('сумма изменений даты поставки'),
        (pl.col("Поставщик").cast(pl.Utf8) + "_" +
         pl.col("Группа материалов").cast(pl.Utf8) + "_" +
         pl.col("Завод").cast(pl.Utf8) + "_" +
         pl.col("Вариант поставки").cast(pl.Utf8) + "_" +
         pl.col("НРП").cast(pl.Utf8) + "_" +
         pl.col("Операционный менеджер").cast(pl.Utf8) + "_" +
         pl.col("Категорийный менеджер").cast(pl.Utf8) + "_" +
         pl.col("Материал").cast(pl.Utf8) + "_" +
         pl.col("Закупочная организация").cast(pl.Utf8) + "_" +
         pl.col("ЕИ").cast(pl.Utf8) + "_" +
         pl.col("Группа закупок").cast(pl.Utf8) + "_" +
         pl.col("Балансовая единица").cast(pl.Utf8) + "_" +
         pl.col("Месяц1").cast(pl.Utf8)).alias("вся_инфа"),
        (pl.col('Месяц2') - pl.col('Месяц1')).alias('м1-м2'),
        (pl.col('Месяц3') - pl.col('Месяц2')).alias('м2-м3')
    ])
    return cur_df


def make_gold_fts(df: pl.DataFrame, recipy=golden_features) -> pl.DataFrame:
    for features in recipy['new_features']:
        col1 = features['feature1']
        col2 = features['feature2']
        operation = features['operation']

        if operation == 'multiply':
            res = df[col1] * df[col2]
        elif operation == 'ratio':
            res = df[col1] / df[col2]
        elif operation == 'sum':
            res = df[col1] + df[col2]
        elif operation == 'diff':
            res = df[col1] - df[col2]
        else:
            print(operation)

        df = df.with_column(res.alias(features['new_column']))
    
    return df


def make_fts(df: pl.DataFrame, sub_df: pl.DataFrame) -> pl.DataFrame:
    combined_df = pl.concat([df, sub_df])
    for col in whynotcols:
        agg_df = combined_df.groupby("вся_инфа").agg([
            pl.col(col).mean().alias(f'{col}_mean')
        ])
        df = df.join(agg_df, on="вся_инфа", how="left")
    return df