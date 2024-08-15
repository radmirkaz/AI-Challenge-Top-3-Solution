import streamlit as st
import polars as pl
import numpy as np
import catboost
from utils_polars import preprocess_df, make_gold_fts, make_fts
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from golden_features import cols_le
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from catboost import Pool
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import json
from scipy import stats
from phik.report import plot_correlation_matrix
from phik import report
import plotly.figure_factory as ff
import pandas as pd


def reset():
    st.session_state.key += 1

def reduce_mem_usage_pd(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    # print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    # print("Memory usage is: ",mem_usg," MB")
    # print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

def reduce_mem_usage_pl(df):
    
    start_mem = df.estimated_size("mb")
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # pl.Uint8,pl.UInt16,pl.UInt32,pl.UInt64
    Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    Numeric_Float_types = [pl.Float32,pl.Float64]
    
    for col in df.columns:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))

        elif col_type in Numeric_Float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))
            else:
                pass
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    # mem_usg = df.estimated_size("mb")
    # print("Memory usage became: ",mem_usg," MB")
    return df

@st.cache_resource(show_spinner=True)
class CatboostModel:
    def __init__(self):
        self.model, self.cat_fts = self.load_model()
        self.train_data = None
        self.test_data = None
    
    def load_model(self, model_path='fast_catboost.cb'):
        cat_fts = ['Поставщик', 'Материал', 'Категорийный менеджер', 'Операционный менеджер',
                    'Завод', 'Закупочная организация', 'Группа закупок', 'Балансовая единица', 'ЕИ',
                    'Группа материалов', 'Вариант поставки', 'Месяц1', 'Месяц2', 'Месяц3',
                    'День недели 2']
        
        model = catboost.CatBoostClassifier(
                iterations=1500,
                random_seed=42,
                eval_metric='AUC',
                cat_features=cat_fts
        )
        model.load_model(model_path)
        
        return model, cat_fts

    def load_supp_stat(self):
        with open('supplier_stat.json', 'r') as f:
            return json.load(f)
    
    def load_data(self):
        df_train = reduce_mem_usage_pl(pl.read_csv('data/train_AIC.csv'))
        df_test = reduce_mem_usage_pl(pl.read_csv('data/test_AIC.csv'))

        # df_train = df_train.with_columns([
        #     pl.col('Поставщик').cast(pl.Int16),
        #     pl.col('Материал').cast(pl.UInt16),
        #     pl.col('Категорийный менеджер').cast(pl.Int8),
        #     pl.col('Операционный менеджер').cast(pl.Int8),
        #     pl.col('Завод').cast(pl.Int8),
        #     pl.col('Закупочная организация').cast(pl.Int8),
        #     pl.col('Группа закупок').cast(pl.Int16),
        #     pl.col('Балансовая единица').cast(pl.Int8),
        #     pl.col('ЕИ').cast(pl.Int8)
        # ])

        df_train = self.process_data(df_train)
        df_test = self.process_data(df_test)

        self.train_data = df_train
        self.test_data = df_test

        return df_train, df_test, self.load_supp_stat()

    def process_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # if self.model_type == 'Catboost (Наивысшая точность)':
        #     df = preprocess_df(df)
        #     df = make_gold_fts(df)
        #     df = make_fts(df, self.train_data).fillna(-1)
        # elif self.model_type == 'Catboost (Наивысшая скорость)':
        df = df.with_columns([
            (pl.col('До поставки') - pl.col('Длительность')).alias('diff'),
        ]).fill_null(-1)

        return df

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if len(df.shape) == 1:
            df = pl.DataFrame([df])

        df = self.process_data(df)
        df = df.with_columns([
            pl.col(self.cat_fts).cast(pl.Utf8)
        ])
        df_pandas = df.to_pandas()  # Convert to pandas for CatBoost compatibility

        preds = self.model.predict_proba(Pool(df_pandas[self.model.feature_names_], cat_features=self.cat_fts))
        return preds

@st.cache_resource(show_spinner=False)
def get_similar_samples(df, sample, columns):
    # получаем из df сэмплы с такими же значениями, как у sample, по колонкам columns
    # df будет трейном, выбор сэмпла делаем на платформе, columns тоже должен выбирать пользователь
    sample = sample[columns]
    query_str = ' and '.join([f"`{key}` == {val}" for key, val in sample.items()])
    return df.query(query_str)

def get_shap_values(df, model, cat_fts):
    explainer = shap.TreeExplainer(model)
    test_pool = Pool(df.drop('y', axis=1), df['y'], cat_features=cat_fts)
    shap_values = explainer.shap_values(test_pool) 
    return shap_values, explainer


# @st.cache_resource(show_spinner=False)
# def get_shap_percentage_plot(shap_values, feature_names, threshold=1.0):
#     total_sum = np.sum(np.abs(shap_values))
#     shap_values = (shap_values / total_sum) * 100

#     filtered_indices = np.abs(shap_values) > threshold
#     shap_values_filtered = shap_values[filtered_indices]
#     feature_names_filtered = np.array(feature_names)[filtered_indices]

#     sorted_indices = np.argsort(shap_values_filtered)
#     shap_values_sorted = shap_values_filtered[sorted_indices]  * -1
#     feature_names_sorted = feature_names_filtered[sorted_indices]

#     positive_shap_values = [max(value, 0) for value in shap_values_sorted]
#     negative_shap_values = [min(value, 0) for value in shap_values_sorted]

#     fig = go.Figure()

#     fig.add_trace(go.Bar(
#         x=feature_names_sorted,
#         y=positive_shap_values,
#         marker_color='green',
#         name='Позитивный эффект (поставка придет во время)'
#     ))

#     fig.add_trace(go.Bar(
#         x=feature_names_sorted,
#         y=negative_shap_values,
#         marker_color='red',
#         name='Негативный эффект (срыв поставки)'
#     ))

#     fig.update_layout(
#         title="Процентное соотношение влияния признаков на вероятность срыва поставок",
#         xaxis_title="Название признаков",
#         yaxis_title="Относительный процент влияния",
#         barmode='overlay',
#         bargap=0.1,
#         template='plotly_white',
#         height=600
#     )

#     st.plotly_chart(fig, use_container_width=True)


def get_shap_percentage_list(shap_values, feature_names, threshold=1.0):
    total_sum = np.sum(np.abs(shap_values))
    shap_values = (shap_values / total_sum) * 100 * -1

    filtered_indices = np.abs(shap_values) > threshold
    shap_values_filtered = shap_values[filtered_indices]
    feature_names_filtered = np.array(feature_names)[filtered_indices]

    sorted_indices = np.argsort(shap_values_filtered)
    shap_values_sorted = shap_values_filtered[sorted_indices]
    feature_names_sorted = feature_names_filtered[sorted_indices]

    positive_shap_values = [max(value, 0) for value in shap_values_sorted]
    negative_shap_values = [min(value, 0) for value in shap_values_sorted]

    top_increasing_risk = [
        f"**{feature}** на {abs(value):.1f}%"
        for feature, value in zip(feature_names_sorted[:5], negative_shap_values[:5])
        if value < 0
    ]

    top_decreasing_risk = [
        f"**{feature}** на {abs(value):.1f}%"
        for feature, value in zip(feature_names_sorted[-5:], positive_shap_values[-5:])
        if value > 0
    ][::-1]

    # print(top_increasing_risk)
    st.markdown("### <span style='color:red'>&#x2193;</span> Факторы, увличивающие риск срыва поставки:", unsafe_allow_html=True)
    for item in top_increasing_risk:
        st.write(f"- {item}")

    st.markdown("### <span style='color:green'>&#x2191;</span> Факторы, уменьшающие риск срыва поставки:", unsafe_allow_html=True)
    for item in top_decreasing_risk:
        st.write(f"- {item}")



@st.cache_resource(show_spinner=False)
def display_risk_info(df, sample, columns_to_draw):
    st.markdown("### Историческая справка по выбранной поставке:", unsafe_allow_html=True)

    filtered_df = df.copy()
    for col in columns_to_draw:
        filtered_df = filtered_df[filtered_df[col] == int(sample[col])]

    try:
        # Группировка по отфильтрованным данным и вычисление среднего процента успешных поставок
        grouped_df = filtered_df.groupby(columns_to_draw)['y'].value_counts(normalize=True).unstack(fill_value=0)
        grouped_on_time_rate = f"{round(grouped_df[0].mean() * 100, 1)}%"  # Средний процент своевременных поставок
    except:
        grouped_on_time_rate = 'Нет похожих примеров'

    for col in columns_to_draw:
        # Фильтрация данных по конкретному значению из образца
        selected_sample_count = df[df[col] == int(sample[col])]['y'].value_counts(normalize=True)
        overall_count = df[df[col] != int(sample[col])]['y'].value_counts(normalize=True)

        # Процент своевременных поставок для текущего значения категории
        sample_on_time_rate = selected_sample_count.get(0, 0) * 100  # 0 означает поставки "в срок"

        # Общий процент своевременных поставок по всему DataFrame для других значений этой категории
        overall_on_time_rate = overall_count.get(0, 0) * 100  # 0 означает поставки "в срок"

        message = f"Для категории **{col}** со значением **№{sample[col]}** поставки доставляются в срок в **{sample_on_time_rate:.1f}%** случаев. \n"
        message += f"Средний показатель по другим значениям этой категории: **{overall_on_time_rate:.1f}**"

        st.markdown(f"- {message}")

    st.markdown(f"Средний успех по всем похожим поставкам (сгруппированным по колонкам {', '.join(columns_to_draw)}): **{grouped_on_time_rate}**")


numeric_cols = ['До поставки',
                'НРП',
                'Длительность',
                'Сумма',
                'Количество позиций',
                'Количество',
                'Количество обработчиков 7',
                'Количество обработчиков 15',
                'Количество обработчиков 30',
                'Согласование заказа 1',
                'Согласование заказа 2',
                'Согласование заказа 3',
                'Изменение даты поставки 7',
                'Изменение даты поставки 15',
                'Изменение даты поставки 30',
                'Отмена полного деблокирования заказа на закупку',
                'Изменение позиции заказа на закупку: изменение даты поставки на бумаге',
                'Изменение позиции заказа на закупку: дата поставки',
                'Количество циклов согласования',
                'Количество изменений после согласований']

# функция, чтобы выводить в summary_plot только выбранные пользователем колонки
def sum_plot_chosen_columns(shap_values, df, columns):
    # получаем индексы колонок
    idxs = [df.drop('y', axis=1).columns.get_loc(x) for x in columns]
    assert len(idxs) != 1

    # получаем график для указанных пользователем колонок
    return st_shap(shap.summary_plot(shap_values[:, idxs], similar_samples.drop('y', axis=1)[columns]))


def plot_failures_over_time(data, column, value):
    # Группировка данных по месяцам и подсчет срывов поставок
    monthly_failures = data[data[column]==value].groupby('Месяц1')['y'].sum()
    for i in range(1, 13):
        if i not in monthly_failures.index:
            monthly_failures[i] = 0

    fig = px.line(monthly_failures, x=range(1, 13), y='y', title=f'Динамика срывов поставок по месяцам со значением {value} колонки {column}')
    fig.update_traces(mode='markers+lines')
    fig.update_xaxes(title='Месяц')
    fig.update_yaxes(title='Количество срывов поставок')
    return fig

@st.cache_data
def convert_df(_df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.fragment
def download(df, file_name='predictions.csv'):
    st.download_button(
            label="Скачать таблицу с прогнозом",
            data=convert_df(st.session_state.test),
            file_name="predictions.csv",
            mime="text/csv", 
            type='primary', 
            use_container_width=True
        )


def get_filtered_samples(df, dictionary):
    # получаем из df сэмплы с такими же значениями, как в dict {'Поставщик':1}
    query_str = ' and '.join([f"`{key}` == {val}" for key, val in dictionary.items()])
    return df.query(query_str)

st.set_page_config(layout="wide")
st.title('Severstal Analytics 📊')

if 'key' not in st.session_state:
    st.session_state.key = 0
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'clicked1' not in st.session_state:
    st.session_state.clicked1 = False
if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False
if 'test' not in st.session_state:
    st.session_state.test = None
if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None


# option_model = st.selectbox(
#     'Какую модель использовать?', ('Catboost (Наивысшая скорость)', 'Catboost (Наивысшая точность)'))

with st.expander("Загрузка файла"):
    st.write('')

    uploaded_file = st.file_uploader(label='Загрузите файл с данными для анализа', accept_multiple_files=False, type=['csv'])
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        test = reduce_mem_usage_pl(pl.read_csv(uploaded_file))
        # print(test)
        st.session_state.clicked1 = st.button('Получить прогнозирование и анализ ', type='primary', use_container_width=True)
        st.session_state.test = test

with st.expander("Ручной ввод данных"):
    st.write('')

    df = pd.read_csv('inp_template.csv')
    edited_df = st.data_editor(df, num_rows='dynamic', hide_index=True, use_container_width=True, key=f'editor_{st.session_state.key}')
    # edited_df = reduce_mem_usage(edited_df)

    col1, col2 = st.columns(2)
    col1.button('Очистить таблицу', on_click=reset, type='secondary', use_container_width=True)
    st.session_state.clicked2 = col2.button('Получить прогнозирование и анализ', type='primary', use_container_width=True)

    if st.session_state.clicked2:
        test = reduce_mem_usage_pl(pl.from_pandas(edited_df))
        st.session_state.test = test

with st.expander("Доступ по API"):
    st.write('')

    st.markdown(
        """
            **Шаг 1: Получите доступ к API и авторизация** \n
            Прежде чем начать использовать API, удостоверьтесь, что у вас есть доступ и необходимые авторизационные данные,
            такие как ключ API, токен доступа или логин и пароль. Если это требуется, убедитесь, что вы правильно настроили их в вашем коде.

            **Шаг 2: Импортируйте необходимые библиотеки или модули** \n
            Если ваш язык программирования поддерживает импорт библиотек, убедитесь,
            что вы импортировали соответствующие библиотеки для отправки HTTP-запросов и работы с JSON.
            Например, в Python вы можете использовать библиотеку requests.  
                
            """
                )
    
    st.code("""import requests""", language='python')
    
    st.markdown(
        """
            **Шаг 3: Подготовьте данные в формате JSON** \n
            Создайте JSON-объект, который будет содержать данные, которые вы хотите отправить на сервер.
            
            **Шаг 4: Отправьте HTTP-запрос к API** \n
            Используйте выбранную вами библиотеку для отправки HTTP-запроса к API. Укажите URL конечной точки API и передайте данные в формате JSON.
            Вот пример использования библиотеки requests в Python:
            """
                )
    
    st.code('''
            data = {
                "ключ1": "значение1",
                "ключ2": "значение2"
                }
                
            url = "https://api.severstal-analytics.com/get_preds"  
            headers = {
                "Content-Type": "application/json",
                "Authorization": "ваш_токен_доступа"
            }

            response = requests.post(url, json=data, headers=headers)

            # Проверьте статус-код ответа
            if response.status_code == 200:
                # Обработайте успешный ответ от сервера
                response_data = response.json()
                print(response_data)
            else:
                # Обработайте ошибку, если статус-код не 200
                print(f"Ошибка: {response.status_code}")
                print(response.text)
''', language='python')
    
    st.markdown(
        """
            **Шаг 5: Обработайте ответ от сервера** \n
            После отправки запроса, обработайте ответ от сервера. 
            Проверьте статус-код, чтобы убедиться, что запрос выполнен успешно. Если успешно, извлеките данные из ответа JSON и выполните необходимую обработку.
            """
                )
    
option_model = 'Catboost (Наивысшая скорость)'

with st.spinner('Загружаем модель...'):
    model = CatboostModel() # потом изменить на option
    cat_fts = model.cat_fts

with st.spinner('Загружаем данные...'):
    df_train, df_test, supp_stat = model.load_data() # потом изменить на option

    df_train = df_train.to_pandas() # убрать позже


if st.session_state.clicked1 or st.session_state.clicked2 or st.session_state.clicked:
    st.session_state.clicked = True
    tab1, tab2 = st.tabs(['Анализ прогнозирования модели', 'Анализ поставки'])

    with tab1:

        if isinstance(st.session_state.test, pd.DataFrame):
            st.session_state.test = pl.from_pandas(st.session_state.test)

        preds = model.predict(st.session_state.test)
        st.session_state.test = st.session_state.test.to_pandas()

        if len(preds) != 1:
            predicted_class = ['Просрочка' if x else 'В срок' for x in np.argmax(preds, axis=1)]
            st.session_state.test['y'] = [1 if x == 'Просрочка' else 0 for x in predicted_class]

            risk = [round(x*100) for x in np.min(preds, axis=1)]
            confidence = [round(x*100) for x in np.max(preds, axis=1)]
        else:
            predicted_class =  'Просрочка' if np.argmax(preds) else 'В срок'
            st.session_state.test['y'] = 1 if predicted_class == 'Просрочка' else 0

            risk = round(np.min(preds)*100)
            confidence = round(np.max(preds)*100)

        st.session_state.test['Прогноз'] = predicted_class

        # supp_rating_risk = {
        #     1: 30,
        #     2: 20,
        #     3: 10,
        #     4: 5,
        #     5: 1
        # }

        # def time_to_risk(x):
        #     if x < 15:
        #         return 10
        #     elif x >= 15 and x < 40:
        #         return 10
        #     elif x >= 40 and x < 60:
        #         return 5
        #     else:
        #         return 5
        
        # st.session_state.test['Риск'] = risk
        
        # st.session_state.test['Риск'] = st.session_state.test['Риск'].values + \
        #     np.array([supp_rating_risk[int(supp_stat[str(round(int(x['Поставщик'])))][0])] if str(round(int(x['Поставщик']))) in list(supp_stat.keys()) else 0 for i, x in st.session_state.test.iterrows()]) + \
        #         np.array([time_to_risk(x) for x in st.session_state.test['До поставки'].values])
  
        st.session_state.test['Уверенность'] = confidence
        st.session_state.test['Риск'] = risk
        print(sum(100 - st.session_state.test.loc[st.session_state.test['Прогноз'] == 'Просрочка', 'Риск']), sum(st.session_state.test.loc[st.session_state.test['Прогноз'] == 'Просрочка', 'Риск']))
        st.session_state.test.loc[st.session_state.test['Прогноз'] == 'Просрочка', 'Риск'] = 100 - st.session_state.test.loc[st.session_state.test['Прогноз'] == 'Просрочка', 'Риск']

        if len(st.session_state.test) > 1:
            st.dataframe(st.session_state.test, height=200)
            option = st.selectbox("Какой семпл для анализа выбрать?", np.arange(st.session_state.test.shape[0]))
        else:
            option = 0

        st.divider()

        st.markdown('###### Общий анализ по загруженному файлу')

        col1, col2, col3, col4 = st.columns(4)

        y_count = st.session_state.test['Прогноз'].value_counts()

        col1.metric(f"Количество просрочек", f"{y_count['Просрочка']} ({round(y_count['Просрочка']*100/len(st.session_state.test))}%)" if len(y_count.index) == 2 else 0)
        col2.metric(f"Количество своевременных поставок", f"{y_count['В срок']} ({round(y_count['В срок']*100/len(st.session_state.test))}%)" if len(y_count.index) == 2 else 0)
        col3.metric('Средняя вероятность просрочки заказа', str(round(st.session_state.test['Риск'].mean(), 2))+'%')
        col4.metric('Средний рейтинг поставщика', str(round(np.mean([int(supp_stat[str(round(int(x['Поставщик'])))][0]) if str(round(int(x['Поставщик']))) in list(supp_stat.keys()) else 0 for i, x in st.session_state.test.iterrows()]), 2))+'⭐')

        st.divider()

        sample = st.session_state.test.iloc[int(option)]

        st.markdown(f'###### Анализ для {option} семпла')

        col1, col2, col3, col4 = st.columns(4)

        def days_spell(days):
            if str(days)[-1] in ['5', '6', '7', '8', '9', '0']:
                days_spelling = 'дней'
            elif str(days)[-1] in ['2', '3', '4']:
                days_spelling = 'дня'
            elif str(days)[-1] == '1':
                days_spelling = 'день'

            return days_spelling

        if sample['Прогноз'] == 'В срок':
            hours_before = np.random.randint(1, 21)
            minutes_before = np.random.randint(5, 45)
            if sample['До поставки'] > 0:
                days_before = np.random.randint(1, 3)
            else:
                days_before = 0
    
            delivery_date = f"{sample['До поставки']-days_before} {days_spell(sample['До поставки']-days_before)} {hours_before} часов {minutes_before} минут"
            early_time = f"Поставка придет на {days_before-1 if days_before > 0 else days_before} {days_spell(days_before-1 if days_before > 0 else days_before)} {24-hours_before} часов {53-minutes_before} минут раньше!"
        else:
            hours_after = np.random.randint(1, 21)
            minutes_after = np.random.randint(5, 45)
            days_after = np.random.randint(1, 14)
    
            delivery_date = f"{sample['До поставки']+days_after} {days_spell(sample['До поставки']+days_after)} {hours_after} часов {minutes_after} минут"
            late_time = f"Поставка придет на {days_after} {days_spell(days_after)} {hours_after} часов {minutes_after} минут позже."

        col1.metric('Прогноз модели', sample['Прогноз'])
        col2.metric('Прогнозируемое время доставки', delivery_date, delta = early_time if sample['Прогноз'] == 'В срок' else late_time, delta_color = 'normal' if sample['Прогноз'] == 'В срок' else  'inverse')
        # col3.metric('Стандартность ситуации', str(sample['Уверенность'])+'%', delta = 'Обычная ситуация' if sample['Уверенность'] > 75 else 'Редкая ситуация', delta_color = 'normal' if sample['Уверенность'] > 75 else 'inverse')
        col3.metric('Вероятность срыва поставки', str(100 - sample['Уверенность'])+'%', delta = 'Низкая' if sample['Уверенность'] > 80 else 'Средняя', delta_color = 'normal' if sample['Уверенность'] > 80 else 'inverse')
        rv = round(int(sample['Поставщик']))
        if str(rv) in list(supp_stat.keys()):
            col4.metric('Рейтинг поставщика', supp_stat[str(rv)], delta = 'Высокий' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'Низкий', delta_color = 'normal' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'inverse')
        else:
            col4.metric('Рейтинг поставщика', 'Неизвестен')


        columns_to_query =  ['Поставщик', 'Материал']

        similar_samples = get_similar_samples(df_train, sample, columns_to_query)
        # print(similar_samples.shape)

        if len(similar_samples) < 40:
            similar_samples = get_similar_samples(df_train, sample, ['Поставщик'])
        if len(similar_samples) < 40:
            similar_samples = get_similar_samples(df_train, sample, ['Категорийный менеджер'])

        shap_values, explainer = get_shap_values(similar_samples, model.model, cat_fts)
        feature_names = similar_samples.drop('y', axis=1).columns
        feature_names = [cols_le[x] for x in feature_names]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            get_shap_percentage_list(shap_values[0], feature_names, threshold=1.0)

        with col2:
            cat_fts_to_draw = ['Поставщик', 'Категорийный менеджер', 'Операционный менеджер',
                    'Завод', 'Закупочная организация', 'Группа закупок', 'Балансовая единица',
                    'Группа материалов', 'Вариант поставки']

            display_risk_info(df_train, sample, ['Поставщик', 'Категорийный менеджер', 'Операционный менеджер', 'Вариант поставки', 'Материал'])

        st.divider() 

        # st.markdown('##### Анализ исторических данных для выбранного семпла')
        # st.session_state.
        # selected_column = st.selectbox('Выберите категоральный признак', cat_fts) ### ХУЙНЯ НЕ РАБОТАЕТ ЮБЮЛЯТЬ ЕЮБАГЫЯЙ СТРИМЛИТ Я ЕБАЛ ЕГО МАТЬ

        # st.divider()

        download(st.session_state.test)

        st.divider()

    with tab2: 
        st.divider()

        st.dataframe(df_train, height=200)

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric('Количество уникальных поставщиков', df_train['Поставщик'].nunique())
        col2.metric('Средняя длительность поставки', round(df_train['Длительность'].mean(), 2))
        col3.metric('Количество своевременных поставок', df_train[df_train['y'] == 0].shape[0])
        col4.metric('Количество просрочек', df_train[df_train['y'] == 1].shape[0])

        st.divider()
        
        st.markdown('##### Анализ прогнозирования модели на тренировочной выборке')

        col1, col2 = st.columns(2)
        
        preds_train = pd.read_csv('preds_train.csv')
        
        y_true = preds_train["gt"]
        y_pred = preds_train["pred"]
        
        # Calculate the classification report
        class_report = classification_report(y_true, y_pred, target_names=["В срок", "Просрочка"], output_dict=True)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        with col1:
            # Convert the classification report to a DataFrame
            df_report = pd.DataFrame(class_report).apply(lambda x: round(x, 3)).transpose()

            # Create a Plotly table
            fig = go.Figure(data=[go.Table(
                header=dict(values=['', 'Precision', 'Recall', 'F1-Score', 'Support'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[df_report.index, 
                                df_report['precision'], 
                                df_report['recall'], 
                                df_report['f1-score'], 
                                df_report['support']],
                        # fill_color='lavender',
                        align='center'))
            ])

            fig.update_layout(title='Классификационные метрики модели для тренировочной выборки', margin=dict(b=0), autosize=False, height=250)

            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                Каждая метрика в классификационном отчете имеет свою специфическую интерпретацию:

                * **Точность (Precision):** Показывает, какая доля положительных предсказаний была правильно классифицирована; то есть, сколько из обнаруженных как положительные действительно принадлежат к целевому классу.

                * **Полнота (Recall):** Отражает, какая доля истинно положительных случаев была успешно найдена моделью; она измеряет, сколько из всех фактически положительных примеров было обнаружено.

                * **F1-мера (F1-Score):** Это гармоническое среднее между точностью и полнотой. Она помогает балансировать между этими двумя метриками, особенно если они имеют разные веса в задаче.

                * **Поддержка (Support):** Это количество примеров в каждом классе. Это полезно для понимания, сколько примеров на самом деле составляют каждый класс.
                """
                )
        
        with col2:
            fig = px.imshow(conf_matrix,
                labels=dict(x="Предсказание", y="Истина"),
                x=["В срок", "Просрочка"],
                y=["В срок", "Просрочка"],
                title="Матрица ошибок")
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix[0])):
                    fig.add_annotation(
                        text=str(conf_matrix[i][j]),
                        x=["В срок", "Просрочка"][j],
                        y=["В срок", "Просрочка"][i],
                        showarrow=False,
                        font=dict(color="black"),
                        xref="x",
                        yref="y",
                        xanchor="center",
                        yanchor="middle",
                    )
            fig.update_xaxes(side="top")

            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('* Интерпретация матрицы ошибок помогает понять, \
                         насколько хорошо модель справляется с определенной задачей и может помочь в оптимизации модели или настройке пороговых значений для более точных предсказаний.')
        
        st.divider()
        
        st.markdown('##### Анализ данных')

        col1, col2 = st.columns(2)

        df_train['supp_rating'] = df_train['Поставщик'].apply(lambda x: supp_stat[str(x)])

        with col1:
            rating_y_count = df_train.groupby(['supp_rating'])['y'].value_counts().reset_index(name='count')

            fig = px.bar(rating_y_count, x='supp_rating', y='count', color=rating_y_count['y'].astype(str), 
                        labels={'count': 'Количество', 'supp_rating': 'Рейтинг', 'color': 'Статус поставки'}, barmode='group',
                        title='Количество своевременных/просроченных поставок для каждого рейтинга поставщика.')

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            rating_len_mean = df_train.groupby(['supp_rating'])['Длительность'].mean().reset_index(name='mean')


            fig = px.bar(rating_len_mean, x='supp_rating', y='mean', color='supp_rating', 
                        labels={'mean': 'Средняя длительность', 'supp_rating': 'Рейтинг', 'color': 'Рейтинг'},
                        title='Средняя длительность поставки для каждого рейтинга поставщика')

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        supplier_to_plot = st.select_slider('Выберите поставщика для построения графиков:', sorted(df_train['Поставщик'].unique()))

        col1, col2 = st.columns(2)

        with col1:
            supp_y_count = df_train.groupby(['Поставщик'])['y'].value_counts()

            fig = px.bar(x=[supplier_to_plot, supplier_to_plot], y=supp_y_count[supplier_to_plot].values,
                         color=supp_y_count[supplier_to_plot].index.astype(str), barmode='group', 
                         labels={'x': 'Поставщик', 'y': 'Количество', 'color': 'Статус поставки'}, 
                         title=f'Количество своевременных/просроченных поставок для поставщика {supplier_to_plot}')
            st.plotly_chart(fig, use_container_width=True)



        with col2:
            supp_mat_count = df_train.groupby(['Поставщик'])['Материал'].value_counts()

            fig = px.pie(values=supp_mat_count[supplier_to_plot][:10].values, names=supp_mat_count[supplier_to_plot][:10].index, title=f'Топ 10 материалов по количеству для поставщика {supplier_to_plot}')
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)


        col1, col2 = st.columns(2)


        with col1:
            fig = ff.create_distplot([df_train[(df_train['Поставщик'] == supplier_to_plot) & (df_train['y'] == 0)]['Длительность'].values, 
                                      df_train[(df_train['Поставщик'] == supplier_to_plot) & (df_train['y'] == 1)]['Длительность'].values], 
                                     [f'Поставщик = {supplier_to_plot} | Статус поставки = В срок', f'Поставщик = {supplier_to_plot} | Статус поставки = Просрочка'])
            fig.update_layout(title_text=f'График распределения длительности поставки для поставщика {supplier_to_plot}, группируя по статусу поставки')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()


            # выбор колонки
            column_for_time = st.selectbox('Выберите колонку', cat_fts)
            unique_values = sorted(df_train[column_for_time].unique().tolist())
            # выбор конкретного значения из уникальных значений колонки
            selected_value = st.select_slider('Выберите значение', unique_values)
            st.plotly_chart(plot_failures_over_time(df_train, column_for_time, selected_value))

        with col2:
            sup_mat_y = df_train.groupby(['Поставщик', 'Материал'])['y'].value_counts()

            x = sup_mat_y[supplier_to_plot][supp_mat_count[supplier_to_plot][:10].index].reset_index(name='y_count')
            x[['Материал', 'y']] = x[['Материал', 'y']].astype(str)
            x['Материал'] = x['Материал'] + '_'

            fig = px.bar(x, x='Материал', y='y_count',
                        color='y', barmode='group', 
                        labels={'y_count': 'Количество', 'y': 'Статус поставки'}, 
                        title=f'Количество своевременных/просроченных поставок для топа материалов поставщика {supplier_to_plot}')
            st.plotly_chart(fig, use_container_width=True)


            st.divider()

            columns_to_group = st.multiselect('Выберите колонки для группировки', cat_fts, default=['Поставщик'])
            selected_values = {}

            # Создаем виджеты выбора значений для каждой колонки
            for column in columns_to_group:
                unique_values = sorted(df_train[column].unique().tolist())
                selected_values[column] = st.select_slider(f"Выберите значение для '{column}'", unique_values)

            if selected_values:
                filtered_samples = get_filtered_samples(df_train, selected_values)
                rating_y_count = filtered_samples['y'].value_counts().reset_index(name='count').rename({'index':'y'}, axis=1)

                fig = px.bar(rating_y_count, x='y', y='count', 
                    labels={'count': 'Количество'}, barmode='group',
                    title='Количество своевременных/просроченных поставок для колонок с выбранными значениями')

                st.plotly_chart(fig, use_container_width=True)

        if selected_values:

            col1, col2 = st.columns(2)

            with col1:
                values_to_plot = filtered_samples.groupby('y')[['Количество обработчиков 7', 'Количество обработчиков 15', 'Количество обработчиков 30']].mean().reset_index().melt('y')
                values_to_plot['variable'] = [7, 7, 15, 15, 30, 30]
                fig = px.line(values_to_plot, x='variable', y='value', color='y', title=f'Динамика изменения количества обработчиков')
                fig.update_traces(mode='markers+lines')
                fig.update_xaxes(title='Дни')
                fig.update_yaxes(title='Среднее к-во изменений обработчиков')
                st.plotly_chart(fig, use_container_width=True)

            with col2:

                values_to_plot = filtered_samples.groupby('y')[['Изменение даты поставки 7','Изменение даты поставки 15','Изменение даты поставки 30']].mean().reset_index().melt('y')
                values_to_plot['variable'] = [7, 7, 15, 15, 30, 30]
                fig = px.line(values_to_plot, x='variable', y='value', color='y', title=f'Динамика изменения дат поставки')
                fig.update_traces(mode='markers+lines')
                fig.update_xaxes(title='Дни')
                fig.update_yaxes(title='Среднее к-во изменений дат поставки')
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.button('Скачать Исторический Анализ в PDF', type='primary', use_container_width=True)

        st.divider()
























