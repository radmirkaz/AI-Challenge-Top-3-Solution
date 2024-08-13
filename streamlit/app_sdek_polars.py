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
import json
from scipy import stats
from phik.report import plot_correlation_matrix
from phik import report
import plotly.figure_factory as ff
import pandas as pd


def reset():
    st.session_state.key += 1


@st.cache_resource(show_spinner=True)
class CatboostModel:
    def __init__(self, model_type='Catboost (Наивысшая скорость)'):
        self.model_type = model_type
        self.model, self.cat_fts = self.load_model()
        self.train_data = None
        self.test_data = None
    
    def load_model(self, model_path='fast_catboost.cb'):
        cat_fts = ['Поставщик', 'Материал', 'Категорийный менеджер', 'Операционный менеджер',
                    'Завод', 'Закупочная организация', 'Группа закупок', 'Балансовая единица', 'ЕИ',
                    'Группа материалов', 'Вариант поставки', 'Месяц1', 'Месяц2', 'Месяц3',
                    'День недели 2']
        
        if self.model_type == 'Catboost (Наивысшая точность)':
            model = catboost.CatBoostClassifier(
                depth=10,
                iterations=5000,
                learning_rate=0.035,
                random_seed=42,
                eval_metric='AUC',
                cat_features=cat_fts
            )
            model.load_model(model_path)
        elif self.model_type == 'Catboost (Наивысшая скорость)':
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
        df_train = pl.read_csv('data/train_AIC.csv')
        df_test = pl.read_csv('data/test_AIC.csv')

        df_train = df_train.with_columns([
            pl.col('Поставщик').cast(pl.Int16),
            pl.col('Материал').cast(pl.UInt16),
            pl.col('Категорийный менеджер').cast(pl.Int8),
            pl.col('Операционный менеджер').cast(pl.Int8),
            pl.col('Завод').cast(pl.Int8),
            pl.col('Закупочная организация').cast(pl.Int8),
            pl.col('Группа закупок').cast(pl.Int16),
            pl.col('Балансовая единица').cast(pl.Int8),
            pl.col('ЕИ').cast(pl.Int8)
        ])
        
        df_train = self.process_data(df_train, self.model_type)
        df_test = self.process_data(df_test, self.model_type)

        self.train_data = df_train
        self.test_data = df_test
            
        return df_train, df_test, self.load_supp_stat()
    
    def process_data(self, df: pl.DataFrame, model_type='Catboost (Наивысшая скорость)') -> pl.DataFrame:
        if self.model_type == 'Catboost (Наивысшая точность)':
            df = preprocess_df(df)
            df = make_gold_fts(df)
            df = make_fts(df, self.train_data).fillna(-1)
        elif self.model_type == 'Catboost (Наивысшая скорость)':
            df = df.with_columns([
                (pl.col('До поставки') - pl.col('Длительность')).alias('diff'),
            ]).fill_null(-1)

        return df

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if len(df.shape) == 1:
            df = pl.DataFrame([df])
            
        df = self.process_data(df, self.model_type)
        df = df.with_columns([
            pl.col(self.cat_fts).cast(pl.Utf8)
        ])
        df_pandas = df.to_pandas()  # Convert to pandas for CatBoost compatibility
        
        preds = self.model.predict_proba(Pool(df_pandas[self.model.feature_names_], cat_features=self.cat_fts))
        return preds


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
            label="Скачать таблицу с предсказаниями",
            data=convert_df(st.session_state.test),
            file_name="predictions.csv",
            mime="text/csv", 
            type='secondary', 
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

option_model = st.selectbox(
    'Какую модель использовать?', ('Catboost (Наивысшая скорость)', 'Catboost (Наивысшая точность)'))

with st.expander("Загрузка файла"):
    st.write('')

    uploaded_file = st.file_uploader(label='Загрузите файл с данными для анализа', accept_multiple_files=False, type=['csv'])
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        test = pl.read_csv(uploaded_file)
        print(test)
        st.session_state.clicked1 = st.button('Получить предсказания и анализ ', type='primary', use_container_width=True)
        st.session_state.test = test

with st.expander("Ручной ввод данных"):
    st.write('')

    df = pd.read_csv('inp_template.csv')
    edited_df = st.data_editor(df, num_rows='dynamic', hide_index=True, use_container_width=True, key=f'editor_{st.session_state.key}')

    col1, col2 = st.columns(2)
    col1.button('Очистить таблицу', on_click=reset, type='secondary', use_container_width=True)
    st.session_state.clicked2 = col2.button('Получить предсказания и анализ', type='primary', use_container_width=True)

    if st.session_state.clicked2:
        test = pl.from_pandas(edited_df)
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
    model = CatboostModel(model_type=option_model) # потом изменить на option
    cat_fts = model.cat_fts

with st.spinner('Загружаем данные...'):
    df_train, df_test, supp_stat = model.load_data() # потом изменить на option

    df_train = df_train.to_pandas() # убрать позже


if st.session_state.clicked1 or st.session_state.clicked2 or st.session_state.clicked:
    st.session_state.clicked = True
    tab1, tab2 = st.tabs(['Анализ предсказаний модели', 'Исторический анализ'])

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

        st.session_state.test['Предсказание'] = predicted_class
        
        supp_rating_risk = {
            1: 30,
            2: 20,
            3: 10,
            4: 5,
            5: 1
        }
        
        def time_to_risk(x):
            if x < 15:
                return 10
            elif x >= 15 and x < 40:
                return 10
            elif x >= 40 and x < 60:
                return 5
            else:
                return 5
        
        st.session_state.test['Риск'] = risk
        
        st.session_state.test['Риск'] = st.session_state.test['Риск'].values + \
            np.array([supp_rating_risk[int(supp_stat[str(round(int(x['Поставщик'])))][0])] if str(round(int(x['Поставщик']))) in list(supp_stat.keys()) else 0 for i, x in st.session_state.test.iterrows()]) + \
                np.array([time_to_risk(x) for x in st.session_state.test['До поставки'].values])
  
        st.session_state.test['Уверенность'] = confidence

        if len(st.session_state.test) > 1:
            st.dataframe(st.session_state.test, height=200)
            option = st.slider("Какой семпл для анализа выбрать?", 0, len(st.session_state.test), 1)
        else:
            option = 0

        st.divider()

        st.markdown('###### Общий анализ')

        col1, col2, col3, col4 = st.columns(4)

        y_count = st.session_state.test['Предсказание'].value_counts()

        col1.metric('Количество просрочек', y_count['Просрочка'] if len(y_count.index) == 2 else 0)
        col2.metric('Количество своевременных поставок', y_count['В срок'] if len(y_count.index) == 2 else 0)
        col3.metric('Средний риск', str(round(st.session_state.test['Риск'].mean(), 2))+'%')
        col4.metric('Средний рейтинг поставщика', str(round(np.mean([int(supp_stat[str(round(int(x['Поставщик'])))][0]) if str(round(int(x['Поставщик']))) in list(supp_stat.keys()) else 0 for i, x in st.session_state.test.iterrows()]), 2))+'⭐')

        st.divider()

        sample = st.session_state.test.iloc[int(option)]

        st.markdown(f'###### Анализ для {option} семпла')

        col1, col2, col3, col4 = st.columns(4)

        col1.metric('Предсказание модели', '0' if sample['Предсказание'] == 'В срок' else '1', delta = 'Товар поступит в срок!' if sample['Предсказание'] == 'В срок' else 'Товар задержится', delta_color = 'normal' if sample['Предсказание'] == 'В срок' else 'inverse')
        col2.metric('Риск', str(sample['Риск'])+'%', delta = 'Низкий' if sample['Риск'] < 30 else 'Высокий', delta_color = 'normal' if sample['Риск'] < 30 else 'inverse')
        col3.metric('Уверенность модели', str(sample['Уверенность'])+'%', delta = 'Высокая' if sample['Уверенность'] > 60 else 'Слабая', delta_color = 'normal' if sample['Уверенность'] > 60 else 'inverse')
        rv = round(int(sample['Поставщик']))
        if str(rv) in list(supp_stat.keys()):
            col4.metric('Рейтинг поставщика', supp_stat[str(rv)], delta = 'Высокий' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'Низкий', delta_color = 'normal' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'inverse')
        else:
            col4.metric('Рейтинг поставщика', 'Неизвестен')


        st.divider()

        columns_to_query =  ['Поставщик', 'Материал']

        similar_samples = get_similar_samples(df_train, sample, columns_to_query)
        print(similar_samples.shape)

        if len(similar_samples) < 40:
            similar_samples = get_similar_samples(df_train, sample, ['Поставщик'])
        if len(similar_samples) < 40:
            similar_samples = get_similar_samples(df_train, sample, ['Категорийный менеджер'])

        shap_values, explainer = get_shap_values(similar_samples, model.model, cat_fts)
        feature_names = similar_samples.drop('y', axis=1).columns
        feature_names = [cols_le[x] for x in feature_names]

        st.markdown("### 📚 Как читать эти графики?")

        col1, col2 = st.columns(2)

        col1.markdown("""* Значения слева от центральной вертикальной линии — это **negative** класс (0 - Поставка произойдет в срок), справа — **positive** (1 - Поставка будет просрочена) \n* Чем толще линия на графике, тем больше таких точек наблюдения. \n* Чем краснее точки на графике, тем выше значения фичи в ней.""")
        col2.markdown("""* График помогает определить, какие признаки оказывают наибольшее влияние на вероятность задержки поставки и в какую сторону – положительную или отрицательную. \n* Длина столбца - это величина вклада этого фактора. Положительная высота означает, что фактор увеличивает предсказание, а отрицательная - уменьшает. \n* Признаки упорядочены сверху вниз в порядке влияния на предсказание.""")

        # дальше выводим разные графики по полученным shap values (можно указывать height и width аргументами в st_shap)
        col1, col2 = st.columns(2)

        with col1:
            st_shap(shap.summary_plot(shap_values, similar_samples.drop('y', axis=1), max_display=12, feature_names=feature_names), height=500)

        with col2:
            st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names), height=500)

        st.write('')
        st.write('')
        st.markdown("""###### Влияние признаков на вероятность задержки поставки - длина и количество синих стрелок уменьшает вероятность задержки, длина и количество красных стрелок увеличивает веротность просрочки.""")

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:],
                similar_samples.drop('y', axis=1).iloc[0,:], feature_names=feature_names))

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # интерактивный force plot по всем похожим сэмплам
            # shap_html = shap.force_plot(explainer.expected_value, shap_values, similar_samples.drop('y', axis=1), show=False)
            # shap_html_str = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"
            # st.components.v1.html(shap_html_str, height=500)

            fig, ax = plt.subplots()  # создаем объекты figure и axes
            perm_importance = permutation_importance(model.model, similar_samples.drop('y', axis=1), similar_samples['y'], n_repeats=5, random_state=42)
            sorted_idx = perm_importance.importances_mean.argsort()[-10:]

            try:
                data = np.array(perm_importance.importances_mean[sorted_idx])
                if (data <= 0).any():
                    data += 1e-6
                imp = stats.boxcox(data)[0]
            except:
                imp = np.array(perm_importance.importances_mean[sorted_idx])
            theta = np.array(similar_samples.drop('y', axis=1).columns)[sorted_idx]

            # print(perm_importance.importances_mean[sorted_idx])
            fig = go.Figure(data=go.Scatterpolar(r=[imp[x] for x in range(len(imp)-1, -1, -1)], theta=[theta[x] for x in range(len(imp)-1, -1, -1)]))
            fig.update_traces(fill='toself')
            fig.update_layout(polar = dict(
                              radialaxis_angle = -45,
                              angularaxis = dict(
                              direction = "clockwise",
                              period = 6)
                              ))
            fig.update_layout(title_text="Топ 10 самых важных признаков.")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('###### График использует метод перестановочной важности (permutation importance) для оценки важности признаков. \n* Чем выше значение важности, тем более важным является признак для модели. \n* Если важность признака близка к нулю или отрицательна, это может указывать на то, что данный признак слабо влияет на модель.')

        with col2:
            st.markdown("""###### График показывает влияние на предсказание в разрезе каждого признака и его значений. \n* **Ось X**: Значения признака. \n* **Ось Y**: Важность данного признака для предсказаний модели. Чем выше значение на оси Y, тем более важным является данный признак для модели.""")

            column_to_plot = st.selectbox('Выберите колонку для построения графика:', numeric_cols)
            st_shap(shap.dependence_plot(column_to_plot, shap_values, similar_samples.drop('y', axis=1), feature_names=feature_names), height=400)

        st.divider()

        st.markdown('**Корреляционная матрица** - это таблица, отображающая коэффициенты корреляции между различными переменными или признаками в наборе данных. Каждая ячейка в матрице содержит коэффициент корреляции, который измеряет силу и направление связи между двумя соответствующими признаками. Значения ближе к 1 указывают на сильную положительную корреляцию, около 0 указывают на слабую или отсутствующую корреляцию. Аналитики используют корреляционные матрицы для выявления взаимосвязей между переменными и оценки их влияния друг на друга, что полезно для анализа данных и моделирования.')

        phk_df = similar_samples.copy()
        phk_df['y'] = model.predict(pl.from_pandas(similar_samples))
        phik_overview = phk_df.phik_matrix(interval_cols=[i for i in phk_df.columns if i not in cat_fts and i != 'y']).round(3).sort_values('y')

        data = phik_overview.values
        columns = [cols_le[x] for x in phik_overview.columns]
        index = [cols_le[x] for x in phik_overview.index]

        fig = px.imshow(data, x=columns, y=index)

        fig.update_layout(xaxis_title="", yaxis_title="", height=800)

        fig.update_xaxes(tickfont=dict(size=12), tickangle=45)
        fig.update_yaxes(tickfont=dict(size=12))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('*Для рассчета матрицы используется набор данных похожий на выбранный семпл')

        st.divider()

        st.button(f'Скачать анализ для {option} семпла в PDF', type='primary', use_container_width=True)

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
        
        st.markdown('##### Анализ предсказания модели на тренировочной выборке')

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
























