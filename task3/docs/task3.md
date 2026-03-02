# Task 3: Accelerated Data Science с RAPIDS

## Часть 1: Работа с данными (cuDF, Polars, Dask)

### Основы JupyterLab и GPU

#### Мониторинг GPU с nvidia-smi

```bash
!nvidia-smi
```

Показывает: модель GPU, драйвер, CUDA-версию, использование памяти, загрузку. Полезно для отслеживания ресурсов при работе с данными на GPU.

#### Магические команды Jupyter

- `%time <statement>` — замеряет время одной строки
- `%%time` (в начале ячейки) — замеряет время всей ячейки
- `%load_ext` — загружает расширение ядра
- `!<command>` — запускает shell-команду

### cuDF — GPU DataFrame

**cuDF** — GPU-ускоренная библиотека для работы с табличными данными из экосистемы RAPIDS. API практически идентичен pandas.

#### Чтение данных

```python
import cudf

df = cudf.read_csv('./data/pop_1-01.csv')
```

#### Первичное исследование

```python
df.head()          # первые 5 строк
df.dtypes          # типы данных
df.shape           # размерность (строки, столбцы)
df.columns         # список столбцов
df.info()          # общая информация (типы, пропуски, память)
```

#### Индексирование и выбор данных

Аксессор `.loc` — выбор по меткам (индексу):
```python
df.loc[0]               # первая строка
df.loc[0:4]             # первые 5 строк
df.loc[:, 'column']     # весь столбец
df.loc[0:4, ['col1', 'col2']]  # подмножество строк и столбцов
```

#### Строковые операции

```python
df['county'] = df['county'].str.title()  # первая буква каждого слова заглавная
df['county'].str.contains('shire')       # проверка наличия подстроки
```

#### Агрегация с groupby

```python
df.groupby('county').size()          # количество записей по группе
df.groupby('county')['age'].mean()   # среднее по группе
```

#### Фильтрация

```python
df.loc[df['age'] > 50]
df.loc[(df['county'] == 'DEVON') & (df['sex'] == 'M')]
```

#### Создание новых столбцов

```python
df['density'] = df['count'] / df['area']
```

#### cudf.pandas — автоматическое ускорение

Модуль `cudf.pandas` перехватывает вызовы pandas и автоматически направляет их на GPU через cuDF. Если операция не поддерживается на GPU — откатывается на CPU:
```python
%load_ext cudf.pandas
import pandas as pd  # теперь pd автоматически использует GPU где возможно
```

### Управление памятью GPU

#### Проблема ограниченной памяти

GPU имеет ограниченную память (16–80 ГБ). Передача данных между CPU и GPU — «бутылочное горлышко». Поэтому:
- Минимизируйте передачи данных
- Оптимизируйте типы данных

#### Оптимизация типов данных

```python
# Уменьшение размера числовых данных
df['age'] = df['age'].astype('int8')        # 1 байт вместо 8
df['count'] = df['count'].astype('float32')  # 4 байта вместо 8

# Использование категориальных типов
df['county'] = df['county'].astype('category')
```

Пример экономии: строковый столбец `county` (~50 уникальных значений) из 3.1 МБ → 80 КБ при использовании `category`.

#### Проверка использования памяти

```python
df.memory_usage(deep=True)  # точное использование памяти по столбцам
```

### Интероперабельность: NumPy, CuPy, cuDF

#### CuPy — GPU-ускоренный NumPy

CuPy — аналог NumPy для GPU. Поддерживает большинство операций NumPy.

Когда использовать:
- **cuDF** — табличные данные с именованными столбцами, SQL-подобные операции
- **CuPy** — линейная алгебра, числовые массивы, математические операции

#### Преобразования между форматами

```python
# cuDF → CuPy
cupy_array = df['column'].values
cupy_array = df.to_cupy()

# CuPy → cuDF
series = cudf.Series(cupy_array)

# cuDF → NumPy (через CPU!)
numpy_array = df['column'].to_numpy()
```

**Важно:** `.to_numpy()` копирует данные с GPU на CPU. Избегайте частых конвертаций.

#### Boolean Array Indexing

CuPy поддерживает булевую индексацию, аналогичную NumPy:
```python
mask = array > 0
filtered = array[mask]
```

### Группировка данных

#### Парадигма Split-Apply-Combine

1. **Split** — разбить данные на группы по значению столбца
2. **Apply** — применить функцию к каждой группе
3. **Combine** — объединить результаты

```python
grouped = df.groupby('county')
grouped.size()       # размер каждой группы
grouped.mean()       # среднее по группам
grouped.agg({'age': 'mean', 'count': 'sum'})  # разные агрегации
```

#### Биннинг (Binning)

Разбиение непрерывных значений на дискретные группы:
```python
bins = [0, 18, 35, 50, 65, 100]
labels = ['0-17', '18-34', '35-49', '50-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

#### Pivot Table

Перекрёстная таблица для агрегации:
```python
pivot = df.pivot_table(values='count', index='county', columns='sex', aggfunc='sum')
```

### Визуализация данных

#### Типы графиков

- **Столбчатая диаграмма (Bar chart)** — сравнение категорий
- **Гистограмма** — распределение непрерывных данных
- **Scatter plot** — взаимосвязь двух переменных
- **Line chart** — изменение во времени

#### Datashader — визуализация больших данных

Datashader растеризует данные прямо на GPU — подходит для миллионов точек:
```python
import datashader as ds
canvas = ds.Canvas(plot_width=800, plot_height=600)
agg = canvas.points(df, 'x', 'y')
```

#### cuxfilter — интерактивные GPU-дашборды

```python
import cuxfilter
cux_df = cuxfilter.DataFrame.from_dataframe(df)
chart = cuxfilter.charts.scatter(x='easting', y='northing', ...)
dashboard = cux_df.dashboard([chart])
```

### ETL-процесс

1. **Extract** — загрузка данных из различных источников (`cudf.read_csv`)
2. **Transform** — очистка, преобразование, обогащение:
   - Слияние таблиц (`merge`)
   - Вычисление новых столбцов
   - Фильтрация
3. **Load** — сохранение результата (`to_parquet`)

```python
# Пример ETL
df = cudf.read_csv('population.csv')
counties = cudf.read_csv('counties.csv')
merged = df.merge(counties, on='county')
merged['distance'] = calculate_distance(merged)
merged.to_parquet('output.parquet')
```

### cuDF-Polars — GPU-ускорение для Polars

**Polars** — быстрая библиотека для работы с DataFrame, написанная на Rust:
- Быстрее pandas на CPU
- Поддерживает ленивое (lazy) и немедленное (eager) выполнение
- Оптимизатор запросов для lazy-режима

**cuDF-Polars** — GPU-бэкенд для Polars:
```python
import polars as pl

# Ленивое выполнение с GPU
lf = pl.scan_csv('data.csv')
result = lf.filter(pl.col('age') > 30).collect(engine=pl.GPUEngine())
```

Основные операции Polars:
```python
df.filter(pl.col('county') == 'DEVON')
df.sort('age', descending=True)
df.group_by('county').agg(pl.col('count').sum())
df.with_columns((pl.col('count') / pl.col('area')).alias('density'))
```

### Dask cuDF — распределённые GPU-вычисления

#### Настройка кластера

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)
```

#### Чтение данных

```python
import dask_cudf

ddf = dask_cudf.read_csv('data/*.csv')
```

#### Вычислительный граф

Dask строит граф операций. Визуализация:
```python
ddf['column'].mean().visualize()
```

Для запуска вычислений:
```python
result = ddf['column'].mean().compute()
```

#### persist() — сохранение в памяти

```python
ddf = ddf.persist()  # данные остаются в GPU-памяти
```

---

## Часть 2: Графовая аналитика (cuGraph)

### Подготовка графовых данных

Графы состоят из **узлов** (nodes) и **рёбер** (edges). Для дорожной сети Великобритании:
- Узлы — пересечения дорог с координатами
- Рёбра — дорожные сегменты с типом дороги

#### Подготовка данных для cuGraph

cuGraph требует, чтобы идентификаторы узлов были **целыми числами** от 0 до N-1:
```python
# Переиндексация строковых ID в целые числа
edges['src_id'] = edges['start_node'].astype('category').cat.codes
edges['dst_id'] = edges['end_node'].astype('category').cat.codes
```

#### Построение графа

```python
import cugraph as cg

G = cg.Graph()
G.from_cudf_edgelist(edges, source='src_id', destination='dst_id', edge_attr='length')
```

#### Добавление весов (время в пути)

Преобразование типа дороги в скорость и расчёт времени:
```python
speed_map = {'Motorway': 110, 'A Road': 80, 'B Road': 60, 'Minor Road': 40}
edges['speed_kmh'] = edges['road_type'].map(speed_map)
edges['time_sec'] = (edges['length_km'] / edges['speed_kmh']) * 3600
```

### cuGraph — GPU-графовая аналитика

#### Кратчайший путь (SSSP)

Single Source Shortest Path — находит кратчайшие пути от одного узла до всех остальных:
```python
distances = cugraph.sssp(G, source=start_node)
```

Результат — DataFrame с колонками: vertex, distance, predecessor.

#### Визуализация путей на карте

После нахождения кратчайших путей можно визуализировать их на карте через Plotly, конвертируя координаты обратно и строя `Scattergeo`.

### NetworkX с бэкендом nx-cugraph

**nx-cugraph** — GPU-бэкенд для NetworkX, позволяющий ускорить графовые алгоритмы без изменения кода.

#### Три способа использования

1. **Переменная окружения:**
```bash
NETWORKX_BACKEND_PRIORITY=cugraph python script.py
```

2. **Ключевое слово backend:**
```python
nx.betweenness_centrality(G, backend='cugraph')
```

3. **Диспетчеризация по типу:**
```python
nxcg_G = nxcg.from_networkx(G)  # конвертация в GPU-граф
nx.betweenness_centrality(nxcg_G)  # автоматически использует GPU
```

#### Алгоритмы центральности (Centrality)

- **Betweenness Centrality** — как часто узел находится на кратчайших путях между другими парами узлов
- **Degree Centrality** — нормализованное число соседей узла
- **Katz Centrality** — влияние узла с учётом длины путей (ближние связи важнее)
- **PageRank** — вероятность оказаться в узле при случайном блуждании
- **Eigenvector Centrality** — связи с важными узлами повышают важность самого узла

---

## Часть 3: Машинное обучение (cuML)

### K-Means кластеризация

**K-Means** — алгоритм кластеризации, разбивающий данные на K кластеров:
1. Выбрать K начальных центров
2. Назначить каждую точку ближайшему центру
3. Пересчитать центры как среднее назначенных точек
4. Повторять шаги 2–3 до сходимости

```python
from cuml import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(df[['easting', 'northing']])

labels = kmeans.labels_          # метки кластеров
centers = kmeans.cluster_centers_  # координаты центров
```

**Особенность:** K задаётся заранее. Результат зависит от начальной инициализации.

### DBSCAN кластеризация

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise):
- Не требует указания числа кластеров
- Находит кластеры произвольной формы
- Помечает точки в разреженных областях как шум (-1)

Ключевые параметры:
- `eps` — максимальное расстояние между соседними точками
- `min_samples` — минимальное число точек для формирования кластера

```python
from cuml import DBSCAN

dbscan = DBSCAN(eps=5000, min_samples=5)
labels = dbscan.fit_predict(df[['easting', 'northing']])
```

### Логистическая регрессия

**Логистическая регрессия** — алгоритм бинарной или мультиклассовой классификации. Предсказывает вероятность принадлежности к классу.

```python
from cuml import LogisticRegression
from cuml.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
```

#### Интерпретируемость модели

```python
model.coef_       # веса признаков
model.intercept_  # смещение (bias)
```

Знак и величина коэффициентов показывают направление и силу влияния каждого признака на предсказание.

### K-Nearest Neighbors (KNN)

**KNN** — алгоритм поиска ближайших соседей. Может использоваться как для классификации, так и для поиска:

```python
from cuml import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(road_nodes[['easting', 'northing']])

distances, indices = nn.kneighbors(hospitals[['easting', 'northing']])
```

Результат: расстояния до K ближайших соседей и их индексы.

### XGBoost

**XGBoost** (eXtreme Gradient Boosting) — один из самых популярных алгоритмов машинного обучения для табличных данных. Строит ансамбль деревьев решений.

#### Обучение модели на GPU

```python
import xgboost as xgb
from cuml.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 8,
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # GPU-ускоренный метод
    'device': 'cuda'
}

model = xgb.train(params, dtrain, num_boost_round=100)
```

#### Анализ модели

```python
xgb.plot_tree(model, num_trees=0)       # визуализация дерева
xgb.plot_importance(model)               # важность признаков
```

#### Предсказание

```python
predictions = model.predict(dtest)
```

### Развёртывание модели с Triton Inference Server

**Triton Inference Server** — сервер от NVIDIA для развёртывания ML-моделей в продакшене.

#### Подготовка модели

Структура директории:
```
models/
└── xgboost_model/
    ├── 1/
    │   └── xgboost.json
    └── config.pbtxt
```

Файл `config.pbtxt` описывает формат входных и выходных данных:
```
name: "xgboost_model"
backend: "fil"
max_batch_size: 8192
input [
  { name: "input__0", data_type: TYPE_FP32, dims: [num_features] }
]
output [
  { name: "output__0", data_type: TYPE_FP32, dims: [1] }
]
```

#### Тестирование через Triton Client

```python
import tritonclient.http as tritonhttpclient

client = tritonhttpclient.InferenceServerClient(url='localhost:8000')

inputs = tritonhttpclient.InferInput('input__0', data.shape, 'FP32')
inputs.set_data_from_numpy(data)
outputs = tritonhttpclient.InferRequestedOutput('output__0')

response = client.infer('xgboost_model', [inputs], outputs=[outputs])
predictions = response.as_numpy('output__0')
```

#### Анализ производительности

```bash
perf_analyzer -m xgboost_model -u localhost:8000 --concurrency-range 1:4
```

Показывает latency и throughput при различных уровнях параллелизма.

### Multi-GPU K-Means с Dask

Для масштабирования на несколько GPU:

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.dask.cluster import KMeans as DaskKMeans

cluster = LocalCUDACluster()
client = Client(cluster)

ddf = dask_cudf.read_csv('data/*.csv')
ddf = ddf.persist()

dkm = DaskKMeans(n_clusters=5)
dkm.fit(ddf)
labels = dkm.predict(ddf).compute()
```

---

## Вопросы и ответы

**1. Что такое cuDF и чем он отличается от pandas?**

cuDF — это GPU-ускоренная библиотека для работы с табличными данными из экосистемы RAPIDS. Её API практически идентичен pandas, но вычисления выполняются на GPU с использованием CUDA. Это позволяет обрабатывать данные в десятки раз быстрее на больших датасетах. Для маленьких данных pandas может быть предпочтительнее из-за отсутствия накладных расходов на передачу данных на GPU.

**2. Как оптимизировать использование GPU-памяти при работе с cuDF?**

Основные приёмы: использование типов данных минимальной разрядности — `int8` вместо `int64`, `float32` вместо `float64`; перевод строковых столбцов с малым количеством уникальных значений в тип `category`; минимизация передачи данных между CPU и GPU; проверка потребления через `df.memory_usage(deep=True)`. Например, категоризация строкового столбца с 50 уникальными значениями может сократить его размер в десятки раз.

**3. Что такое cudf.pandas и зачем он нужен?**

Модуль `cudf.pandas` — прокси-ускоритель, который перехватывает вызовы стандартного `import pandas as pd` и автоматически направляет совместимые операции на GPU. Если операция не поддерживается на GPU, она прозрачно откатывается на CPU. Это позволяет ускорить существующий pandas-код без его переписывания — нужно лишь добавить `%load_ext cudf.pandas` перед импортом.

**4. Что такое cuGraph и какие задачи он решает?**

cuGraph — GPU-ускоренная библиотека для графовой аналитики из экосистемы RAPIDS. Она решает задачи: поиск кратчайших путей (SSSP), анализ центральности (betweenness, PageRank, eigenvector), обнаружение сообществ, анализ связности и другие. Графы строятся из cuDF DataFrame с рёбрами и весами. cuGraph требует, чтобы идентификаторы узлов были целыми числами от 0 до N-1.

**5. Как nx-cugraph ускоряет NetworkX?**

nx-cugraph — это GPU-бэкенд для библиотеки NetworkX. Его можно подключить тремя способами: через переменную окружения `NETWORKX_BACKEND_PRIORITY=cugraph`, через аргумент `backend='cugraph'` в вызове алгоритма, или через конвертацию графа в GPU-формат с автоматической диспетчеризацией. Это позволяет ускорить существующие NetworkX-алгоритмы без изменения логики кода.

**6. В чём разница между K-Means и DBSCAN?**

K-Means требует заранее указать число кластеров K и формирует сферические кластеры вокруг центров. DBSCAN не требует задания числа кластеров — он определяет их автоматически по плотности данных, может находить кластеры произвольной формы и помечает шумовые точки как -1. K-Means лучше для равномерных кластеров, DBSCAN — для данных с произвольной формой кластеров и наличием шума.

**7. Как работает логистическая регрессия в cuML и как интерпретировать её результаты?**

Логистическая регрессия предсказывает вероятность принадлежности к классу. После обучения через `model.fit(X_train, y_train)` модель сохраняет коэффициенты `model.coef_` и смещение `model.intercept_`. Положительный коэффициент означает, что увеличение признака повышает вероятность принадлежности к классу 1, отрицательный — понижает. Чем больше абсолютное значение коэффициента, тем сильнее влияние признака.

**8. Что такое XGBoost и как его обучить на GPU?**

XGBoost — алгоритм градиентного бустинга, строящий ансамбль деревьев решений. Для GPU-обучения нужно установить параметры `tree_method='hist'` и `device='cuda'`. Данные оборачиваются в `xgb.DMatrix`, после чего модель обучается через `xgb.train()`. Результаты можно анализировать через `plot_tree` для визуализации деревьев и `plot_importance` для оценки важности признаков.

**9. Что такое Triton Inference Server и как развернуть в нём модель?**

Triton Inference Server — это сервер от NVIDIA для высокопроизводительного развёртывания ML-моделей. Для развёртывания модели нужно: создать директорию с правильной структурой (models/model_name/version/model_file), написать конфигурационный файл `config.pbtxt` с описанием входных/выходных тензоров, запустить Triton. Клиентский код отправляет данные через HTTP или gRPC. Утилита `perf_analyzer` позволяет измерить latency и throughput при разных нагрузках.

**10. Как масштабировать cuML-алгоритмы на несколько GPU?**

Через Dask CUDA. Сначала создаётся `LocalCUDACluster` и `Client`, затем данные загружаются через `dask_cudf` и сохраняются в памяти GPU через `.persist()`. Модели из `cuml.dask` (например, `cuml.dask.cluster.KMeans`) принимают Dask DataFrame и автоматически распределяют вычисления между GPU. Результат можно собрать через `.compute()`.

**11. Что такое Polars и как cuDF-Polars ускоряет его на GPU?**

Polars — библиотека для работы с DataFrame, написанная на Rust, которая быстрее pandas на CPU. Она поддерживает lazy-выполнение с оптимизатором запросов. cuDF-Polars — GPU-бэкенд для Polars, позволяющий выполнять запросы на GPU через `collect(engine=pl.GPUEngine())`. Это даёт ускорение для больших датасетов без изменения логики Polars-кода.

**12. Какие типы центральности поддерживает nx-cugraph и что они означают?**

nx-cugraph поддерживает несколько типов: Betweenness Centrality — как часто узел лежит на кратчайших путях между другими парами; Degree Centrality — число связей узла; Katz Centrality — влияние с учётом длины путей; PageRank — вероятность попадания в узел при случайном блуждании; Eigenvector Centrality — важность узла определяется важностью его соседей. Каждый тип отвечает на разный вопрос о значимости узла в графе.

**13. Что такое Dask cuDF и когда его использовать вместо обычного cuDF?**

Dask cuDF — распределённый аналог cuDF, поддерживающий ленивые вычисления и работу с множеством файлов и GPU. Его стоит использовать, когда данные не помещаются в память одного GPU, когда нужно параллельно читать сотни файлов, или когда вычисления должны масштабироваться на кластер GPU. Для данных, помещающихся в один GPU, обычный cuDF проще и быстрее из-за отсутствия накладных расходов Dask.

**14. Как KNN используется для поиска ближайших соседей в cuML?**

В cuML создаётся объект `NearestNeighbors(n_neighbors=K)`, который обучается на данных через `fit()`. Затем метод `kneighbors()` принимает запросные точки и возвращает расстояния и индексы K ближайших соседей из обучающего набора. Это полезно для задач типа «найти ближайший узел дорожной сети к заданной точке» — например, для привязки координат больниц к дорожным узлам.

**15. Какие алгоритмы центральности выполняются быстрее всего на GPU через nx-cugraph?**

Наибольшее ускорение получают вычислительно-дорогие алгоритмы: Betweenness Centrality (сложность O(V·E)) и Eigenvector Centrality (итеративный алгоритм). Degree Centrality, напротив, имеет линейную сложность и ускоряется меньше, так как на CPU он и так быстр. PageRank и Katz Centrality также хорошо ускоряются на GPU благодаря параллелизации итеративных вычислений.
