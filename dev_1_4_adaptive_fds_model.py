""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть первая."""

""" ## Глава IV: Построение сплайновой модели с адаптированной длине волны сеткой """

"""
### Задача

Построение сплайновой модели с адаптивной по длине волны сеткой для точного моделирования распространения сейсмических волн.

В cae-fidesys можно сделать сплайновую сетку. Для этого нам нужны следующие команды jou:

`create vertex location <x> <y> <z>` - создает вершину по указанным координатам (назначяет её индекс по порядку: 1,2,3,...)

`create curve spline location vertex [<v1> <v2> <v3> ...]` - создает сплайновую кривую через указанный набор вершин. В скобках - список id вершин через пробел

`create surface skin curve <c1> <c2>` - создает сплайновую поверхность, ограниченную двумя указанными вершинами.
"""

""" Достаем нужные данные """

import numpy as np


# Загрузка данных профильных сечений геологических поверхностей

# Загружаем файл с профильными сечениями
model1_mesh_coords = np.load('data/dev_1_2_model1_mesh_coords.npz')

x_coordinates = model1_mesh_coords['x_coordinates']
z_coordinates = model1_mesh_coords['z_coordinates']
layers_sep=[0, 17, 30, 48, 90, 112, 121, 137, 165, 220, 222, 224, 226, 228, 230, 232, 240, 242]

import pandas as pd

# Загружаем данные из CSV файла
static_material = pd.read_csv('data/static_material.csv', sep='\t')

# Выводим информацию о загруженных данных
print("Форма данных:", static_material.shape)
print("\nПервые 5 строк:")
print(static_material.head())
print("\nИнформация о столбцах:")
print(static_material.info())
print("\nОписательная статистика:")
print(static_material.describe())

# Извлекаем массив глубин для layers_sep
layers_sep = np.array([0] + static_material['depth'].tolist())//10
print(f"\nМассив layers_sep: {layers_sep}")

""" берем каждую пятую координату и только по нужным слоям, что бы построить сплайны """

z_data = z_coordinates[::5, layers_sep].T
x_data = x_coordinates[::5]

vertex_coords = np.zeros((*z_data.shape,3))

vertex_coords[:,:,1] = z_data
vertex_coords[:,:,0] = x_data

vertex_coords.shape

""" Создадим скрипт: /home/antonov/Base/Research/MilenSem2D/model3_geom.jou """

with open('model3_geom.jou', 'w') as f:
    f.write(f'reset\n')

    for curve_id in range(1,vertex_coords.shape[0]+1):
        row = vertex_coords[curve_id-1]
        for v in row:
            f.write(f'create vertex location {v[0]} {v[1]} {v[2]}\n')
        v_indexs = f'vertex {vertex_coords.shape[1]*(curve_id-1)+1} to {vertex_coords.shape[1]*(curve_id)}'
        f.write(f'create curve spline location {v_indexs}\n')
        f.write(f'delete {v_indexs}\n')

    for surf_id in range(1,vertex_coords.shape[0]):
        f.write(f'create surface skin curve {surf_id} {surf_id+1}\n')
    
    for curve_id in range(1,vertex_coords.shape[0]+1):
        f.write(f'delete curve {curve_id}\n')

    f.write('merge curve all\n')
    f.write('compress curve all\n')

""" Скрипт генерирует сплайновую геометрию нужной конфигурации. """

""" ![model3_geom.png](img/model3_geom.png) """

""" ### Сетка """

""" Для грамотного неструктурированного мешинга нужно задать количество интервалов вдоль кривых. Зададим их в виде массива размера 18. Для этого извлечем данные о средних материалах в слое: """

""" Нам удоблее будет поделить на h_max длину слоя и получить число интервалов: """


layer_len = 11750

# Создаем массив с результатами деления 11750 на h_max, округленными вверх
intervals_array = np.ceil(layer_len / static_material['h_max']).astype(int)
print("Массив количества интервалов:", intervals_array.tolist())

""" Получен массив:  [235, 231, 210, 176, 146, 157, 133, 130, 125, 130, 133, 124, 118, 100, 104, 118, 107]. Это *нижняя* граница для допустимого числа интервалов на каждой кривой (начиная со второй). Мы не сможем задать её напрямую, поскольку большинство слоев слишком тонкие, что бы создать на них неструктурированную сетку. """

""" Вот примерные ширины слоев в м: """

np.diff(layers_sep*10)

"""
Желательно иметь хотя бы 6 размеров ячейки на слой (размер ячейки начинается от 50), что означает, что для неструктурированности подходит только слой 4 (420 м) и слой 9 (550) м. 

Можно сделать это автоматически, но для лучшего контроля запишем интервалы вручную. Кроме того, надо добавить одно число в конце.

[235, 231, 210, 176, 146, 157, 133, 130, 125, 130, 133, 124, 118, 100, 104, 118, 107] ->

[235, 235, 235, 235, **170**, 175, 175, 175, 175, **135**, 135, 135, 135, 135, 135, 135, 135, 135]

Почему 175 и 135? Число должно быть больше, чем все числа дальше, 

Кроме того, добавим желаемое число интервалов вдоль боковой проверхности. 
"""

""" Индексы кривых, к которым нужно приложить, вычисляются так: """

curve_ids = [1] + list(range(3,17*3+1,3))

""" Строим сетку методами map (стурктурированные слои) и pave (нестуктурированные слои) """

""" /home/antonov/Base/Research/MilenSem2D/model3_mesh.jou """

INT_1 = 235
INT_2 = 175
INT_3 = 135

intervals_x = [INT_1, INT_1, INT_1, INT_1, INT_2, INT_2, INT_2, INT_2, INT_2, INT_3, INT_3, INT_3, INT_3, INT_3, INT_3, INT_3, INT_3, INT_3]
intervals_z = [4, 3, 4, 6, 4, 2, 3, 5, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1]

with open('model3_mesh.jou', 'w') as f:
    f.write(f'delete mesh surface all propagate\n')
    f.write(f'delete mesh curve all propagate\n')

    for i, curve_id in enumerate(curve_ids):
        f.write(f'curve {curve_id} scheme equal interval {intervals_x[i]}\n')
    for i, curve_id in enumerate(range(4, 17*3+2, 3)):
        f.write(f'curve {curve_id} scheme equal interval {intervals_z[i]}\n')
    for i, curve_id in enumerate(range(2, 17*3+1, 3)):
        f.write(f'curve {curve_id} scheme equal interval {intervals_z[i]}\n')
    f.write(f'mesh curve all\n')

    for i, surf_id in enumerate(range(1, 18)):
        if intervals_x[i] == intervals_x[i+1]:
            f.write(f'surface {surf_id} scheme map\n')
        else:
            f.write(f'surface {surf_id} scheme pave\n')

    f.write(f'mesh surface all\n')

""" ### Задаем тестовый материал """

""" Эта сетка слишком грубая, что бы задать и привязать к файлу расчета тот сильно дискретезированный материал, который мы хотим. Но мы можем задать для тестового расчета статический материал, полученный усреднением по слою. **Этот материал - исключительно для пробного расчета и в основной исследовательской программе учитываться не будет.** """

static_material

""" Зададим тестовый порядок спектрального элемента """

SEM_DEG = 3


with open('data/dev_1_4_model3_material.jou', 'w') as f:
    f.write(f"remove material all\n")
    f.write(f"delete block all\n")
    for i in range(1, 18):
        f.write(f"create material {i}\n")
        f.write(f"modify material {i} name 'test_mat_{i}'\n")
        f.write(f"modify material {i} set property 'VP' value {static_material.iloc[i-1]['vp'].replace(',', '.')}\n")
        f.write(f"modify material {i} set property 'VS' value {static_material.iloc[i-1]['vs'].replace(',', '.')}\n")
        f.write(f"modify material {i} set property 'DENSITY' value {static_material.iloc[i-1]['phob']}\n")
        f.write(f"block {i} add surface {i}\n")
        f.write(f"block {i} material {i} cs 1 category plane order {SEM_DEG} \n")

"""
Давайте зададим тестовый набор ГУ и проведем расчет динамики. 
Для теста снизим порядок спектрального элемента (до 3) и частоту (до 10)
"""

TEST_FR = 10
TEST_AMP = 1e5

with open('data/dev_1_4_model3_bc.jou', 'w') as f:
    f.write(f"create force  on node {INT_1//2} force value 1 direction 0 1 0\n")
    f.write(f"bcdep force 1 value 'ricker({TEST_AMP}, {TEST_FR}, 0, time)'\n")
    f.write(f"create absorption on curve 51'\n")
    f.write(f"create absorption on curve {' '.join(map(str, range(4, 17*3+2, 3)))}\n")
    f.write(f"create absorption on curve {' '.join(map(str, range(2, 17*3+1, 3)))}\n")
    f.write(f"create receiver on curve 1  velocity 1 1 1\n")

with open('data/dev_1_4_model3_calc.jou', 'w') as f:
    f.write(f"analysis type dynamic elasticity dim2 planestrain preload off\n")
    f.write(f"dynamic method full_solution scheme explicit maxsteps 30000 maxtime 3\n")
    f.write(f"dynamic results everytime 0.1\n")
    f.write(f"create absorption on curve {' '.join(map(str, range(4, 17*3+2, 3)))}\n")
    f.write(f"output nodalforce off energy off record3d off material off without_smoothing off fullperiodic off\n")


""" Результат расчета дает нам достаточно грубую сейсмограмму, которую мы не будем здесь анализировать (можно посмотреть её в  model3/model3_Vx.sgy и model3/model3_Vy.sgy) """

"""
### Выводы

В данной главе была построена сплайновая модель с адаптивной по длине волны сеткой для точного моделирования распространения сейсмических волн.

Были выполнены следующие задачи:
1. Создание сплайновой геометрии с использованием команд Fidesys
2. Адаптация сетки по длине волны для различных слоев
3. Создание тестовых материалов для проверки работоспособности модели
4. Генерация файлов управления расчетом (.jou) для Fidesys

Результаты сохранены в файлах:
- data/dev_1_4_model3_material.jou - файл материалов
- data/dev_1_4_model3_bc.jou - файл граничных условий
- data/dev_1_4_model3_calc.jou - файл параметров расчета
"""
