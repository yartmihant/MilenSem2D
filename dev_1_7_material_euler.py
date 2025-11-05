""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть первая. """

""" ## Глава VII: Эйлерова декартова сетка материала """

"""
### Научная задача

В предыдущей главе мы создали материал, следующий криволинейной геометрии слоев. Однако такой подход имеет ограничения при передаче данных коллегам-геофизикам, использующим метод конечных разностей.

В этой главе мы реализуем альтернативный подход: поверх криволинейной слоистой геометрии натягиваем абсолютно ровную декартову сетку материала. 

1. Такой подход редко используется в МСЭ, хотя и возможен
2. Позволит отладить методы интерполяции материалов
3. Обеспечит совместимость с методом конечных разностей

### Задача 1: Подготовка сетки материала

Создаем декартову сетку дискретизации 5x5. Это соответствует размеру ячейки для метода конечных разностей, достаточному, что бы улавливать медленные (~ 500 м/с) с частотой 30 Hz. Данная сетка будет передана геофизикам. Для нужд метода конечных элементов в fc будет записана угрубленная версия той же сетки с дискретностью 25x5, поскольку упругие свойства материала слабо меняются по X, и такая запись позволит существенно уменьшить размеры модели.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fc_model import FCModel
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

# Глобальный флаг для управления отображением графиков
SHOW_PLOTS = True  # True - показывать графики, False - только сохранять

""" #### 1. Загрузка данных #### """

# Загрузка геометрии слоев
layer_data_path = Path('data/dev_1_5_2_layer_boundaries_quadratic.npz')
layer_data = np.load(layer_data_path)

layer_boundaries_array = layer_data['layer_boundaries_array']  # shape (75, 1176)
formations = layer_data['formations']
distances = layer_data['distances']  # shape (1176,)
well1_depths = layer_data['well1_depths']
well2_depths = layer_data['well2_depths']

print(f"Загружена геометрия слоев: {layer_boundaries_array.shape}, {distances.shape}")
print(f"Количество формаций: {len(formations)}")

""" Добавляем поверхность (глубина 0) в начало """

all_boundaries = np.vstack([np.zeros((1, distances.shape[0])), layer_boundaries_array])

# Границы модели
x_min, x_max = distances[0], distances[-1]
y_min = 0
y_max = np.max(layer_boundaries_array)

print(f"\nГраницы модели:")
print(f"X: от {x_min:.1f} до {x_max:.1f} м")
print(f"Y: от {y_min:.1f} до {y_max:.1f} м")

# Загрузка материала
material_data_path = Path('data/dev_1_6_model_material_new.npz')
material_data = np.load(material_data_path, allow_pickle=True)
lagrange_coords = material_data['coords']  # координаты узлов материала
lagrange_props = material_data['properties']  # свойства [E, nu, rho]

material_data['properties']

def find_layer_at_point(x, y, distances, all_boundaries):
    """
    Определяет номер слоя (0-74) для точки (x, y).
    
    Args:
        x: горизонтальная координата
        y: глубина
        distances: массив горизонтальных координат профиля
        all_boundaries: массив границ слоев (76 границ для 75 слоев)
        
    Returns:
        layer_index: номер слоя (0-74) или -1 если вне модели
    """
    # Находим ближайший индекс по x
    x_idx = np.argmin(np.abs(distances - x))
    
    # Проверяем, в каком слое находится точка
    for layer_idx in range(len(all_boundaries) - 1):
        y_top = all_boundaries[layer_idx, x_idx]
        y_bottom = all_boundaries[layer_idx + 1, x_idx]
        
        if y_top <= y <= y_bottom:
            return layer_idx
    
    return -1  # Вне модели

# Задаем дискретизацию
dx, dy = 5, 5

# Строим декартову сетку
layer_grids = []

x_cells = np.arange(x_min+dx/2, x_max, dx)
y_cells = np.arange(y_min+dy/2, y_max, dy)

# Создаем сетку точек для интерполяции
# Создаем общий массив координат по X, Y для всех узлов сетки
X_grid, Y_grid = np.meshgrid(x_cells, y_cells, indexing='ij')
coords_grid = np.stack([X_grid, Y_grid], axis=-1)  # shape (N, 2), с координатами (x, y) каждой точки сетки

layer_indexes_grid = np.full(coords_grid.shape[:2], -1)

for i in range(coords_grid.shape[0]):
    for j in range(coords_grid.shape[1]):
        x = coords_grid[i,j][0]
        y = coords_grid[i,j][1]

        # Находим ближайший индекс по x
        x_idx = np.argmin(np.abs(distances - x))

        boundaries = all_boundaries[:-1, x_idx]

        layer_indexes_grid[i,j] = np.argmin(np.abs(boundaries - y))
        layer_indexes_grid[i, j] = np.where(boundaries < y)[0].max()

first_row_coords = lagrange_coords[0]  # shape (386, 2)
_, unic_index = np.unique(first_row_coords, axis=0, return_index=True)

def split_by_missing(input_data):
    diff = np.diff(input_data)
    break_pos = np.where(diff > 1)[0]  # индексы в input_data слева от разрыва
    missing = input_data[break_pos] + 1

    starts = np.r_[0, break_pos + 1]
    ends   = np.r_[break_pos + 1, len(input_data)]
    layers = [input_data[s:e] for s, e in zip(starts, ends)]

    output = []
    for i, layer in enumerate(layers):
        extended = []

        if i > 0:
            prev_last = layers[i-1][-1:-1]
            # prev_last = layers[i-1][-2]
            missing_before = missing[i-1]  # один пропуск на каждый разрыв

            extended.extend(prev_last)
            extended.append(missing_before)

        extended.extend(layer.tolist())

        if i < len(layers) - 1:
            next_first = layers[i+1][0:0]
            # next2_first = layers[i+1][1]

            extended.extend(next_first)

        output.append(extended)
    return output

layer_borders = split_by_missing(unic_index)

# # Проверка
# for row in layer_borders:
#     print(row)

""" #### 2. Построение интерполяторов для каждого слоя #### """

""" Строитм интерполяторы для каждого слоя на основе исходного материала. """

n_layers = len(formations)
layer_interpolators = []

print("\nПостроение интерполяторов для слоев:")

for layer_idx in range(n_layers):
    print(f"  Слой {layer_idx+1}/{n_layers}: {formations[layer_idx]}", end='')
    
    layer_points = lagrange_coords[:, layer_borders[layer_idx]].reshape(-1, 2)  # координаты узлов материала
    layer_props = lagrange_props[:, layer_borders[layer_idx]].reshape(-1, 3)


    interp_E = CloughTocher2DInterpolator(layer_points, layer_props[:, 0], fill_value=np.mean(layer_props[:, 0]))
    interp_nu = CloughTocher2DInterpolator(layer_points, layer_props[:, 1], fill_value=np.mean(layer_props[:, 1]))
    interp_rho = CloughTocher2DInterpolator(layer_points, layer_props[:, 2], fill_value=np.mean(layer_props[:, 2]))

    
    layer_interpolators.append({
        'formation': formations[layer_idx],
        'n_points': len(layer_points),
        'interp_E': interp_E,
        'interp_nu': interp_nu,
        'interp_rho': interp_rho
    })
    
    print(f" - {len(layer_points)} точек")

""" #### 3. Построение декартовой сетки #### """

material_grid = np.zeros((*layer_indexes_grid.shape,3))

print(f"\nПостроение декартовых сеток для слоев (дискретизация {dx}×{dy} м):")

for layer_idx, interp_data in enumerate(layer_interpolators):
    formation = interp_data['formation']
    n_points = interp_data['n_points']
    interp_E = interp_data['interp_E']
    interp_nu = interp_data['interp_nu']
    interp_rho = interp_data['interp_rho']

    coords_in_layer = np.argwhere(layer_indexes_grid == layer_idx)

    layer_coords = coords_grid[coords_in_layer[:, 0], coords_in_layer[:, 1]]
    layer_coords_grid = layer_coords.reshape(-1, 2)

    # Пакетная интерполяция всех свойств
    E_values = interp_E(layer_coords_grid)
    nu_values = interp_nu(layer_coords_grid)
    rho_values = interp_rho(layer_coords_grid)
    
    material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 0] = E_values
    material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 1] = nu_values
    material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 2] = rho_values

    print(f'Слой {layer_idx+1} реинтреполирован', np.min(rho_values), np.max(rho_values),)

""" #### 4. Визуализация сетки #### """

""" Визуализация поля модуля Юнга (E_values) по всей расчетной декартовой сетке """

import matplotlib.pyplot as plt


def show_plot(title, field, cmap='viridis'):

    fig, ax = plt.subplots(figsize=(16,8))
    
    X, Y = np.meshgrid(coords_grid[:,0,0], coords_grid[0,:,1], indexing='ij')

    c = ax.pcolormesh(X, Y, field, cmap=cmap, shading='auto')
    plt.colorbar(c, ax=ax, label=title)

    ax.set_xlabel('x (м)')
    ax.set_ylabel('y (м)')
    ax.set_title(title)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)   # Инверсия оси Y (глубина вниз)
    plt.tight_layout()

    output_path = f'img/dev_1_7_{title}_layers.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Визуализация слоев: {output_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

E_grid = material_grid[:, :, 0]  # Модуль Юнга для всех узлов
show_plot('E_grid', E_grid, 'rainbow')

show_plot('Layers', layer_indexes_grid, 'prism')

show_plot('nu_values', material_grid[:, :, 1], 'rainbow')

show_plot('pho_values', material_grid[:, :, 2], 'rainbow')

E = material_grid[:, :, 0]
nu = material_grid[:, :, 1]
pho = material_grid[:, :, 2]
Vp = np.sqrt(E/pho * (1-nu)/(1+nu)/(1-2*nu)) 
Vs = np.sqrt(E/pho /2/(1+nu)) 

show_plot('Vp', Vp, 'rainbow')

show_plot('Vs', Vs, 'rainbow')

""" Сохраняем массивы material_grid, coords_grid и layer_indexes_grid на диск """

np.savez_compressed('data/dev_1_7_material_grids.npz',
                    material_grid=material_grid,
                    coords_grid=coords_grid,
                    layer_indexes_grid=layer_indexes_grid)
print("Массивы material_grid, coords_grid, layer_indexes_grid сохранены в data/dev_1_7_material_grids.npz")

""" #### 5. Построение сетки FDS модели, обновленное #### """

"""
Возьмем минимум продольной скорости в слое за меру того, сколь сильно мы можем измельчить слой. Предположим, мы хотим, что бы линейный размер элемента составлял 1/4 длины продольной волны. Это удобно потому, что минимальная толщина слоя - 10 , а в верхней части у нас как раз есть слои тощиной 10, размер элемента которых логично подогнать к 10. По таблице Ухнова для 2D для 5 порядка сетки достаточно 2 элемента (у нас 4) для достижения 1%, так что у нас есть хороший запаc. 

Примечание: таблица Уханова составлялась для пульса Берлаге а не более сложного Рикера, для однородного случая. Поэтому она является оценкой точности *снизу*, минимумом, который нельзя нарушать. Кроме того, иног
"""

el_in_wave = 4
fr = 30
max_len_per_depth = np.min(Vp, axis=(0))/30/el_in_wave
y_depth = np.arange(0, 550*5, 5)

plt.plot(y_depth, max_len_per_depth)
plt.xlabel('y_depth')
plt.ylabel('max_len_per_depth')
plt.title('Зависимость допустимого максимального размера элемента от глубины')
plt.grid(True)
plt.show()

layer_depth = np.mean(layer_boundaries_array, axis=(1))

max_param_per_layer = np.empty(layer_depth.shape[0])
for i, depth in enumerate(layer_depth):
    max_index = int(depth / 5)
    max_param_per_layer[i] = max_len_per_depth[:max_index+1].max()

""" Делаем первый слой сторого по 10 для удобства расстановки ресиверов """

max_param_per_layer[0] = 10

plt.plot(layer_depth, max_param_per_layer)
plt.xlabel('y_depth')
plt.ylabel('max_len_per_depth')
plt.title('Зависимость допустимого максимального размера элемента от слоя')
plt.grid(True)
plt.show()

"""
Главное условие построения сетки схемой pave - четная сумма интервалов на границах. Мы обеспечим это равной четностью в горизонтальных и вертикальных группах. 

Во всех горизонтальных группах число элементов должно быть нечетным.

Горизонтальная длина первой и третей группы - 4250, второй - 3250
"""

""" Задаем индексы кривых (так они генерируются в fidesys) """

h1_line_index = [1, 3] + list(range(12,524,7))
h2_line_index = [5, 7] + list(range(15,527,7))
h3_line_index = [8, 10] + list(range(17,529,7))

v1_line_index = [2] + list(range(13,525,7))
v2_line_index = [4] + list(range(11,523,7))
v3_line_index = [6] + list(range(14,526,7))
v4_line_index = [9] + list(range(16,528,7))

len(h1_line_index)


""" Устанавливаем интервалы по горизонтали (с удобным нулевым слоем) """

h1_interval = np.concatenate([[425], np.round(425/max_param_per_layer) * 10 + 5]).astype(int)
h2_interval = np.concatenate([[325], np.round(325/max_param_per_layer) * 10 + 5]).astype(int)
h3_interval = np.concatenate([[425], np.round(425/max_param_per_layer) * 10 + 5]).astype(int)

well1_depths

well1_layer_height = np.diff(np.concatenate([[0], well1_depths[:-1], [2650]])).astype(int)
well2_layer_height = np.diff(np.concatenate([[0], well2_depths[:-1], [2650]])).astype(int)

well1_layer_height

well2_layer_height

well1_layer_interval = np.ceil(well1_layer_height/max_param_per_layer).astype(int)
well2_layer_interval = np.ceil(well2_layer_height/max_param_per_layer).astype(int)

for i in range(len(well1_layer_interval)):
    parity1 = well1_layer_interval[i] % 2
    parity2 = well2_layer_interval[i] % 2
    if parity1 != parity2:
        if well1_layer_interval[i] > well2_layer_interval[i]:
            well1_layer_interval[i] += 1
        else:
            well2_layer_interval[i] += 1

""" Это дает представление о неструктурированности слоев: """

np.concatenate([
    [well1_layer_interval],
    [well2_layer_interval]
], axis=0).transpose()

with open('data/dev_1_7_model_mesh_curve.jou', 'w') as f:
    f.write(f'delete mesh surface all propagate\n')
    f.write(f'delete mesh curve all propagate\n')

    # первая горизональная группа  
    for i, curve_id in enumerate(h1_line_index):
        f.write(f'curve {curve_id} scheme equal interval {h1_interval[i]}\n')

    # вторая горизональная группа  
    for i, curve_id in enumerate(h2_line_index):
        f.write(f'curve {curve_id} scheme equal interval {h2_interval[i]}\n')

    # третья горизональная группа  
    for i, curve_id in enumerate(h3_line_index):
        f.write(f'curve {curve_id} scheme equal interval {h3_interval[i]}\n')

    # первая вертикальная группа
    for i, curve_id in enumerate(v1_line_index):
        f.write(f'curve {curve_id} scheme equal interval {well1_layer_interval[i]}\n')

    # вторая вертикальная группа
    for i, curve_id in enumerate(v2_line_index):
        f.write(f'curve {curve_id} scheme equal interval {well1_layer_interval[i]}\n')

    # третья вертикальная группа
    for i, curve_id in enumerate(v3_line_index):
        f.write(f'curve {curve_id} scheme equal interval {well2_layer_interval[i]}\n')

    # четвертая вертикальная группа
    for i, curve_id in enumerate(v4_line_index):
        f.write(f'curve {curve_id} scheme equal interval {well2_layer_interval[i]}\n')

    f.write(f'mesh curve all\n')

with open('data/dev_1_7_model_mesh_surface.jou', 'w') as f:
    f.write(f'delete mesh surface all\n')

    for i, surf_id in enumerate(range(1,224,3)):
        if h1_interval[i] == h1_interval[i+1]:
            f.write(f'surface {surf_id} scheme map\n')
        else:
            f.write(f'surface {surf_id} scheme pave\n')
        f.write(f'mesh surface {surf_id}\n')

    for i, surf_id in enumerate(range(2,225,3)):
        if h1_interval[i] == h1_interval[i+1] and well1_layer_height[i] == well2_layer_interval[i]:
            f.write(f'surface {surf_id} scheme map\n')
        else:
            f.write(f'surface {surf_id} scheme pave\n')
        f.write(f'mesh surface {surf_id}\n')

    for i, surf_id in enumerate(range(3,226,3)):
        if h1_interval[i] == h1_interval[i+1]:
            f.write(f'surface {surf_id} scheme map\n')
        else:
            f.write(f'surface {surf_id} scheme pave\n')
        f.write(f'mesh surface {surf_id}\n')

""" Затем мы генерируем модель fc из исходного fds с заглушками материалов, что бы записать файл автоматически. """

# Загружаем базовую модель
fc_model_path = 'data/dev_1_7_model_material_stub.fc'
print(f"\n{'='*60}")
print(f"Загрузка FC модели: {fc_model_path}")
fc_model = FCModel(fc_model_path)

for layer_idx, formation in enumerate(formations):
    mat_idx = layer_idx + 1  # Индексация материалов с 1
    
    coords_in_layer = np.argwhere(layer_indexes_grid == layer_idx)

    coords_count = len(coords_in_layer)

    print(f"\nОбновление материала {formation} {mat_idx}")
    
    x_coords = coords_grid[coords_in_layer[:,0], coords_in_layer[:,1], 0]
    y_coords = coords_grid[coords_in_layer[:,0], coords_in_layer[:,1], 1]

    E_data = material_grid[coords_in_layer[:,0], coords_in_layer[:,1], 0]
    nu_data = material_grid[coords_in_layer[:,0], coords_in_layer[:,1], 1]
    pho_data = material_grid[coords_in_layer[:,0], coords_in_layer[:,1], 2]

    # Обновляем модуль Юнга (MODULUS)
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.table[0].value.data = x_coords
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.table[1].value.data = y_coords
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.value.data = E_data
    
    # Обновляем коэффициент Пуассона (POISSON)
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.table[0].value.data = x_coords
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.table[1].value.data = y_coords
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.value.data = nu_data
    
    # Обновляем плотность (DENSITY)
    fc_model.materials[mat_idx].properties['common'][0][0].data.table[0].value.data = x_coords
    fc_model.materials[mat_idx].properties['common'][0][0].data.table[1].value.data = y_coords
    fc_model.materials[mat_idx].properties['common'][0][0].data.value.data = pho_data
    
    # Создаем массив демпфирования
    mass_matrix_damping = np.zeros(coords_count)
    
    for j in range(coords_count):
        x = x_coords[j]
        y = y_coords[j]
        
        # Применяем демпфирование на краях модели
        if x < 250 or x > 11500 or y > 2500:
            damping_x = 0
            damping_y = 0
            
            if x < 250:
                damping_x = 100 * ((250 - x) / 250) ** 2
            elif x > 11500:
                damping_x = 100 * ((x - 11500) / 250) ** 2
            
            if y > 2500:
                damping_y = 100 * ((y - 2500) / 250) ** 2
            
            mass_matrix_damping[j] = max(damping_x, damping_y)
    
    # Обновляем демпфирование (MASS_MATRIX_DAMPING)
    fc_model.materials[mat_idx].properties['common'][0][1].data.table[0].value.data = x_coords
    fc_model.materials[mat_idx].properties['common'][0][1].data.table[1].value.data = y_coords
    fc_model.materials[mat_idx].properties['common'][0][1].data.value.data = mass_matrix_damping
    
    print(f"  Обновлено {coords_count} узлов материала")

""" #### 6. Сохранение результатов #### """

# Сохраняем обновленную модель
output_fc_path = 'data/dev_1_7_model_material_cartes.fc'
fc_model.save(output_fc_path)
print(f"\n{'='*60}")
print(f"FC модель сохранена: {output_fc_path}")
print('='*60)
