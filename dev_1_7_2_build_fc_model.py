""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть первая. """

""" ## Глава VII (часть 2): Построение FC модели с декартовой сеткой материала """

"""
Этот скрипт загружает предварительно подготовленные декартовые сетки материала 
и создает FC модель. Используется для быстрой итерации без перезапуска 
интерполяции из первой части.
"""

import numpy as np
from pathlib import Path
from fc_model import FCModel


""" #### 1. Загрузка предварительно рассчитанных данных #### """

# Выбранная дискретизация
SELECTED_DX = 5
SELECTED_DY = 5

print(f"{'='*60}")
print(f"Построение FC модели с дискретизацией {SELECTED_DX}×{SELECTED_DY} м")
print('='*60)

# Загружаем геометрию слоев
layer_data_path = Path('data/dev_1_5_2_layer_boundaries_quadratic.npz')
layer_data = np.load(layer_data_path)
layer_boundaries_array = layer_data['layer_boundaries_array']  # shape (75, 1176)
formations = layer_data['formations']
distances = layer_data['distances']

# Добавляем поверхность (глубина 0) в начало
all_boundaries = np.vstack([np.zeros((1, distances.shape[0])), layer_boundaries_array])

print(f"\nЗагружена геометрия: {len(formations)} слоев")

# Загружаем общую сетку
all_grids_path = Path('data/dev_1_7_all_grids.npz')
all_grids_data = np.load(all_grids_path, allow_pickle=True)

selected_grid_key = f'{SELECTED_DX}x{SELECTED_DY}'

if selected_grid_key not in all_grids_data:
    print(f"\nОШИБКА: Сетка {selected_grid_key} не найдена!")
    print("Доступные сетки:")
    for key in all_grids_data.files:
        print(f"  - {key}")
    exit(1)

selected_grid = all_grids_data[selected_grid_key].item()
print(f"Загружена общая сетка {selected_grid_key}")

# Загружаем декартовые сетки слоев
# Нужно пересоздать layer_grids из первой части
# Для этого загрузим исходный материал и пересоздадим интерполяторы

material_data_path = Path('data/dev_1_6_model_material_new.npz')
material_data = np.load(material_data_path, allow_pickle=True)
eulerian_coords = material_data['coords']
eulerian_props = material_data['properties']

# Вычислим индексы уникальных пар координат
first_row_coords = eulerian_coords[0]
_, unique_indices_first_row = np.unique(first_row_coords, axis=0, return_index=True)
eulerian_coords = eulerian_coords[:, unique_indices_first_row]
eulerian_props = eulerian_props[:, unique_indices_first_row]

print(f"Загружен исходный материал: {eulerian_coords.shape}")


""" #### 2. Пересоздание декартовых сеток слоев #### """

from scipy.interpolate import CloughTocher2DInterpolator

def build_layer_interpolators(eulerian_coords, eulerian_props, all_boundaries, formations):
    """
    Строит интерполяторы для каждого слоя на основе исходного материала.
    """
    n_layers = len(formations)
    layer_interpolators = []
    
    print("\nПостроение интерполяторов для слоев:")
    
    for layer_idx in range(n_layers):
        print(f"  Слой {layer_idx+1}/{n_layers}: {formations[layer_idx]}", end='')
        
        # Границы слоя
        y_top = all_boundaries[layer_idx, :]
        y_bottom = all_boundaries[layer_idx + 1, :]
        
        y_min_layer = np.min(y_top) - 10
        y_max_layer = np.max(y_bottom) + 10
        
        # Собираем точки материала, которые попадают в этот слой
        layer_points = []
        layer_props = []
        
        n_x, n_y = eulerian_coords.shape[:2]
        
        for i in range(n_x):
            for j in range(n_y):
                x = eulerian_coords[i, j, 0]
                y = eulerian_coords[i, j, 1]
                
                # Проверяем, попадает ли точка в диапазон глубин слоя
                if y_min_layer <= y <= y_max_layer:
                    layer_points.append([x, y])
                    layer_props.append(eulerian_props[i, j, :])
        
        if len(layer_points) == 0:
            print(f" - НЕТ ТОЧЕК!")
            layer_interpolators.append(None)
            continue
        
        layer_points = np.array(layer_points)
        layer_props = np.array(layer_props)
        
        # Создаем интерполяторы для каждого свойства
        interp_E = CloughTocher2DInterpolator(layer_points, layer_props[:, 0])
        interp_nu = CloughTocher2DInterpolator(layer_points, layer_props[:, 1])
        interp_rho = CloughTocher2DInterpolator(layer_points, layer_props[:, 2])
        
        layer_interpolators.append({
            'formation': formations[layer_idx],
            'y_min': y_min_layer,
            'y_max': y_max_layer,
            'n_points': len(layer_points),
            'interp_E': interp_E,
            'interp_nu': interp_nu,
            'interp_rho': interp_rho
        })
        
        print(f" - {len(layer_points)} точек")
    
    return layer_interpolators


def build_layer_cartesian_grids(dx, dy, x_min, x_max, all_boundaries, layer_interpolators):
    """
    Строит декартовые сетки для каждого слоя и интерполирует свойства.
    """
    layer_grids = []
    
    print(f"\nПостроение декартовых сеток для слоев (дискретизация {dx}×{dy} м):")
    
    for layer_idx, interp_data in enumerate(layer_interpolators):
        if interp_data is None:
            print(f"  Слой {layer_idx+1}: пропущен (нет данных)")
            layer_grids.append(None)
            continue
        
        # Определяем границы слоя с небольшим запасом
        y_min_layer = interp_data['y_min'] - dy
        y_max_layer = interp_data['y_max'] + dy
        
        # Создаем узлы сетки, выровненные с глобальной сеткой
        x_start = np.ceil(x_min / dx) * dx
        y_start = np.ceil(y_min_layer / dy) * dy
        
        x_nodes = np.arange(x_start, x_max + dx, dx)
        y_nodes = np.arange(y_start, y_max_layer + dy, dy)
        
        # Создаем сетку точек для интерполяции
        X_grid, Y_grid = np.meshgrid(x_nodes, y_nodes, indexing='ij')
        points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Пакетная интерполяция всех свойств
        E_values = interp_data['interp_E'](points).reshape(X_grid.shape)
        nu_values = interp_data['interp_nu'](points).reshape(X_grid.shape)
        rho_values = interp_data['interp_rho'](points).reshape(X_grid.shape)
        
        # Проверка на NaN
        n_nan_E = np.sum(np.isnan(E_values))
        n_nan_nu = np.sum(np.isnan(nu_values))
        n_nan_rho = np.sum(np.isnan(rho_values))
        
        if n_nan_E > 0 or n_nan_nu > 0 or n_nan_rho > 0:
            print(f"  Слой {layer_idx+1}: ПРЕДУПРЕЖДЕНИЕ - найдены NaN значения!")
            print(f"    E: {n_nan_E} NaN, nu: {n_nan_nu} NaN, rho: {n_nan_rho} NaN")
            
            # Заполняем NaN средними значениями
            if n_nan_E > 0:
                E_values = np.nan_to_num(E_values, nan=np.nanmean(E_values))
            if n_nan_nu > 0:
                nu_values = np.nan_to_num(nu_values, nan=np.nanmean(nu_values))
            if n_nan_rho > 0:
                rho_values = np.nan_to_num(rho_values, nan=np.nanmean(rho_values))
        
        layer_grids.append({
            'formation': interp_data['formation'],
            'x_nodes': x_nodes,
            'y_nodes': y_nodes,
            'E': E_values,
            'nu': nu_values,
            'rho': rho_values
        })
        
        print(f"  Слой {layer_idx+1}: {len(x_nodes)}×{len(y_nodes)} = {len(x_nodes)*len(y_nodes)} узлов")
    
    return layer_grids


# Строим интерполяторы и декартовые сетки слоев
x_min = distances[0]
x_max = distances[-1]

layer_interpolators = build_layer_interpolators(
    eulerian_coords, eulerian_props, all_boundaries, formations
)

selected_layer_grids = build_layer_cartesian_grids(
    SELECTED_DX, SELECTED_DY, x_min, x_max, all_boundaries, layer_interpolators
)


""" #### 3. Подготовка данных для FC модели #### """

def prepare_fc_material_data(layer_grids, all_boundaries, formations):
    """
    Подготавливает данные материала для каждого слоя на основе декартовых сеток слоев.
    """
    n_layers = len(formations)
    layer_materials = []
    
    print("\nПодготовка данных для FC модели:")
    
    for layer_idx in range(n_layers):
        print(f"  Обработка слоя {layer_idx+1}/{n_layers}: {formations[layer_idx]}")
        
        layer_grid = layer_grids[layer_idx]
        
        if layer_grid is None:
            print(f"    Предупреждение: Слой {formations[layer_idx]} не имеет данных!")
            layer_materials.append({
                'formation': formations[layer_idx],
                'coords': np.array([[0, 0]]),
                'properties': np.array([[1e9, 0.25, 2000]])  # Дефолтные значения вместо 0
            })
            continue
        
        # Извлекаем данные из декартовой сетки слоя
        x_nodes = layer_grid['x_nodes']
        y_nodes = layer_grid['y_nodes']
        E_grid = layer_grid['E']
        nu_grid = layer_grid['nu']
        rho_grid = layer_grid['rho']
        
        # Создаем списки координат и свойств для всех узлов сетки слоя
        layer_coords = []
        layer_props_list = []
        
        for i, x in enumerate(x_nodes):
            for j, y in enumerate(y_nodes):
                layer_coords.append([x, y])
                layer_props_list.append([E_grid[i, j], nu_grid[i, j], rho_grid[i, j]])
        
        layer_coords = np.array(layer_coords)
        layer_props_array = np.array(layer_props_list)
        
        # Проверка на NaN
        n_nan = np.sum(np.isnan(layer_props_array))
        if n_nan > 0:
            print(f"    ОШИБКА: найдено {n_nan} NaN значений в свойствах!")
            print(f"    E: min={np.nanmin(layer_props_array[:, 0]):.2e}, max={np.nanmax(layer_props_array[:, 0]):.2e}")
            print(f"    nu: min={np.nanmin(layer_props_array[:, 1]):.3f}, max={np.nanmax(layer_props_array[:, 1]):.3f}")
            print(f"    rho: min={np.nanmin(layer_props_array[:, 2]):.1f}, max={np.nanmax(layer_props_array[:, 2]):.1f}")
        else:
            print(f"    Узлов материала: {len(layer_coords)}")
            print(f"    E: [{np.min(layer_props_array[:, 0]):.2e}, {np.max(layer_props_array[:, 0]):.2e}]")
            print(f"    nu: [{np.min(layer_props_array[:, 1]):.3f}, {np.max(layer_props_array[:, 1]):.3f}]")
            print(f"    rho: [{np.min(layer_props_array[:, 2]):.1f}, {np.max(layer_props_array[:, 2]):.1f}]")
        
        layer_materials.append({
            'formation': formations[layer_idx],
            'coords': layer_coords,
            'properties': layer_props_array
        })
    
    return layer_materials


# Подготавливаем данные
layer_materials = prepare_fc_material_data(selected_layer_grids, all_boundaries, formations)


""" #### 4. Загрузка и модификация FC модели #### """

# Загружаем базовую модель
fc_model_path = 'data/dev_1_6_model_material_stub.fc'
print(f"\n{'='*60}")
print(f"Загрузка FC модели: {fc_model_path}")
fc_model = FCModel(fc_model_path)

print(f"Материалов в модели: {len(fc_model.materials)}")

# Проверяем соответствие количества материалов
if len(fc_model.materials) != len(layer_materials):
    print(f"ОШИБКА: Количество материалов не совпадает!")
    print(f"  В FC модели: {len(fc_model.materials)}")
    print(f"  Подготовлено: {len(layer_materials)}")
else:
    print("Количество материалов совпадает, начинаем обновление...")


""" #### 5. Обновление материалов в FC модели #### """

print(f"\n{'='*60}")
print("Обновление материалов в FC модели")
print('='*60)

for i, layer_mat in enumerate(layer_materials):
    mat_idx = i + 1  # Индексация материалов с 1
    
    print(f"\nОбновление материала {mat_idx}: {layer_mat['formation']}")
    
    coords = layer_mat['coords']
    props = layer_mat['properties']
    
    # Обновляем модуль Юнга (MODULUS)
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.table[0].value.data = coords[:, 0]
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.table[1].value.data = coords[:, 1]
    fc_model.materials[mat_idx].properties['elasticity'][0][0].data.value.data = props[:, 0]
    
    # Обновляем коэффициент Пуассона (POISSON)
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.table[0].value.data = coords[:, 0]
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.table[1].value.data = coords[:, 1]
    fc_model.materials[mat_idx].properties['elasticity'][0][1].data.value.data = props[:, 1]
    
    # Обновляем плотность (DENSITY)
    fc_model.materials[mat_idx].properties['common'][0][0].data.table[0].value.data = coords[:, 0]
    fc_model.materials[mat_idx].properties['common'][0][0].data.table[1].value.data = coords[:, 1]
    fc_model.materials[mat_idx].properties['common'][0][0].data.value.data = props[:, 2]
    
    # Создаем массив демпфирования
    mass_matrix_damping = np.zeros(len(coords))
    
    for j in range(len(coords)):
        x = coords[j, 0]
        y = coords[j, 1]
        
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
    fc_model.materials[mat_idx].properties['common'][0][1].data.table[0].value.data = coords[:, 0]
    fc_model.materials[mat_idx].properties['common'][0][1].data.table[1].value.data = coords[:, 1]
    fc_model.materials[mat_idx].properties['common'][0][1].data.value.data = mass_matrix_damping
    
    print(f"  Обновлено {len(coords)} узлов материала")


""" #### 6. Сохранение результатов #### """

# Сохраняем обновленную модель
output_fc_path = 'data/dev_1_7_model_material_cartes.fc'
fc_model.save(output_fc_path)
print(f"\n{'='*60}")
print(f"FC модель сохранена: {output_fc_path}")
print('='*60)

# Сохраняем данные материалов
output_data_path = f'data/dev_1_7_layer_materials_{SELECTED_DX}x{SELECTED_DY}.npz'
np.savez_compressed(output_data_path, 
                    layer_materials=layer_materials,
                    formations=formations)
print(f"Данные материалов сохранены: {output_data_path}")

print("\nГотово!")

