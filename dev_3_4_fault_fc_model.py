""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть третья. """

""" ## Глава IV: FC-модель с реальным материалом """

"""
### Задача

Загрузить FC-модель (заглушку из Fidesys) и заполнить 75 материалов реальными упругими
свойствами из декартовой сетки (Глава III.2). Каждый материал задаётся на интерполяционных
декартовых точках (x, y) → (E, ν, ρ), принадлежащих соответствующему слою.

Также задаётся демпфирование на краях модели (квадратичная рампа).

Входные данные:
- `data/dev_3_3_model_stub.fc` — FC-модель с заглушками (75 материалов)
- `data/dev_3_2_fault_material.npz` — материал на декартовой сетке 5×5 м
- `data/dev_3_1_fault_layer_boundaries.npz` — границы слоёв с разломом

Выходные данные:
- `data/dev_3_4_model_calc.fc` — FC-модель для расчёта
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fc_model import FCModel
from fc_model.fc_data import FCData, FCDependencyColumn
from fc_model.fc_value import FCValue


""" ## Параметр угла разлома ## """

# Угол разлома определяет суффикс входных/выходных файлов (должен совпадать с dev_3_1)
fault_angle_param = 20.0  # градусы
angle_suffix = f'_a{int(fault_angle_param)}'
print(f"Суффикс файлов: {angle_suffix}")


""" ## Загрузка данных ## """

# Материал на декартовой сетке (из Главы III.2)
mat_data = np.load(f'data/dev_3_2_fault_material{angle_suffix}.npz')
material_grid = mat_data['material_grid']   # (2350, 560, 3) — E, nu, rho
coords_grid = mat_data['coords_grid']       # (2350, 560, 2) — x, y
fault_x = float(mat_data['fault_x'])
fault_throw = float(mat_data['fault_throw'])
model_bottom_depth = float(mat_data['model_bottom_depth'])

print(f"Материал: {material_grid.shape}")
print(f"  X: {coords_grid[0,0,0]:.1f} .. {coords_grid[-1,0,0]:.1f} м")
print(f"  Y: {coords_grid[0,0,1]:.1f} .. {coords_grid[0,-1,1]:.1f} м")
print(f"  Шаг: dx={coords_grid[1,0,0]-coords_grid[0,0,0]:.1f}, "
      f"dy={coords_grid[0,1,1]-coords_grid[0,0,1]:.1f} м")

# Геометрия с разломом (из Главы III.1)
geo_data = np.load(f'data/dev_3_1_fault_layer_boundaries{angle_suffix}.npz', allow_pickle=True)
faulted_boundaries = geo_data['layer_boundaries_array']  # (75, 1176)
distances = geo_data['distances']                         # (1176,)
fault_angle = float(geo_data['fault_angle'])

n_layers = faulted_boundaries.shape[0]
print(f"\nГеометрия: {n_layers} слоёв, {len(distances)} точек профиля")
print(f"  Разлом: x={fault_x:.0f} м, throw={fault_throw:.0f} м, angle={fault_angle}°")


""" ## Построение индексов слоёв на декартовой сетке ## """

"""
Для каждой ячейки (ix, iy) декартовой сетки определяем номер слоя (0..74).
Граница каждого слоя интерполируется из `faulted_boundaries` (10 м шаг)
в позиции 5 м сетки.
"""

x_cells = coords_grid[:, 0, 0]  # (2350,)
y_cells = coords_grid[0, :, 1]  # (560,)
nx, ny = len(x_cells), len(y_cells)

# Добавляем поверхность (z=0) как верхнюю границу
all_boundaries = np.vstack([np.zeros((1, len(distances))), faulted_boundaries])  # (76, 1176)

# Интерполируем границы на 5 м сетку по X
# distances — шаг 10 м, x_cells — шаг 5 м
from scipy.interpolate import interp1d

boundaries_5m = np.zeros((76, nx))
for bnd_idx in range(76):
    f_interp = interp1d(distances, all_boundaries[bnd_idx], kind='linear',
                        fill_value='extrapolate')
    boundaries_5m[bnd_idx] = f_interp(x_cells)

print(f"\nГраницы интерполированы на сетку 5 м: {boundaries_5m.shape}")

# Определяем слой для каждой ячейки
layer_indexes_grid = np.full((nx, ny), -1, dtype=np.int32)

for ix in range(nx):
    # Границы для данного столбца (76 значений глубины)
    col_bounds = boundaries_5m[:, ix]
    for iy in range(ny):
        y = y_cells[iy]
        # Находим слой: y между col_bounds[layer] и col_bounds[layer+1]
        idx = np.searchsorted(col_bounds, y) - 1
        if 0 <= idx < n_layers:
            layer_indexes_grid[ix, iy] = idx

# Статистика
valid_mask = layer_indexes_grid >= 0
print(f"Заполнено: {valid_mask.sum()} из {nx*ny} ячеек "
      f"({100*valid_mask.sum()/(nx*ny):.1f}%)")
print(f"Минимальный слой: {layer_indexes_grid[valid_mask].min()}, "
      f"максимальный: {layer_indexes_grid[valid_mask].max()}")

# Подсчёт ячеек по слоям
cells_per_layer = np.array([(layer_indexes_grid == i).sum() for i in range(n_layers)])
print(f"Ячеек на слой: min={cells_per_layer.min()}, max={cells_per_layer.max()}, "
      f"mean={cells_per_layer.mean():.0f}")


""" ## Загрузка FC-модели (заглушки) ## """

fc_model_path = f'data/dev_3_3_model_stub{angle_suffix}.fc'
print(f"\n{'='*60}")
print(f"Загрузка FC модели: {fc_model_path}")
fc = FCModel.load(fc_model_path)
print(f"Материалов в модели: {len(fc.materials)}")


""" ## Заполнение материалов ## """

"""
Для каждого из 75 материалов:
1. Находим все ячейки декартовой сетки, принадлежащие данному слою
2. Записываем координаты (x, y) и свойства (E, ν, ρ) в FC-модель
3. Вычисляем и записываем демпфирование (квадратичная рампа на краях)

Демпфирование: зоны поглощения шириной 250 м на краях и дне модели.
Максимальный коэффициент = 100 (как в Главе I.7).
"""

DAMP_WIDTH = 250.0    # ширина зоны поглощения, м
DAMP_MAX = 100.0      # максимальный коэффициент демпфирования
X_MIN_DAMP = DAMP_WIDTH            # левая граница демпфирования (x < 250)
X_MAX_DAMP = coords_grid[-1, 0, 0] + coords_grid[1, 0, 0] - coords_grid[0, 0, 0] - DAMP_WIDTH  # правый край - 250
Y_MAX_DAMP = model_bottom_depth - DAMP_WIDTH  # нижняя граница демпфирования (y > 2550)

print(f"\nПараметры демпфирования:")
print(f"  Ширина зоны: {DAMP_WIDTH:.0f} м")
print(f"  Макс. коэффициент: {DAMP_MAX:.0f}")
print(f"  Левый край: x < {X_MIN_DAMP:.0f} м")
print(f"  Правый край: x > {X_MAX_DAMP:.0f} м")
print(f"  Нижний край: y > {Y_MAX_DAMP:.0f} м")

print(f"\n{'='*60}")
print("Заполнение материалов:")

# Вспомогательная функция: создание табличного FCData (зависимость от X, Y)
def make_table_data(x, y, values):
    return FCData(
        value=FCValue(values, 'array'),
        type_code=-1,
        table=[
            FCDependencyColumn('TABULAR_X', FCValue(x, 'array')),
            FCDependencyColumn('TABULAR_Y', FCValue(y, 'array')),
        ]
    )


for layer_idx in range(n_layers):
    mat_idx = layer_idx + 1  # Индексация материалов с 1

    # Ячейки данного слоя
    coords_in_layer = np.argwhere(layer_indexes_grid == layer_idx)
    coords_count = len(coords_in_layer)

    if coords_count == 0:
        print(f"  Слой {mat_idx:2d}: пусто (пропущен)")
        continue

    # Координаты и свойства
    x_coords = coords_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 0].astype(np.float64)
    y_coords = coords_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 1].astype(np.float64)
    E_data = material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 0].astype(np.float64)
    nu_data = material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 1].astype(np.float64)
    rho_data = material_grid[coords_in_layer[:, 0], coords_in_layer[:, 1], 2].astype(np.float64)


    # Обновляем модуль Юнга (YOUNG_MODULE)
    fc.materials[mat_idx].properties['elasticity'][0][0].data = make_table_data(x_coords, y_coords, E_data)

    # Обновляем коэффициент Пуассона (POISSON_RATIO)
    fc.materials[mat_idx].properties['elasticity'][0][1].data = make_table_data(x_coords, y_coords, nu_data)

    # Обновляем плотность (DENSITY)
    fc.materials[mat_idx].properties['common'][0][0].data = make_table_data(x_coords, y_coords, rho_data)

    # Демпфирование (квадратичная рампа)
    mass_matrix_damping = np.zeros(coords_count, dtype=np.float64)

    for j in range(coords_count):
        x = x_coords[j]
        y = y_coords[j]

        damping_x = 0.0
        damping_y = 0.0

        if x < X_MIN_DAMP:
            damping_x = DAMP_MAX * ((X_MIN_DAMP - x) / DAMP_WIDTH) ** 2
        elif x > X_MAX_DAMP:
            damping_x = DAMP_MAX * ((x - X_MAX_DAMP) / DAMP_WIDTH) ** 2

        if y > Y_MAX_DAMP:
            damping_y = DAMP_MAX * ((y - Y_MAX_DAMP) / DAMP_WIDTH) ** 2

        mass_matrix_damping[j] = max(damping_x, damping_y)

    # Добавляем демпфирование (MASS_DAMPING_RATIO) — свойства нет в заглушке, добавляем
    damp_prop = fc.materials[mat_idx].add_property('common', 'MASS_DAMPING_RATIO', 0.0)
    damp_prop.data = make_table_data(x_coords, y_coords, mass_matrix_damping)

    n_damped = np.count_nonzero(mass_matrix_damping)
    print(f"  Слой {mat_idx:2d}: {coords_count:6d} узлов, "
          f"E=[{E_data.min():.2e}..{E_data.max():.2e}], "
          f"damped={n_damped}")


""" ## Сохранение FC-модели ## """

output_fc_path = f'data/dev_3_4_model_calc{angle_suffix}.fc'
fc.save(output_fc_path)
print(f"\n{'='*60}")
print(f"FC модель сохранена: {output_fc_path}")
print(f"{'='*60}")


""" ## Визуализация: проверка слоёв ## """

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Индексы слоёв
ax = axes[0]
X, Y = np.meshgrid(x_cells, y_cells, indexing='ij')
c = ax.pcolormesh(X, Y, layer_indexes_grid, cmap='prism', shading='auto')
plt.colorbar(c, ax=ax, label='Номер слоя')
z_fl = np.linspace(0, model_bottom_depth, 100)
x_fl = fault_x + z_fl * np.tan(np.radians(fault_angle))
ax.plot(x_fl, z_fl, 'k--', linewidth=1.5, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title('Индексы слоёв на декартовой сетке')
ax.set_ylim(model_bottom_depth, 0)
ax.legend()

# Демпфирование (пример из последнего слоя)
ax = axes[1]
# Построим карту демпфирования для всей сетки
damping_map = np.zeros((nx, ny))
for ix in range(nx):
    for iy in range(ny):
        x = x_cells[ix]
        y = y_cells[iy]
        dx_val = 0.0
        dy_val = 0.0
        if x < X_MIN_DAMP:
            dx_val = DAMP_MAX * ((X_MIN_DAMP - x) / DAMP_WIDTH) ** 2
        elif x > X_MAX_DAMP:
            dx_val = DAMP_MAX * ((x - X_MAX_DAMP) / DAMP_WIDTH) ** 2
        if y > Y_MAX_DAMP:
            dy_val = DAMP_MAX * ((y - Y_MAX_DAMP) / DAMP_WIDTH) ** 2
        damping_map[ix, iy] = max(dx_val, dy_val)

c = ax.pcolormesh(X, Y, damping_map, cmap='hot_r', shading='auto')
plt.colorbar(c, ax=ax, label='Коэффициент демпфирования')
ax.plot(x_fl, z_fl, 'c--', linewidth=1, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title('Зоны поглощения (демпфирования)')
ax.set_ylim(model_bottom_depth, 0)
ax.legend()

plt.tight_layout()
plt.savefig(f'img/dev_3_4_layers_damping{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_4_layers_damping{angle_suffix}.png")


""" ## Визуализация: проверка материала из FC ## """

"""
Считываем обратно свойства из FC-модели для всех слоёв и строим карты Vp, Vs, ρ,
чтобы убедиться, что данные записаны корректно и смещение на разломе выражено.
"""

# Собираем свойства из FC обратно на сетку
E_check = np.full((nx, ny), np.nan)
nu_check = np.full((nx, ny), np.nan)
rho_check = np.full((nx, ny), np.nan)

for layer_idx in range(n_layers):
    mat_idx = layer_idx + 1
    coords_in_layer = np.argwhere(layer_indexes_grid == layer_idx)
    if len(coords_in_layer) == 0:
        continue
    # Считываем из FC-модели (данные уже записаны туда)
    E_fc = fc.materials[mat_idx].properties['elasticity'][0][0].data.value.data
    nu_fc = fc.materials[mat_idx].properties['elasticity'][0][1].data.value.data
    rho_fc = fc.materials[mat_idx].properties['common'][0][0].data.value.data

    E_check[coords_in_layer[:, 0], coords_in_layer[:, 1]] = E_fc
    nu_check[coords_in_layer[:, 0], coords_in_layer[:, 1]] = nu_fc
    rho_check[coords_in_layer[:, 0], coords_in_layer[:, 1]] = rho_fc

# Вычисляем скорости
Vp = np.sqrt(E_check / rho_check * (1 - nu_check) / (1 + nu_check) / (1 - 2 * nu_check))
Vs = np.sqrt(E_check / rho_check / 2 / (1 + nu_check))

print(f"\nМатериал из FC-модели (проверка):")
print(f"  Vp: {np.nanmin(Vp):.0f} .. {np.nanmax(Vp):.0f} м/с")
print(f"  Vs: {np.nanmin(Vs):.0f} .. {np.nanmax(Vs):.0f} м/с")
print(f"  ρ:  {np.nanmin(rho_check):.0f} .. {np.nanmax(rho_check):.0f} кг/м³")


""" ### Vp — полный вид """

fig, ax = plt.subplots(figsize=(16, 8))
c = ax.pcolormesh(X, Y, Vp, cmap='rainbow', shading='auto')
plt.colorbar(c, ax=ax, label='Vp (м/с)')
z_fl = np.linspace(0, model_bottom_depth, 100)
x_fl = fault_x + z_fl * np.tan(np.radians(fault_angle))
ax.plot(x_fl, z_fl, 'k--', linewidth=1.5, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title(f'Vp — FC-модель с разломом (throw={fault_throw:.0f} м)')
ax.set_xlim(x_cells[0], x_cells[-1])
ax.set_ylim(model_bottom_depth, 0)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'img/dev_3_4_Vp{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_4_Vp{angle_suffix}.png")


""" ### Vs — полный вид """

fig, ax = plt.subplots(figsize=(16, 8))
c = ax.pcolormesh(X, Y, Vs, cmap='rainbow', shading='auto')
plt.colorbar(c, ax=ax, label='Vs (м/с)')
z_fl = np.linspace(0, model_bottom_depth, 100)
x_fl = fault_x + z_fl * np.tan(np.radians(fault_angle))
ax.plot(x_fl, z_fl, 'k--', linewidth=1.5, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title(f'Vs — FC-модель с разломом (throw={fault_throw:.0f} м)')
ax.set_xlim(x_cells[0], x_cells[-1])
ax.set_ylim(model_bottom_depth, 0)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'img/dev_3_4_Vs{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_4_Vs{angle_suffix}.png")


""" ### Плотность — полный вид """

fig, ax = plt.subplots(figsize=(16, 8))
c = ax.pcolormesh(X, Y, rho_check, cmap='rainbow', shading='auto')
plt.colorbar(c, ax=ax, label='ρ (кг/м³)')
z_fl = np.linspace(0, model_bottom_depth, 100)
x_fl = fault_x + z_fl * np.tan(np.radians(fault_angle))
ax.plot(x_fl, z_fl, 'k--', linewidth=1.5, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title(f'Плотность — FC-модель с разломом (throw={fault_throw:.0f} м)')
ax.set_xlim(x_cells[0], x_cells[-1])
ax.set_ylim(model_bottom_depth, 0)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'img/dev_3_4_rho{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_4_rho{angle_suffix}.png")


""" ### Vp — детальный вид зоны разлома """

x_detail_left = fault_x - 800
x_detail_right = fault_x + 800
ix_left = np.searchsorted(x_cells, x_detail_left)
ix_right = np.searchsorted(x_cells, x_detail_right)

fig, ax = plt.subplots(figsize=(12, 10))
X_d, Y_d = np.meshgrid(x_cells[ix_left:ix_right], y_cells, indexing='ij')
c = ax.pcolormesh(X_d, Y_d, Vp[ix_left:ix_right, :], cmap='rainbow', shading='auto')
plt.colorbar(c, ax=ax, label='Vp (м/с)')
z_fl = np.linspace(0, model_bottom_depth, 100)
x_fl = fault_x + z_fl * np.tan(np.radians(fault_angle))
ax.plot(x_fl, z_fl, 'k--', linewidth=2, label='Разлом')
ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.set_title('Vp — зона разлома (детальный вид)')
ax.set_xlim(x_detail_left, x_detail_right)
ax.set_ylim(model_bottom_depth, 0)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'img/dev_3_4_Vp_detail{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_4_Vp_detail{angle_suffix}.png")


""" ## Сводка ## """

file_size_mb = Path(output_fc_path).stat().st_size / 1024 / 1024
print(f"\n{'='*60}")
print(f"=== Сводка Главы IV ===")
print(f"{'='*60}")
print(f"\nВходные данные:")
print(f"  FC заглушка: {fc_model_path}")
print(f"  Материал: data/dev_3_2_fault_material{angle_suffix}.npz ({material_grid.shape})")
print(f"  Геометрия: data/dev_3_1_fault_layer_boundaries{angle_suffix}.npz ({n_layers} слоёв)")
print(f"\nВыходные данные:")
print(f"  FC модель: {output_fc_path} ({file_size_mb:.1f} МБ)")
print(f"\nПараметры:")
print(f"  Материалов: {n_layers}")
print(f"  Сетка материала: {nx}×{ny} (шаг 5×5 м)")
print(f"  Демпфирование: ширина {DAMP_WIDTH:.0f} м, макс. {DAMP_MAX:.0f}")
print(f"  Всего узлов материала: {valid_mask.sum()}")


""" ## Выводы ## """

"""
Выполнена сборка FC-модели с реальным материалом для модели с разломом:

1. Загружена FC-модель (заглушка с 75 материалами из Fidesys)
2. Построены индексы слоёв на декартовой сетке 5×5 м по границам с разломом
3. Для каждого из 75 материалов заполнены:
   - Модуль Юнга (E) на интерполяционных точках (x, y)
   - Коэффициент Пуассона (ν)
   - Плотность (ρ)
   - Демпфирование (квадратичная рампа на краях)
4. Модель сохранена в `data/dev_3_4_model_calc.fc`

Материал следует смещению разлома: правый блок берёт свойства со сдвигом,
что обеспечивается данными из Главы III.2.
"""
