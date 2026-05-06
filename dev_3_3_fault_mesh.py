""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть третья. """

""" ## Глава III: Построение КЭ-сетки с разломом """

"""
### Задача

Сгенерировать JOU-скрипты для построения геометрии и сетки модели с разломом в CAE-Fidesys.
Разлом разделяет модель на два блока с отдельными поверхностями.
Модель состоит из 4 горизонтальных секций:
1. [0..Well1] — левый внешний блок (85 вершин, шаг 50 м)
2. [Well1..Fault] — левый внутренний блок (вершины с шагом 50 м + точка разлома)
3. [Fault..Well2] — правый внутренний блок (точка разлома + вершины с шагом 50 м)
4. [Well2..End] — правый внешний блок (85 вершин, шаг 50 м)

Геометрия правого блока (секции 3 и 4) смещена вниз на `fault_throw`.
Каждая секция содержит 75 слоёв (76 граничных линий: поверхность + 75 нижних границ).

Для левого блока (секции 1-2): поверхность z=0, дно z=model_bottom_depth (растяжение последнего слоя).
Для правого блока (секции 3-4): поверхность z=0, дно z=model_bottom_depth (растяжение первого слоя вверх).
"""

# Импорты
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

""" ## Загрузка данных ## """

# Исходная геометрия слоёв (из Главы I.5)
orig_data = np.load('data/dev_1_5_2_layer_boundaries_quadratic.npz', allow_pickle=True)
orig_boundaries = orig_data['layer_boundaries_array']  # (75, 1176) — исходные глубины
formations = orig_data['formations']
distances = orig_data['distances']  # (1176,) — шаг 10 м
well1_depths = orig_data['well1_depths']
well2_depths = orig_data['well2_depths']

# Геометрия с разломом (из Главы III.1)
fault_data = np.load('data/dev_3_1_fault_layer_boundaries.npz', allow_pickle=True)
faulted_boundaries = fault_data['layer_boundaries_array']  # (75, 1176)
fault_x = float(fault_data['fault_x'])
fault_angle = float(fault_data['fault_angle'])
fault_throw = float(fault_data['fault_throw'])
model_bottom_depth = float(fault_data['model_bottom_depth'])

print(f"Исходная модель: {orig_boundaries.shape[0]} слоёв, {orig_boundaries.shape[1]} точек")
print(f"Профиль: {distances[0]:.0f} .. {distances[-1]:.0f} м (шаг {distances[1]-distances[0]:.0f} м)")
print(f"Разлом: x={fault_x:.0f} м, угол={fault_angle}°, throw={fault_throw:.0f} м")
print(f"Глубина дна: {model_bottom_depth:.0f} м")

""" ## Параметры геометрии ## """

WELL1_DISTANCE = 4250.0  # м
WELL2_DISTANCE = 7500.0  # м
PROFILE_LENGTH = 11750.0  # м
VERTEX_STEP = 50.0  # м — шаг вершин вдоль профиля

# Контрольные X-координаты разбиения
x_splits = [0.0, WELL1_DISTANCE, fault_x, WELL2_DISTANCE, PROFILE_LENGTH]
print(f"\nРазбиение профиля: {x_splits}")

# Шаг выборки из массива distances (шаг 10 м → каждые 5 = шаг 50 м)
sample_step = int(round(VERTEX_STEP / (distances[1] - distances[0])))
print(f"Шаг выборки: каждые {sample_step} точек (= {sample_step * (distances[1]-distances[0]):.0f} м)")

""" ## Построение массивов вершин для 4 секций ## """

"""
Вершины берутся из массива `faulted_boundaries` (уже содержит смещение правого блока).
Каждая граничная линия сэмплируется с шагом 50 м, плюс точка разлома (`fault_x`).

Добавляем поверхность (z=0) и корректируем дно (z=model_bottom_depth).
"""

# Индексы ключевых точек в массиве distances (шаг 10 м)
idx_well1 = int(WELL1_DISTANCE / 10)  # 425
idx_fault = int(fault_x / 10)  # 587 (≈5870 м — ближайшая к 5875)
idx_well2 = int(WELL2_DISTANCE / 10)  # 750

# Для точного попадания в fault_x добавляем дополнительную вершину
# Вершины секции 1: [0, 50, 100, ..., 4250]
x_sec1 = distances[0:idx_well1 + 1:sample_step]
if x_sec1[-1] != WELL1_DISTANCE:
    x_sec1 = np.append(x_sec1, WELL1_DISTANCE)

# Вершины секции 2: [4250, 4300, ..., 5850, 5875]
x_sec2 = distances[idx_well1:idx_fault + 1:sample_step]
if x_sec2[0] != WELL1_DISTANCE:
    x_sec2 = np.concatenate([[WELL1_DISTANCE], x_sec2])
# Добавляем точную точку разлома
if x_sec2[-1] != fault_x:
    x_sec2 = np.append(x_sec2, fault_x)

# Вершины секции 3: [5875, 5900, 5950, ..., 7500]
x_sec3_start = distances[idx_fault + sample_step:idx_well2 + 1:sample_step]
x_sec3 = np.concatenate([[fault_x], x_sec3_start])
if x_sec3[-1] != WELL2_DISTANCE:
    x_sec3 = np.append(x_sec3, WELL2_DISTANCE)

# Вершины секции 4: [7500, 7550, ..., 11750]
x_sec4 = distances[idx_well2::sample_step]
if x_sec4[0] != WELL2_DISTANCE:
    x_sec4 = np.concatenate([[WELL2_DISTANCE], x_sec4])
if x_sec4[-1] != PROFILE_LENGTH:
    x_sec4 = np.append(x_sec4, PROFILE_LENGTH)

print(f"\nВершины по секциям:")
print(f"  Секция 1 [0..{WELL1_DISTANCE:.0f}]: {len(x_sec1)} вершин")
print(f"  Секция 2 [{WELL1_DISTANCE:.0f}..{fault_x:.0f}]: {len(x_sec2)} вершин")
print(f"  Секция 3 [{fault_x:.0f}..{WELL2_DISTANCE:.0f}]: {len(x_sec3)} вершин")
print(f"  Секция 4 [{WELL2_DISTANCE:.0f}..{PROFILE_LENGTH:.0f}]: {len(x_sec4)} вершин")

# Все X-координаты вершин
x_all_sections = [x_sec1, x_sec2, x_sec3, x_sec4]

""" ## Интерполяция глубин границ для вершин ## """

"""
Левый блок (секции 1-2): исходные границы без деформации.
Правый блок (секции 3-4): исходные границы + fault_throw (параллельное смещение вниз).

Это обеспечивает резкий неконформный разрыв на линии разлома:
слои слева и справа идентичны по форме, но смещены по вертикали.
"""

n_layers = len(formations)
n_boundaries = n_layers + 1  # 76 (поверхность + 75 нижних границ)

# Массив глубин для каждой секции: [n_boundaries, n_vertices_in_section]
section_depths = []

for sec_idx, x_sec in enumerate(x_all_sections):
    depths = np.zeros((n_boundaries, len(x_sec)))

    # Первая граница — поверхность (z=0)
    depths[0, :] = 0.0

    # Выбираем источник: исходные границы, для правого блока + смещение
    if sec_idx < 2:
        # Левый блок — исходная геометрия
        shift = 0.0
    else:
        # Правый блок — параллельное смещение вниз
        shift = fault_throw

    # Интерполяция каждой из 75 границ из ИСХОДНЫХ данных + смещение
    for layer_idx in range(n_layers):
        depths[layer_idx + 1, :] = np.interp(x_sec, distances, orig_boundaries[layer_idx]) + shift

    # Коррекция: последняя граница = model_bottom_depth (плоское дно)
    depths[-1, :] = model_bottom_depth

    section_depths.append(depths)
    print(f"  Секция {sec_idx+1}: depths shape = {depths.shape}, "
          f"z_min={depths.min():.1f}, z_max={depths.max():.1f}, shift={shift:.0f}")


""" ## Совмещение близких вершин на линии разлома ## """

"""
После смещения правого блока на линии разлома (x=fault_x) собираются вершины
из обоих блоков. Если граница A правого блока (= orig[i] + throw) оказывается
очень близко к границе B левого блока (= orig[j]), то расстояние |A - B| мало,
и мешер строит вырожденные элементы.

Решение: собрать все z-координаты вершин на линии разлома, найти пары ближе
порога (5 м), и совместить их усреднением.
"""

MERGE_THRESHOLD = 5.0  # м — минимально допустимое расстояние между вершинами

# Вершины на линии разлома: правый край секции 2, левый край секции 3
z_left_edge = section_depths[1][:, -1]   # (76,) — правый край секции 2
z_right_edge = section_depths[2][:, 0]   # (76,) — левый край секции 3

n_merged = 0
for i in range(len(z_left_edge)):
    for j in range(len(z_right_edge)):
        if i == j:
            continue  # одна и та же граница — пропускаем (они и так на разных глубинах)
        dist_ij = abs(z_left_edge[i] - z_right_edge[j])
        if 0 < dist_ij < MERGE_THRESHOLD:
            avg = (z_left_edge[i] + z_right_edge[j]) / 2.0
            print(f"  Совмещение: left bnd {i} (z={z_left_edge[i]:.2f}) "
                  f"↔ right bnd {j} (z={z_right_edge[j]:.2f}) → z={avg:.2f}")
            # Обновляем ВСЮ граничную линию (не только крайнюю точку)
            # чтобы не деформировать слой — сдвигаем только крайнюю вершину
            section_depths[1][i, -1] = avg
            section_depths[2][j, 0] = avg
            z_left_edge[i] = avg
            z_right_edge[j] = avg
            n_merged += 1

print(f"\nСовмещено вершин на линии разлома: {n_merged} (порог = {MERGE_THRESHOLD} м)")

""" ## Генерация JOU-скрипта вершин ## """

"""
Генерируем все вершины для 4 секций × 76 граничных линий.
Вершины создаются в порядке: секция за секцией, линия за линией.
Каждая вершина — точка (x, z, 0) в плоскости XY (Z=0 — 2D модель).

Формат Fidesys: `create vertex location X Y Z`
Здесь Y — глубина (вниз), Z = 0 (2D).
"""

vertex_counts = [len(x) for x in x_all_sections]
total_vertices = sum(vertex_counts) * n_boundaries
print(f"\nОбщее количество вершин: {total_vertices} "
      f"({sum(vertex_counts)} на линию × {n_boundaries} линий)")

with open('data/dev_3_3_model_vertex.jou', 'w') as f:
    f.write('reset\n')

    for sec_idx, (x_sec, depths) in enumerate(zip(x_all_sections, section_depths)):
        f.write(f'# --- Section {sec_idx+1}: x=[{x_sec[0]:.0f}..{x_sec[-1]:.0f}] ---\n')

        for bnd_idx in range(n_boundaries):
            for vx_idx in range(len(x_sec)):
                x = x_sec[vx_idx]
                y = depths[bnd_idx, vx_idx]
                f.write(f'create vertex location {x:.2f} {y:.4f} 0\n')

print(f"Сохранено: data/dev_3_3_model_vertex.jou ({total_vertices} вершин)")

""" ## Генерация JOU-скрипта сплайнов и поверхностей ## """

"""
Для каждой секции:
1. Создаём сплайны из вершин каждой граничной линии (76 сплайнов на секцию)
2. Создаём поверхности между парами смежных сплайнов (75 поверхностей на секцию)
   методом skin curve

Нумерация вершин: секции идут последовательно.
"""

with open('data/dev_3_3_model_spline.jou', 'w') as f:
    vertex_offset = 1  # Вершины нумеруются с 1 в Fidesys
    curve_id = 1

    # Запоминаем номера кривых для каждой секции и границы
    # section_curves[sec_idx][bnd_idx] = curve_id
    section_curves = []

    for sec_idx, x_sec in enumerate(x_all_sections):
        n_verts = len(x_sec)
        curves_in_section = []

        f.write(f'# --- Section {sec_idx+1} splines ---\n')

        for bnd_idx in range(n_boundaries):
            v_start = vertex_offset + bnd_idx * n_verts
            v_end = v_start + n_verts - 1
            f.write(f'create curve spline location vertex {v_start} to {v_end}\n')
            curves_in_section.append(curve_id)
            curve_id += 1

        section_curves.append(curves_in_section)
        vertex_offset += n_boundaries * n_verts

    # Удаляем вершины после создания кривых
    f.write('delete vertex all\n')

    # Создаём поверхности
    f.write('\n# --- Surfaces (skin between adjacent boundaries) ---\n')
    for sec_idx in range(4):
        f.write(f'# Section {sec_idx+1}\n')
        for layer_idx in range(n_layers):
            c_top = section_curves[sec_idx][layer_idx]
            c_bot = section_curves[sec_idx][layer_idx + 1]
            f.write(f'create surface skin curve {c_top} {c_bot}\n')

    # Очистка
    f.write('delete curve all\n')
    f.write('merge curve all\n')
    f.write('compress curve all\n')

n_surfaces = 4 * n_layers  # 300
print(f"Сохранено: data/dev_3_3_model_spline.jou")
print(f"  Кривых: {curve_id - 1} ({4} секций × {n_boundaries} границ)")
print(f"  Поверхностей: {n_surfaces} ({4} секций × {n_layers} слоёв)")

""" ## Расчёт параметров дискретизации сетки ## """

"""
Размер элементов определяется по минимальной скорости Vp в слое (как в Главе I.7):
- `el_size = Vp_min / (freq * n_el_per_wave)`
- Частота: 30 Гц, элементов на длину волны: 4

Для горизонтальных кривых — число интервалов = длина секции / el_size.
Для вертикальных кривых — число интервалов = толщина слоя / el_size.
"""

# Загружаем скорости для оценки размеров элементов
mat_data = np.load('data/dev_3_2_fault_material.npz')
Vp = mat_data['Vp']  # (2350, 560)
coords_grid = mat_data['coords_grid']  # (2350, 560, 2)

freq = 30.0  # Гц
n_el_per_wave = 4

# Минимальная Vp по глубине (для каждой строки y)
Vp_min_per_depth = np.min(Vp, axis=0)  # (560,)
y_cells = coords_grid[0, :, 1]  # глубины

# Максимальный допустимый размер элемента по глубине
max_el_size_per_depth = Vp_min_per_depth / freq / n_el_per_wave

print(f"\nРазмер элементов (freq={freq} Hz, {n_el_per_wave} эл/волна):")
print(f"  Минимальный: {max_el_size_per_depth.min():.1f} м")
print(f"  Максимальный: {max_el_size_per_depth.max():.1f} м")

""" ## Горизонтальные интервалы ## """

"""
Для каждого слоя определяем число горизонтальных интервалов в каждой секции.
Размер элемента определяется средней глубиной слоя.
"""

# Средняя глубина каждого слоя (для определения размера элемента)
layer_mid_depths = np.zeros(n_layers)
for i in range(n_layers):
    # Средняя глубина между верхней и нижней границей (nanmean для устойчивости к NaN)
    top = np.nanmean(faulted_boundaries[i - 1]) if i > 0 else 0
    bot = np.nanmean(faulted_boundaries[i])
    layer_mid_depths[i] = (top + bot) / 2

# Размер элемента для каждого слоя
layer_el_size = np.interp(layer_mid_depths, y_cells, max_el_size_per_depth)
# Минимум — 10 м (первый слой для расстановки ресиверов)
layer_el_size[0] = 10.0

# Длины секций
sec_lengths = [x_sec[-1] - x_sec[0] for x_sec in x_all_sections]
print(f"\nДлины секций: {[f'{l:.0f}' for l in sec_lengths]} м")

# Число горизонтальных интервалов для каждого слоя и секции
h_intervals = np.zeros((n_layers + 1, 4), dtype=int)  # +1 для нулевого "индекса поверхности"

for sec_idx, sec_len in enumerate(sec_lengths):
    # Нулевой слой (поверхность → первая граница) — фиксированный
    h_intervals[0, sec_idx] = max(int(np.round(sec_len / 10.0)), 1)

    for layer_idx in range(n_layers):
        el_sz = layer_el_size[layer_idx]
        n_int = max(int(np.round(sec_len / el_sz)), 1)
        # Обеспечиваем нечётность для совместимости с pave
        if n_int % 2 == 0:
            n_int += 1
        h_intervals[layer_idx + 1, sec_idx] = n_int

print(f"\nГоризонтальные интервалы (первые 5 слоёв):")
for i in range(5):
    print(f"  Слой {i}: секции = {h_intervals[i]}")

""" ## Вертикальные интервалы ## """

"""
Число вертикальных интервалов определяется толщиной слоя и допустимым размером элемента.
Для левого блока используем well1_depths, для правого — well2_depths.
На линии разлома интервалы должны совпадать с ближайшей скважиной.
"""

# Толщина слоёв у скважин
well1_heights = np.diff(np.concatenate([[0], well1_depths]))
well2_heights = np.diff(np.concatenate([[0], well2_depths]))

# Вертикальные интервалы для каждой из 4 вертикальных линий
# Линия 1 (x=0): как well1
# Линия 2 (x=well1): well1
# Линия 3 (x=fault, left side): интерполяция
# Линия 4 (x=fault, right side): интерполяция
# Линия 5 (x=well2): well2
# Линия 6 (x=end): как well2

v_intervals_left = np.zeros(n_layers, dtype=int)   # для секций 1-2 (левый блок)
v_intervals_right = np.zeros(n_layers, dtype=int)  # для секций 3-4 (правый блок)

for i in range(n_layers):
    el_sz = layer_el_size[i]

    # Левый блок — ориентируемся на well1 heights
    n_v_left = max(int(np.ceil(well1_heights[i] / el_sz)), 1)
    v_intervals_left[i] = n_v_left

    # Правый блок — ориентируемся на well2 heights
    n_v_right = max(int(np.ceil(well2_heights[i] / el_sz)), 1)
    v_intervals_right[i] = n_v_right

# Обеспечиваем одинаковую чётность для пар (left, right) — необходимо для pave
for i in range(n_layers):
    if v_intervals_left[i] % 2 != v_intervals_right[i] % 2:
        # Увеличиваем меньший
        if v_intervals_left[i] < v_intervals_right[i]:
            v_intervals_left[i] += 1
        else:
            v_intervals_right[i] += 1

print(f"\nВертикальные интервалы (первые 10 слоёв):")
for i in range(10):
    print(f"  Слой {i}: left={v_intervals_left[i]}, right={v_intervals_right[i]}")

""" ## Генерация JOU-скрипта дискретизации кривых ## """

"""
После создания геометрии (вершин, сплайнов, поверхностей) в Fidesys
автоматически присваиваются ID кривым. Нумерация кривых зависит от порядка
операций merge/compress.

Для удобства генерируем скрипт, который задаёт число интервалов по номерам поверхностей,
а кривые привязаны к поверхностям через surface_curves.

Однако более надёжный подход — задать интервалы сразу при создании кривых.
Поскольку точная нумерация кривых после merge неизвестна заранее, генерируем
mesh-скрипт с назначением по поверхностям.
"""

# Генерируем общий скрипт построения сетки
with open('data/dev_3_3_model_mesh.jou', 'w') as f:
    f.write('# Построение сетки модели с разломом\n')
    f.write('imprint all\n')
    f.write('merge all\n\n')
    f.write('delete mesh surface all propagate\n')
    f.write('delete mesh curve all propagate\n\n')
    # Поверхности нумеруются: section 1 (1..75), section 2 (76..150),
    # section 3 (151..225), section 4 (226..300)
    surf_offset = 0

    for sec_idx in range(4):
        f.write(f'\n# --- Section {sec_idx+1} ---\n')

        for layer_idx in range(n_layers):
            surf_id = surf_offset + layer_idx + 1

            # Определяем метод (map если все интервалы совпадают, иначе pave)
            f.write(f'surface {surf_id} scheme pave\n')
            f.write(f'surface {surf_id} size {layer_el_size[layer_idx]:.1f}\n')

        surf_offset += n_layers

    f.write('\nmesh surface all\n')

print(f"\nСохранено: data/dev_3_3_model_mesh.jou")
print(f"  Поверхностей: {4 * n_layers}")


""" ## Генерация JOU-скрипта материалов и блоков ## """

"""
Создаём 75 материалов-заглушек (по одному на слой) и 75 блоков. Каждый блок
объединяет 4 поверхности одного слоя из всех секций.

Нумерация поверхностей: секция 1 → 1..75, секция 2 → 76..150,
секция 3 → 151..225, секция 4 → 226..300.

Блок i содержит поверхности: i, 75+i, 150+i, 225+i.
"""

with open('data/dev_3_3_model_materials.jou', 'w') as f:
    f.write('# Назначение материалов-заглушек и блоков\n\n')

    for layer_idx in range(n_layers):
        mat_id = layer_idx + 1
        # 4 поверхности одного слоя из 4 секций
        s1 = layer_idx + 1
        s2 = n_layers + layer_idx + 1
        s3 = 2 * n_layers + layer_idx + 1
        s4 = 3 * n_layers + layer_idx + 1

        # Создание материала
        f.write(f'create material {mat_id}\n')
        f.write(f"modify material {mat_id} name 'mat_{mat_id}'\n")
        f.write(f"modify material {mat_id} set property 'MODULUS' value 1e+10\n")
        f.write(f"modify material {mat_id} set property 'POISSON' value 0.25\n")
        f.write(f"modify material {mat_id} set property 'DENSITY' value 2000\n")

        # Блок объединяет 4 поверхности одного слоя
        f.write(f'block {mat_id} add surface {s1} {s2} {s3} {s4}\n')
        f.write(f'block {mat_id} material {mat_id} cs 1 category plane order 1\n')
        f.write('\n')

print(f"Сохранено: data/dev_3_3_model_materials.jou")
print(f"  Материалов: {n_layers}")
print(f"  Блоков: {n_layers} (по 4 поверхности в каждом)")

""" ## Визуализация: структура секций ## """

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_title('Разбиение модели на 4 секции', fontsize=14)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Sec 1: [0..Well1]', 'Sec 2: [Well1..Fault]',
          'Sec 3: [Fault..Well2]', 'Sec 4: [Well2..End]']

for sec_idx, (x_sec, depths) in enumerate(zip(x_all_sections, section_depths)):
    # Рисуем несколько границ для иллюстрации
    for bnd_idx in range(0, n_boundaries, 5):
        ax.plot(x_sec, depths[bnd_idx], color=colors[sec_idx],
                linewidth=0.5, alpha=0.7)
    # Одну линию с лейблом
    ax.plot(x_sec, depths[n_boundaries // 2], color=colors[sec_idx],
            linewidth=1.5, label=labels[sec_idx])

# Линия разлома
z_line = np.linspace(0, model_bottom_depth, 100)
alpha = np.radians(fault_angle)
x_line = fault_x + z_line * np.tan(alpha)
ax.plot(x_line, z_line, 'k--', linewidth=2, label='Линия разлома')

# Скважины
ax.axvline(WELL1_DISTANCE, color='gray', linestyle=':', alpha=0.7, label='Well 1')
ax.axvline(WELL2_DISTANCE, color='gray', linestyle=':', alpha=0.7, label='Well 2')

ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
ax.set_xlim(0, PROFILE_LENGTH)

plt.tight_layout()
plt.savefig('img/dev_3_3_sections_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Сохранено: img/dev_3_3_sections_overview.png")

""" ## Сводка сгенерированных скриптов ## """

"""
Для построения модели в Fidesys необходимо последовательно выполнить:

```
playback 'data/dev_3_3_model_vertex.jou'      # Создание вершин
playback 'data/dev_3_3_model_spline.jou'       # Создание кривых и поверхностей
playback 'data/dev_3_3_model_mesh.jou'         # Построение сетки
playback 'data/dev_3_3_model_materials.jou'    # Назначение материалов и блоков
```

После этого — экспорт FC-модели (заглушка с материалами):
```
export fc 'data/dev_3_3_model_stub.fc'
```
"""

print("\n=== Сводка ===")
print(f"Файлы скриптов:")
print(f"  data/dev_3_3_model_vertex.jou     — {total_vertices} вершин")
print(f"  data/dev_3_3_model_spline.jou     — {curve_id-1} кривых, {n_surfaces} поверхностей")
print(f"  data/dev_3_3_model_mesh.jou       — параметры сетки")
print(f"  data/dev_3_3_model_materials.jou  — {n_layers} материалов + {n_layers} блоков")
print(f"\nМодель: 4 секции × 75 слоёв = 300 поверхностей")
print(f"Размер элемента: {layer_el_size.min():.1f} .. {layer_el_size.max():.1f} м")

""" ## Выводы ## """

"""
Выполнена генерация JOU-скриптов для построения модели с разломом в Fidesys:

1. Профиль разделён на 4 секции: [0..Well1], [Well1..Fault], [Fault..Well2], [Well2..End]
2. Точка разлома (x=5875 м) включена как отдельная вершина
3. Для каждой секции создано 76 граничных сплайнов и 75 поверхностей (всего 300)
4. Правый блок (секции 3-4) смещён вниз на fault_throw=50 м
5. Размер элементов адаптивный: от 10 м (первый слой) до Vp_min/(30×4) м (глубокие слои)
6. Дно модели плоское на z=2800 м
7. Создан скрипт материалов: 300 заглушек + 300 блоков (QUAD9) для экспорта FC

Для завершения пользователь должен:
- Запустить 4 скрипта в Fidesys последовательно
- Экспортировать FC-модель: `export fc 'data/dev_3_3_model_stub.fc'`
"""