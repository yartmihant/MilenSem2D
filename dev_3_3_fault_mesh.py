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

""" ## Параметр угла разлома ## """

# Угол разлома определяет суффикс входных/выходных файлов (должен совпадать с dev_3_1)
fault_angle_param = 20.0  # градусы
angle_suffix = f'_a{int(fault_angle_param)}'
print(f"Суффикс файлов: {angle_suffix}")


""" ## Загрузка данных ## """

# Исходная геометрия слоёв (из Главы I.5)
orig_data = np.load('data/dev_1_5_2_layer_boundaries_quadratic.npz', allow_pickle=True)
orig_boundaries = orig_data['layer_boundaries_array']  # (75, 1176) — исходные глубины
formations = orig_data['formations']
distances = orig_data['distances']  # (1176,) — шаг 10 м
well1_depths = orig_data['well1_depths']
well2_depths = orig_data['well2_depths']

# Геометрия с разломом (из Главы III.1)
fault_data = np.load(f'data/dev_3_1_fault_layer_boundaries{angle_suffix}.npz', allow_pickle=True)
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

r"""
Модель делится на 4 секции по горизонтали:
- Секция 1: $[0, \text{Well1}]$ — левый блок, далеко от разлома  
- Секция 2: $[\text{Well1}, \text{fault\_line}]$ — левый блок, до линии разлома
- Секция 3: $[\text{fault\_line}, \text{Well2}]$ — правый блок, от линии разлома
- Секция 4: $[\text{Well2}, \text{end}]$ — правый блок, далеко от разлома

**Ключевые принципы:**
1. Линия разлома — строго прямая: $x_{fault}(z) = fault\_x + z \cdot \tan(\alpha)$
2. Слои остаются горизонтальными (с естественной криволинейностью от каротажа),
   но разорваны на линии разлома: правый блок смещён на $fault\_throw$ вниз
3. Плотность опорных вершин сплайнов ≈ `VERTEX_STEP` (50 м) —
   число вершин на границу **переменное** (зависит от ширины секции на данной глубине)
4. Глубины: секции 1-2 из `orig_boundaries`, секции 3-4 из `orig_boundaries + throw`
"""

n_layers = len(formations)
n_boundaries = n_layers + 1  # 76 (поверхность + 75 нижних границ)
alpha_rad = np.radians(fault_angle)
tan_alpha = np.tan(alpha_rad)

# --- Секция 1: [0 .. Well1] — фиксированные X, одинаковые для всех границ ---
idx_well1 = int(WELL1_DISTANCE / 10)
x_sec1 = distances[0:idx_well1 + 1:sample_step]
if x_sec1[-1] != WELL1_DISTANCE:
    x_sec1 = np.append(x_sec1, WELL1_DISTANCE)

# --- Секция 4: [Well2 .. end] — фиксированные X ---
idx_well2 = int(WELL2_DISTANCE / 10)
x_sec4 = distances[idx_well2::sample_step]
if x_sec4[0] != WELL2_DISTANCE:
    x_sec4 = np.concatenate([[WELL2_DISTANCE], x_sec4])
if x_sec4[-1] != PROFILE_LENGTH:
    x_sec4 = np.append(x_sec4, PROFILE_LENGTH)

# --- Секции 2 и 3: X зависит от глубины ---
# Для каждой границы определяем X пересечения с прямой линией разлома.
# Линия разлома: x = fault_x + z·tan(α)
#
# Левый блок (секция 2): граница i на глубине z_left = orig_boundaries[i]
# → пересечение: fault_x + z_left·tan(α)
# Правый блок (секция 3): граница i на глубине z_right = orig_boundaries[i] + throw
# → пересечение: fault_x + z_right·tan(α) — дальше вправо!
#
# Между этими точками на линии разлома — разрыв (сама поверхность сброса).

def find_fault_intersection(bnd_idx, use_throw=False):
    """Итеративный поиск x-координаты пересечения границы с линией разлома."""
    if bnd_idx == 0:
        return fault_x  # поверхность z=0
    if bnd_idx == n_boundaries - 1:
        return fault_x + model_bottom_depth * tan_alpha  # дно модели
    x_iter = fault_x
    for _ in range(20):
        z_iter = np.interp(x_iter, distances, orig_boundaries[bnd_idx - 1])
        if use_throw:
            z_iter += fault_throw
        x_new = fault_x + z_iter * tan_alpha
        if abs(x_new - x_iter) < 0.01:
            break
        x_iter = x_new
    return x_iter

# fault_x для левого блока (секция 2 — правый край)
fault_x_left = np.array([find_fault_intersection(b, use_throw=False) for b in range(n_boundaries)])
# fault_x для правого блока (секция 3 — левый край)
fault_x_right = np.array([find_fault_intersection(b, use_throw=True) for b in range(n_boundaries)])

print(f"\nX на линии разлома (поверхность): left={fault_x_left[0]:.1f}, right={fault_x_right[0]:.1f}")
print(f"X на линии разлома (середина):    left={fault_x_left[n_boundaries//2]:.1f}, "
      f"right={fault_x_right[n_boundaries//2]:.1f}")
print(f"X на линии разлома (дно):         left={fault_x_left[-1]:.1f}, right={fault_x_right[-1]:.1f}")

# Для каждой границы строим X-массивы секций 2 и 3 с шагом ~VERTEX_STEP
# Число вершин переменное: больше где секция шире, меньше где уже
sec_data = [[], [], [], []]  # sec_data[sec_idx][bnd_idx] = (x_arr, z_arr)

for bnd_idx in range(n_boundaries):
    fx_left = fault_x_left[bnd_idx]    # правый край секции 2
    fx_right = fault_x_right[bnd_idx]  # левый край секции 3

    # --- Глубины ---
    if bnd_idx == 0:
        # Поверхность z=0
        z_left_fn = lambda x: np.zeros_like(x) if hasattr(x, '__len__') else 0.0
        z_right_fn = lambda x: np.zeros_like(x) if hasattr(x, '__len__') else 0.0
    elif bnd_idx == n_boundaries - 1:
        # Дно модели
        z_left_fn = lambda x: np.full_like(x, model_bottom_depth) if hasattr(x, '__len__') else model_bottom_depth
        z_right_fn = lambda x: np.full_like(x, model_bottom_depth) if hasattr(x, '__len__') else model_bottom_depth
    else:
        layer_idx = bnd_idx - 1
        orig_bnd = orig_boundaries[layer_idx]  # (1176,)
        z_left_fn = lambda x, ob=orig_bnd: np.interp(x, distances, ob)
        z_right_fn = lambda x, ob=orig_bnd: np.interp(x, distances, ob) + fault_throw

    # --- Секция 1: фиксированные X ---
    z1 = z_left_fn(x_sec1)
    sec_data[0].append((x_sec1.copy(), np.atleast_1d(z1)))

    # --- Секция 2: [Well1 .. fault_x_left], шаг ~VERTEX_STEP ---
    mask2 = (distances >= WELL1_DISTANCE) & (distances <= fx_left)
    x2_grid = distances[mask2][::sample_step]
    x2_list = list(x2_grid)
    if len(x2_list) == 0 or x2_list[0] != WELL1_DISTANCE:
        x2_list.insert(0, WELL1_DISTANCE)
    if x2_list[-1] != fx_left:
        if fx_left - x2_list[-1] < VERTEX_STEP * 0.3 and len(x2_list) > 1:
            x2_list[-1] = fx_left
        else:
            x2_list.append(fx_left)
    x2 = np.array(x2_list)
    z2 = z_left_fn(x2)
    sec_data[1].append((x2, np.atleast_1d(z2)))

    # --- Секция 3: [fault_x_right .. Well2], шаг ~VERTEX_STEP ---
    mask3 = (distances >= fx_right) & (distances <= WELL2_DISTANCE)
    x3_grid = distances[mask3][::sample_step]
    x3_list = list(x3_grid)
    if len(x3_list) == 0 or x3_list[0] != fx_right:
        if len(x3_list) > 0 and x3_list[0] - fx_right < VERTEX_STEP * 0.3:
            x3_list[0] = fx_right
        else:
            x3_list.insert(0, fx_right)
    if x3_list[-1] != WELL2_DISTANCE:
        x3_list.append(WELL2_DISTANCE)
    x3 = np.array(x3_list)
    z3 = z_right_fn(x3)
    sec_data[2].append((x3, np.atleast_1d(z3)))

    # --- Секция 4: фиксированные X ---
    z4 = z_right_fn(x_sec4)
    sec_data[3].append((x_sec4.copy(), np.atleast_1d(z4)))

# Статистика
for sec_idx in range(4):
    n_verts_list = [len(sec_data[sec_idx][b][0]) for b in range(n_boundaries)]
    sec_names = ['[0..Well1]', '[Well1..Fault]', '[Fault..Well2]', '[Well2..End]']
    print(f"  Секция {sec_idx+1} {sec_names[sec_idx]}: "
          f"вершин/границу = {min(n_verts_list)}..{max(n_verts_list)}")

""" ## Совмещение близких вершин на линии разлома ## """

"""
На линии разлома вершины левого блока (секция 2) и правого блока (секция 3)
находятся при разных X и Z, но обе — на прямой `x = fault_x + z·tan(α)`.
Если граница `i` левого блока и граница `j` правого блока оказались очень
близко по Z, мешер строит вырожденные элементы.

Решение: найти такие пары, усреднить Z и пересчитать X по прямой линии разлома.
**Инвариант:** каждая крайняя вершина на разломе ВСЕГДА лежит на прямой
`x = fault_x + z·tan(α)` с машинной точностью.
"""

MERGE_THRESHOLD = 5.0  # м — минимально допустимое расстояние между вершинами

def snap_to_fault_line(z):
    """Вычислить X на прямой линии разлома для данной глубины Z."""
    return fault_x + z * tan_alpha

# Z-координаты крайних вершин на линии разлома
z_left_edge = np.array([sec_data[1][b][1][-1] for b in range(n_boundaries)])   # правый край секции 2
z_right_edge = np.array([sec_data[2][b][1][0] for b in range(n_boundaries)])   # левый край секции 3

n_merged = 0
for i in range(len(z_left_edge)):
    for j in range(len(z_right_edge)):
        if i == j:
            continue
        dist_ij = abs(z_left_edge[i] - z_right_edge[j])
        if 0 < dist_ij < MERGE_THRESHOLD:
            avg_z = (z_left_edge[i] + z_right_edge[j]) / 2.0
            avg_x = snap_to_fault_line(avg_z)
            print(f"  Совмещение: left bnd {i} (z={z_left_edge[i]:.2f}) "
                  f"↔ right bnd {j} (z={z_right_edge[j]:.2f}) → z={avg_z:.2f}, x={avg_x:.2f}")
            # Обновляем X и Z крайней вершины — она остаётся на линии разлома
            x_l, z_l = sec_data[1][i]
            x_l[-1] = avg_x
            z_l[-1] = avg_z
            sec_data[1][i] = (x_l, z_l)

            x_r, z_r = sec_data[2][j]
            x_r[0] = avg_x
            z_r[0] = avg_z
            sec_data[2][j] = (x_r, z_r)

            z_left_edge[i] = avg_z
            z_right_edge[j] = avg_z
            n_merged += 1

print(f"\nСовмещено вершин на линии разлома: {n_merged} (порог = {MERGE_THRESHOLD} м)")

""" ## Генерация JOU-скрипта вершин ## """

"""
Генерируем все вершины для 4 секций × 76 граничных линий.
Число вершин на границу переменное (зависит от ширины секции на данной глубине).
Каждая вершина — точка (x, z, 0) в плоскости XY (Z=0 — 2D модель).

Формат Fidesys: `create vertex location X Y Z`
Здесь Y — глубина (вниз), Z = 0 (2D).
"""

total_vertices = sum(len(sec_data[s][b][0])
                     for s in range(4) for b in range(n_boundaries))
print(f"\nОбщее количество вершин: {total_vertices}")

with open(f'data/dev_3_3_model_vertex{angle_suffix}.jou', 'w') as f:
    f.write('reset\n')

    for sec_idx in range(4):
        f.write(f'# --- Section {sec_idx+1} ---\n')

        for bnd_idx in range(n_boundaries):
            x_arr, z_arr = sec_data[sec_idx][bnd_idx]
            for k in range(len(x_arr)):
                f.write(f'create vertex location {x_arr[k]:.2f} {z_arr[k]:.4f} 0\n')

print(f"Сохранено: data/dev_3_3_model_vertex{angle_suffix}.jou ({total_vertices} вершин)")

""" ## Генерация JOU-скрипта сплайнов, вертикальных рёбер и поверхностей ## """

r"""
Геометрия модели строится из кривых трёх типов:

1. **Горизонтальные сплайны** — 4 секции × 76 граничных линий = 304 кривых.
   Создаются из опорных вершин командой `create curve spline location vertex V1 to V2`.

2. **Вертикальные рёбра** на 4 простых линиях раздела секций (x=0, Well1, Well2, end) —
   по 75 прямых кривых на каждую линию = 300 кривых.
   Создаются командой `create curve vertex V1 V2`.

3. **Рёбра вдоль линии разлома** — между каждой парой соседних вершин,
   отсортированных по глубине $z$. Вершины из правого края секции 2 (левый блок)
   и левого края секции 3 (правый блок) чередуются на прямой $x = fault\_x + z \cdot \tan(\alpha)$.
   Количество рёбер > 75, так как на одну линию приходятся вершины обоих блоков.

После создания всех кривых выполняется `merge vertex all` для слияния
совпадающих вершин (на стыках секций и на линии разлома), затем `delete vertex all`.

Поверхности создаются командой `create surface curve c1 c2 c3 ...`
по замкнутому контуру ограничивающих кривых (≥ 4 кривых на поверхность).
"""

# === Маппинг vertex ID: vertex_id[sec][bnd] = [vid_0, vid_1, ..., vid_n] ===
vertex_id = []
vid = 1
for sec_idx in range(4):
    sec_vids = []
    for bnd_idx in range(n_boundaries):
        n_v = len(sec_data[sec_idx][bnd_idx][0])
        bnd_vids = list(range(vid, vid + n_v))
        vid += n_v
        sec_vids.append(bnd_vids)
    vertex_id.append(sec_vids)

with open(f'data/dev_3_3_model_spline{angle_suffix}.jou', 'w') as f:
    curve_id = 1

    # ==================================================================
    # 1. Горизонтальные сплайны (4 секции × 76 границ = 304 кривых)
    # ==================================================================
    horiz_curves = []  # horiz_curves[sec][bnd] = curve_id

    for sec_idx in range(4):
        sec_curves = []
        f.write(f'# --- Section {sec_idx+1} horizontal splines ---\n')

        for bnd_idx in range(n_boundaries):
            vids = vertex_id[sec_idx][bnd_idx]
            f.write(f'create curve spline location vertex {vids[0]} to {vids[-1]}\n')
            sec_curves.append(curve_id)
            curve_id += 1

        horiz_curves.append(sec_curves)

    n_horiz = curve_id - 1

    # ==================================================================
    # 2. Вертикальные рёбра на 4 простых линиях раздела (4 × 75 = 300)
    # ==================================================================
    # x=0: sec1 leftmost (левый край модели)
    f.write('\n# --- Vertical edges: x=0 ---\n')
    vert_x0 = []
    for layer in range(n_layers):
        v1 = vertex_id[0][layer][0]
        v2 = vertex_id[0][layer + 1][0]
        f.write(f'create curve vertex {v1} {v2}\n')
        vert_x0.append(curve_id)
        curve_id += 1

    # x=Well1: sec1 rightmost (после merge vertex соединится с sec2 leftmost)
    f.write('\n# --- Vertical edges: x=Well1 ---\n')
    vert_well1 = []
    for layer in range(n_layers):
        v1 = vertex_id[0][layer][-1]
        v2 = vertex_id[0][layer + 1][-1]
        f.write(f'create curve vertex {v1} {v2}\n')
        vert_well1.append(curve_id)
        curve_id += 1

    # x=Well2: sec4 leftmost (после merge vertex соединится с sec3 rightmost)
    f.write('\n# --- Vertical edges: x=Well2 ---\n')
    vert_well2 = []
    for layer in range(n_layers):
        v1 = vertex_id[3][layer][0]
        v2 = vertex_id[3][layer + 1][0]
        f.write(f'create curve vertex {v1} {v2}\n')
        vert_well2.append(curve_id)
        curve_id += 1

    # x=end: sec4 rightmost (правый край модели)
    f.write('\n# --- Vertical edges: x=end ---\n')
    vert_end = []
    for layer in range(n_layers):
        v1 = vertex_id[3][layer][-1]
        v2 = vertex_id[3][layer + 1][-1]
        f.write(f'create curve vertex {v1} {v2}\n')
        vert_end.append(curve_id)
        curve_id += 1

    n_vert_simple = curve_id - 1 - n_horiz

    # ==================================================================
    # 3. Рёбра вдоль линии разлома
    # ==================================================================
    f.write('\n# --- Fault line edges ---\n')

    # Собираем ВСЕ вершины на линии разлома (sec2 правый край + sec3 левый край)
    fault_verts = []  # (z, vertex_fidesys_id, side, bnd_idx)
    for bnd_idx in range(n_boundaries):
        vid_l = vertex_id[1][bnd_idx][-1]
        z_l = sec_data[1][bnd_idx][1][-1]
        fault_verts.append((z_l, vid_l, 'L', bnd_idx))

        vid_r = vertex_id[2][bnd_idx][0]
        z_r = sec_data[2][bnd_idx][1][0]
        fault_verts.append((z_r, vid_r, 'R', bnd_idx))

    # Сортируем по глубине Z
    fault_verts.sort(key=lambda x: x[0])

    # Группируем вершины с одинаковой Z (в пределах ε — совпавшие после слияния)
    eps = 1e-4
    fault_groups = []  # каждая группа: [(z, vid, side, bnd_idx), ...]
    current_group = [fault_verts[0]]
    for fv in fault_verts[1:]:
        if abs(fv[0] - current_group[0][0]) < eps:
            current_group.append(fv)
        else:
            fault_groups.append(current_group)
            current_group = [fv]
    fault_groups.append(current_group)

    # Создаём ребро между каждой парой соседних групп
    fault_edge_curves = []  # fault_edge_curves[i] = curve_id ребра group[i] → group[i+1]
    for g_idx in range(len(fault_groups) - 1):
        v1 = fault_groups[g_idx][0][1]      # первая вершина текущей группы
        v2 = fault_groups[g_idx + 1][0][1]  # первая вершина следующей группы
        f.write(f'create curve vertex {v1} {v2}\n')
        fault_edge_curves.append(curve_id)
        curve_id += 1

    # Маппинг: (side, bnd_idx) → индекс группы на линии разлома
    fault_group_map = {}
    for g_idx, group in enumerate(fault_groups):
        for (z, v, side, bnd_idx) in group:
            fault_group_map[(side, bnd_idx)] = g_idx

    n_fault_edges = len(fault_edge_curves)
    n_total_curves = curve_id - 1
    print(f"  Горизонтальные сплайны: {n_horiz}")
    print(f"  Вертикальные рёбра (4 линии): {n_vert_simple}")
    print(f"  Линия разлома: {len(fault_groups)} позиций, {n_fault_edges} рёбер")
    print(f"  Всего кривых: {n_total_curves}")

    # ==================================================================
    # 4. Слияние совпадающих вершин + удаление свободных
    # ==================================================================
    f.write('\n# Merge coincident vertices, delete free\n')
    f.write('merge vertex all\n')
    f.write('delete vertex all\n')

    # ==================================================================
    # 5. Создание поверхностей по ограничивающим кривым
    # ==================================================================
    f.write('\n# --- Surfaces (bounded by curves) ---\n')

    for sec_idx in range(4):
        f.write(f'# Section {sec_idx+1}\n')

        for layer_idx in range(n_layers):
            c_top = horiz_curves[sec_idx][layer_idx]
            c_bot = horiz_curves[sec_idx][layer_idx + 1]

            if sec_idx == 0:
                # Sec1: top, right(Well1), bottom, left(x=0)
                curves = [c_top, vert_well1[layer_idx], c_bot, vert_x0[layer_idx]]

            elif sec_idx == 1:
                # Sec2: top, fault edges..., bottom, left(Well1)
                g_start = fault_group_map[('L', layer_idx)]
                g_end = fault_group_map[('L', layer_idx + 1)]
                right_edges = [fault_edge_curves[g] for g in range(g_start, g_end)]
                curves = [c_top] + right_edges + [c_bot, vert_well1[layer_idx]]

            elif sec_idx == 2:
                # Sec3: top, right(Well2), bottom, fault edges...
                g_start = fault_group_map[('R', layer_idx)]
                g_end = fault_group_map[('R', layer_idx + 1)]
                left_edges = [fault_edge_curves[g] for g in range(g_start, g_end)]
                curves = [c_top, vert_well2[layer_idx], c_bot] + left_edges

            elif sec_idx == 3:
                # Sec4: top, right(end), bottom, left(Well2)
                curves = [c_top, vert_end[layer_idx], c_bot, vert_well2[layer_idx]]

            curves_str = ' '.join(str(c) for c in curves)
            f.write(f'create surface curve {curves_str}\n')

n_surfaces = 4 * n_layers  # 300
print(f"Сохранено: data/dev_3_3_model_spline{angle_suffix}.jou")
print(f"  Кривых: {n_total_curves}")
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

# Длины секций (используем поверхностную границу как представительную)
sec_lengths = [sec_data[s][0][0][-1] - sec_data[s][0][0][0] for s in range(4)]
print(f"\nДлины секций (поверхность): {[f'{l:.0f}' for l in sec_lengths]} м")

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
with open(f'data/dev_3_3_model_mesh{angle_suffix}.jou', 'w') as f:
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

print(f"\nСохранено: data/dev_3_3_model_mesh{angle_suffix}.jou")
print(f"  Поверхностей: {4 * n_layers}")


""" ## Генерация JOU-скрипта материалов и блоков ## """

"""
Создаём 75 материалов-заглушек (по одному на слой) и 75 блоков. Каждый блок
объединяет 4 поверхности одного слоя из всех секций.

Нумерация поверхностей: секция 1 → 1..75, секция 2 → 76..150,
секция 3 → 151..225, секция 4 → 226..300.

Блок i содержит поверхности: i, 75+i, 150+i, 225+i.
"""

with open(f'data/dev_3_3_model_materials{angle_suffix}.jou', 'w') as f:
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
        f.write(f'block {mat_id} material {mat_id} cs 1 category plane order 2\n')
        f.write('\n')

print(f"Сохранено: data/dev_3_3_model_materials{angle_suffix}.jou")
print(f"  Материалов: {n_layers}")
print(f"  Блоков: {n_layers} (по 4 поверхности в каждом)")

""" ## Визуализация: структура секций ## """

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_title('Разбиение модели на 4 секции', fontsize=14)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Sec 1: [0..Well1]', 'Sec 2: [Well1..Fault]',
          'Sec 3: [Fault..Well2]', 'Sec 4: [Well2..End]']

for sec_idx in range(4):
    # Рисуем несколько границ для иллюстрации
    for bnd_idx in range(0, n_boundaries, 5):
        x_arr, z_arr = sec_data[sec_idx][bnd_idx]
        ax.plot(x_arr, z_arr, color=colors[sec_idx],
                linewidth=0.5, alpha=0.7)
    # Одну линию с лейблом
    mid = n_boundaries // 2
    x_arr, z_arr = sec_data[sec_idx][mid]
    ax.plot(x_arr, z_arr, color=colors[sec_idx],
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
plt.savefig(f'img/dev_3_3_sections_overview{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_3_sections_overview{angle_suffix}.png")


""" ## Визуализация: узлы (вершины) по секциям ## """

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_title(f'Вершины модели (angle={fault_angle}°, throw={fault_throw:.0f} м)', fontsize=14)

for sec_idx in range(4):
    # Все вершины секции
    all_x = np.concatenate([sec_data[sec_idx][b][0] for b in range(n_boundaries)])
    all_z = np.concatenate([sec_data[sec_idx][b][1] for b in range(n_boundaries)])
    ax.scatter(all_x, all_z, s=0.3, color=colors[sec_idx], alpha=0.6, label=labels[sec_idx])

# Линия разлома
ax.plot(x_line, z_line, 'k--', linewidth=1.5, label='Линия разлома')

ax.set_xlabel('X (м)')
ax.set_ylabel('Глубина (м)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', markerscale=10)
ax.set_xlim(0, PROFILE_LENGTH)

plt.tight_layout()
plt.savefig(f'img/dev_3_3_vertices{angle_suffix}.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Сохранено: img/dev_3_3_vertices{angle_suffix}.png")


""" ## Сводка сгенерированных скриптов ## """

"""
Для построения модели в Fidesys необходимо последовательно выполнить (суффикс зависит от угла):

```
playback 'data/dev_3_3_model_vertex{angle_suffix}.jou'
playback 'data/dev_3_3_model_spline{angle_suffix}.jou'
playback 'data/dev_3_3_model_mesh{angle_suffix}.jou'
playback 'data/dev_3_3_model_materials{angle_suffix}.jou'
```

После этого — экспорт FC-модели (заглушка с материалами):
```
export fc 'data/dev_3_3_model_stub{angle_suffix}.fc'
```
"""

print("\n=== Сводка ===")
print(f"Файлы скриптов (суффикс {angle_suffix}):")
print(f"  data/dev_3_3_model_vertex{angle_suffix}.jou     — {total_vertices} вершин")
print(f"  data/dev_3_3_model_spline{angle_suffix}.jou     — {curve_id-1} кривых, {n_surfaces} поверхностей")
print(f"  data/dev_3_3_model_mesh{angle_suffix}.jou       — параметры сетки")
print(f"  data/dev_3_3_model_materials{angle_suffix}.jou  — {n_layers} материалов + {n_layers} блоков")
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