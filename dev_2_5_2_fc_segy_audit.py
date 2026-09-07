""" # MILEN SEM 2D. Часть II. Проверка материала FC и SEG-Y # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 2: таблицы FC, растры и загрузчик SEMGPU ## """

"""
Сравним исходные таблицы в одинаковых координатах, затем поля в общей
геометрии FC, учитывая принадлежность элементов материалам. Отдельно
воспроизведём фактический загрузчик указанного расчётного скрипта.
Данные и исходный код других проектов не изменяем. Решатель не запускаем.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial import KDTree
import segyio


""" ## 1. Контекст точного расчётного скрипта и входные файлы ## """

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
runtime_root = Path("/home/antonov/Base/Research/TrySEMGPU")
prefix = "dev_2_5_2_fc_segy_audit"
fc_path = root / "data/dev_2_2_milen2Do5pore_full.fc"
script_path = runtime_root / "research/seismic/test_exp_seismic_sweep.py"
for relative_path in ("semgpu/src", "fc-model/src", "fc-calc/src"):
    sys.path.insert(0, str(runtime_root / relative_path))

from fc_model import FCModel
import fc_calc.builders.seismic as builders

spec = importlib.util.spec_from_file_location("audited_seismic_sweep", script_path)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
runtime_fc_path = Path(sweep.model_cfg["fc"]["path"])
assert not sweep.model_cfg["material"]["use_homogeneous"]


def file_hash(path):
    """Вычисляет SHA-256 файла потоковым чтением без изменения содержимого."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


source_paths = [fc_path, runtime_fc_path, script_path, Path(builders.__file__),
                runtime_root / "fc-model/src/fc_model/fc_data.py",
                root / "dev_1_7_material_euler.py", root / "dev_2_1_segy_material.py"]
manifest = {str(path): file_hash(path) for path in source_paths}
assert manifest[str(fc_path)] == manifest[str(runtime_fc_path)]
print("FC files: identical SHA-256", manifest[str(fc_path)], flush=True)
fc = FCModel.load(str(fc_path))
nodes = fc.mesh.nodes_xyz[:, :2]
domain_min = nodes.min(axis=0)
domain_max = nodes.max(axis=0)
print("FC domain:", domain_min, domain_max, flush=True)
assert np.allclose(domain_min, [0, 0])

rasters = []
aligned_rasters = []
raster_metadata = {}
for name in ("Vp", "Vs", "Density"):
    pair = []
    for suffix in ("", "_node_aligned"):
        path = root / (f"data/dev_2_5_legacy_{name}_model_node_aligned.sgy" if suffix else f"data/dev_2_1_{name}_model.sgy")
        manifest[str(path)] = file_hash(path)
        with segyio.open(str(path), "r", ignore_geometry=True) as model:
            values = segyio.tools.collect(model.trace[:]).astype(float)
            raster_metadata[path.name] = dict(
                shape=list(values.shape), samples_first=float(model.samples[0]),
                samples_last=float(model.samples[-1]),
                x_first=int(model.header[0][segyio.TraceField.CDP_X]),
                x_last=int(model.header[model.tracecount - 1][segyio.TraceField.CDP_X]))
            assert np.allclose(np.diff(model.samples), 5)
        pair.append(values)
    raster_metadata[name + "_aligned_is_edge_padding"] = bool(
        np.array_equal(np.pad(pair[0], ((0, 1), (0, 1)), mode="edge"), pair[1]))
    rasters.append(pair[0])
    aligned_rasters.append(pair[1])
sgy = np.stack(rasters, axis=-1)
aligned = np.stack(aligned_rasters, axis=-1)
sgy[:, :, 2] *= 1000.0
aligned[:, :, 2] *= 1000.0
nx, nz = sgy.shape[:2]
x_centers = 2.5 + 5 * np.arange(nx)
z_centers = 2.5 + 5 * np.arange(nz)
common_z = z_centers < domain_max[1]


""" ## 2. Декодируем таблицы FC через fc-model ## """

def speeds(young, poisson, density):
    """Возвращает Vp, Vs (м/с), rho (кг/м³) для плоской деформации."""
    return np.stack((np.sqrt(young / density * (1 - poisson)
                             / ((1 + poisson) * (1 - 2 * poisson))),
                     np.sqrt(young / (2 * density * (1 + poisson))), density), axis=-1)


tables = {}
fc_table_grid = np.full_like(sgy, np.nan)
table_layer_grid = np.zeros((nx, nz), dtype=np.int16)
first_values = {}
material_audit = []
for mid, mat in fc.materials.items():
    properties = {prop.name: prop.data for groups in mat.properties.values()
                  for group in groups for prop in group}
    data = [properties[name] for name in ("YOUNG_MODULE", "POISSON_RATIO", "DENSITY")]
    assert all(item.type == "TABLE" for item in data)
    coordinates = []
    for item in data:
        columns = {column.type: column.value.data for column in item.table}
        coordinates.append(np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"])))
    assert all(np.array_equal(coordinates[0], xy) for xy in coordinates[1:])
    xy = coordinates[0]
    primitive = np.column_stack([item.value.data for item in data])
    assert np.all((primitive[:, 1] >= 0) & (primitive[:, 1] <= 0.49))
    values = speeds(*primitive.T)
    indices = np.rint((xy - 2.5) / 5).astype(int)
    assert np.allclose(xy, indices * 5 + 2.5)
    assert np.all(table_layer_grid[indices[:, 0], indices[:, 1]] == 0)
    fc_table_grid[indices[:, 0], indices[:, 1]] = values
    table_layer_grid[indices[:, 0], indices[:, 1]] = mid
    first_values[mid] = values[0]
    tables[mid] = dict(xy=xy, primitive=primitive, values=values, tree=KDTree(xy))
    runtime_values = np.column_stack(builders._evaluate_material_triplet(
        mat, xy[:, 0], xy[:, 1]))  # фактический порядок: rho, E, nu
    evaluated = speeds(runtime_values[:, 1], runtime_values[:, 2], runtime_values[:, 0])
    assert np.allclose(evaluated, values[0], rtol=1e-13)
    material_audit.append(dict(id=int(mid), name=mat.name, count=len(xy),
                               fc_dependency_type=str(data[0].type),
                               runtime_kind=builders._extract_material_field(
                                   mat, "YOUNG_MODULE", default=1e10).kind,
                               first_xy=xy[0].tolist(), first_values=values[0].tolist()))
assert np.all(np.isfinite(fc_table_grid))
print("All 75 runtime materials collapse to their first table row", flush=True)


""" ## 3. Геометрия: сравнение в одних точках внутри элементов FC ## """

"""
Используем 5×5 точек Гаусса внутри каждого QUAD8: точки принадлежат
конкретному материалу FC, общие интерфейсные узлы не усредняются.
Вес точки — вес квадратуры × |якобиан|, поэтому статистика приближённо
взвешена по площади, а не по густоте сетки. Карты показывают среднее
различие по площади элемента. Таблицы сравниваются отдельно без усреднения.
"""

elements = list(sorted(fc.mesh.elements["QUAD8"].values(), key=lambda element: element.id))
id_to_index = {int(nid): i for i, nid in enumerate(fc.mesh.nodes_ids)}
connectivity = np.array([[id_to_index[int(nid)] for nid in element.nodes]
                         for element in elements], dtype=int)
element_material = np.array([fc.blocks[int(element.block)].material_id for element in elements])
element_nodes = nodes[connectivity]
converter = builders.FCtoSEMGPUConverter(fc.mesh.nodes_xyz, connectivity, "QUAD8")
element_centers = np.einsum("i,eij->ej", converter._shape_quad8(0, 0), element_nodes)
polygons = element_nodes[:, [0, 4, 1, 5, 2, 6, 3, 7]]


def sample_raster(points, node_linear=False):
    """Читает физические ячейки исходного растра или билинейный узловой растр.

    Второй вариант — явная проверка чувствительности к трактовке отсчётов.
    Он не утверждает, какой именно интерполятор применяет Tesseral.
    """
    ix = np.clip(np.floor(points[..., 0] / 5).astype(int), 0, nx - 1)
    iz = np.clip(np.floor(points[..., 1] / 5).astype(int), 0, nz - 1)
    if not node_linear:
        return sgy[ix, iz]
    fx = np.clip(points[..., 0] / 5 - ix, 0, 1)[..., None]
    fz = np.clip(points[..., 1] / 5 - iz, 0, 1)[..., None]
    return ((1 - fx) * (1 - fz) * aligned[ix, iz]
            + fx * (1 - fz) * aligned[ix + 1, iz]
            + (1 - fx) * fz * aligned[ix, iz + 1]
            + fx * fz * aligned[ix + 1, iz + 1])


def weighted_stats(difference, weights):
    """Среднее, MAE, RMSE, квантили |ошибки| и доли площади; ошибки в %."""
    values = np.asarray(difference).ravel()
    weights = np.broadcast_to(weights, difference.shape).ravel()
    valid = np.isfinite(values) & (weights > 0)
    values, weights = values[valid], weights[valid]
    absolute = np.abs(values)
    order = np.argsort(absolute)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantiles = np.interp([0.5, 0.95, 0.99], cumulative, absolute[order])
    return dict(mean=float(np.average(values, weights=weights)),
                mae=float(np.average(absolute, weights=weights)),
                rmse=float(np.sqrt(np.average(values ** 2, weights=weights))),
                p50=float(quantiles[0]), p95=float(quantiles[1]), p99=float(quantiles[2]),
                max_abs=float(absolute.max()),
                above_1_pct=float(100 * weights[absolute > 1].sum() / weights.sum()),
                above_5_pct=float(100 * weights[absolute > 5].sum() / weights.sum()),
                above_10_pct=float(100 * weights[absolute > 10].sum() / weights.sum()))


def compare_quadrature(order):
    """Сравнивает поля на квадратуре реальной QUAD8-геометрии, без изменения FC."""
    gauss, gauss_weights = np.polynomial.legendre.leggauss(order)
    points_list, weights_list = [], []
    for j, s in enumerate(gauss):
        for i, r in enumerate(gauss):
            shape = converter._shape_quad8(r, s)
            dr = converter._shape_quad8(r + 1e-30j, s).imag / 1e-30
            ds = converter._shape_quad8(r, s + 1e-30j).imag / 1e-30
            a = np.einsum("i,eij->ej", dr, element_nodes)
            b = np.einsum("i,eij->ej", ds, element_nodes)
            jacobian = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            points_list.append(np.einsum("i,eij->ej", shape, element_nodes))
            weights_list.append(np.abs(jacobian) * gauss_weights[i] * gauss_weights[j])
    points = np.stack(points_list, axis=1)
    weights = np.stack(weights_list, axis=1)
    raw = sample_raster(points)
    node_values = sample_raster(points, node_linear=True)
    expected = np.empty_like(raw)
    runtime = np.empty_like(raw)
    nearest_distance = np.empty(weights.shape)
    for mid, table in tables.items():
        mask = element_material == mid
        distance, index = table["tree"].query(points[mask].reshape(-1, 2))
        expected[mask] = table["values"][index].reshape((-1, order * order, 3))
        runtime[mask] = first_values[mid]
        nearest_distance[mask] = distance.reshape((-1, order * order))
    ix = np.clip(np.floor(points[..., 0] / 5).astype(int), 0, nx - 1)
    iz = np.clip(np.floor(points[..., 1] / 5).astype(int), 0, nz - 1)
    same_layer = table_layer_grid[ix, iz] == element_material[:, None]
    differences = {
        "sgy_cell_minus_fc_nn": 100 * (raw - expected) / expected,
        "sgy_node_linear_minus_fc_nn": 100 * (node_values - expected) / expected,
        "runtime_minus_fc_nn": 100 * (runtime - expected) / expected,
        "runtime_minus_sgy_cell": 100 * (runtime - raw) / raw,
    }
    stats, maps = {}, {}
    for name, difference in differences.items():
        stats[name] = [weighted_stats(difference[..., k], weights) for k in range(3)]
        stats[name + "_same_layer"] = [weighted_stats(difference[..., k], weights * same_layer)
                                        for k in range(3)]
        maps[name] = np.sum(difference * weights[..., None], axis=1) / weights.sum(axis=1)[:, None]
    stats["layer_mismatch_area_pct"] = float(100 * weights[~same_layer].sum() / weights.sum())
    stats["area_m2"] = float(weights.sum())
    stats["sample_count"] = int(weights.size)
    stats["nearest_table_distance_max_m"] = float(nearest_distance.max())
    return stats, maps


quadrature_stats, element_maps = compare_quadrature(5)
print("5x5 quadrature finished:", quadrature_stats, flush=True)
check_stats, _ = compare_quadrature(3)
print("3x3 quadrature convergence check finished", flush=True)
assert np.isclose(quadrature_stats["area_m2"], np.prod(domain_max - domain_min), rtol=1e-8)
table_difference = 100 * (sgy - fc_table_grid) / fc_table_grid
table_stats = [weighted_stats(table_difference[:, common_z, k], np.ones((nx, common_z.sum())))
               for k in range(3)]
per_layer = []
for mid, table in tables.items():
    mask = (table_layer_grid == mid) & common_z[None, :]
    per_layer.append(dict(id=int(mid), name=fc.materials[mid].name, count=int(mask.sum()),
                          stats=[weighted_stats(table_difference[..., k][mask], np.ones(mask.sum()))
                                 for k in range(3)]))


""" ## 4. Точный вертикальный разрез геометрии FC при x=6000 м ## """

source_x = 6000.0
layer_bounds = {}
edges = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
for mid in fc.materials:
    mask = ((element_material == mid)
            & (element_nodes[:, :, 0].min(axis=1) <= source_x)
            & (element_nodes[:, :, 0].max(axis=1) >= source_x))
    intersections = []
    for cell in element_nodes[mask]:
        for start, middle, end in edges:
            p0, pm, p1 = cell[[start, middle, end]]
            a = 0.5 * (p0 + p1) - pm
            b = 0.5 * (p1 - p0)
            c = pm - [source_x, 0]
            if abs(a[0]) < 1e-10:
                roots = [-c[0] / b[0]] if abs(b[0]) > 1e-10 else []
                if abs(b[0]) <= 1e-10 and abs(c[0]) < 1e-7:
                    intersections.extend([p0[1], pm[1], p1[1]])
            else:
                roots = np.roots([a[0], b[0], c[0]])
            for t in roots:
                if abs(np.imag(t)) < 1e-8 and -1 - 1e-8 <= np.real(t) <= 1 + 1e-8:
                    t = float(np.real(t))
                    intersections.append(float(a[1] * t * t + b[1] * t + pm[1]))
    layer_bounds[mid] = [min(intersections), max(intersections)]
boundaries = np.array([layer_bounds[mid][1] for mid in fc.materials])
assert np.allclose([layer_bounds[mid][0] for mid in range(2, 76)], boundaries[:-1], atol=1e-5)
depth_edges = np.unique(np.r_[np.arange(0, boundaries[-1], 0.5), boundaries])
depth = (depth_edges[:-1] + depth_edges[1:]) / 2
profile_mid = np.searchsorted(boundaries, depth) + 1
assert np.all((profile_mid >= 1) & (profile_mid <= len(fc.materials)))
assert depth_edges[-1] <= domain_max[1] + 1e-7
profile_points = np.column_stack((np.full_like(depth, source_x), depth))
profile_sgy = sample_raster(profile_points)
profile_fc = np.full_like(profile_sgy, np.nan)
profile_runtime = np.full_like(profile_sgy, np.nan)
for mid, table in tables.items():
    mask = profile_mid == mid
    _, index = table["tree"].query(profile_points[mask])
    profile_fc[mask] = table["values"][index]
    profile_runtime[mask] = first_values[mid]
assert np.all(np.isfinite(profile_fc)) and np.all(np.isfinite(profile_runtime))
source_peak = 1 / sweep.model_cfg["source"]["f0"]
source_depth = sweep.model_cfg["source"]["y_depth"]
profile_times = {}
for name, values in (("sgy", profile_sgy), ("fc_nn", profile_fc), ("runtime", profile_runtime)):
    one_way = np.r_[0, np.cumsum(np.diff(depth_edges) / values[:, 0])]
    profile_times[name] = source_peak + 2 * one_way - np.interp(source_depth, depth_edges, one_way)
travel_times = {name: {"boundary_73": float(np.interp(boundaries[72], depth_edges, t)),
                       "boundary_74": float(np.interp(boundaries[73], depth_edges, t)),
                       "depth_2495": float(np.interp(2495, depth_edges, t)),
                       "bottom": float(t[-1])} for name, t in profile_times.items()}
print("Travel times:", travel_times, flush=True)


""" ## 5. Проверяем полный загрузчик GLL порядка 7, без решения волновой задачи ## """

mesh, material = sweep.load_mesh_and_material(sweep.model_cfg)
en_per_element = (sweep.model_cfg["fc"]["order"] + 1) ** 2
expected_primitive = np.array([tables[mid]["primitive"][0][[2, 0, 1]] for mid in element_material])
material_values = material.values.reshape((-1, en_per_element, 3))
max_loader_difference = 0.0
for start in range(0, len(elements), 2000):
    difference = material_values[start:start + 2000] - expected_primitive[start:start + 2000, None, :]
    max_loader_difference = max(max_loader_difference, float(np.abs(difference).max()))
assert max_loader_difference == 0
loader_proof = dict(element_count=len(elements), gll_nodes=int(len(mesh.nodes_coords)),
                    element_local_nodes=int(material.values.shape[0]),
                    max_primitive_difference_from_first_row=max_loader_difference,
                    material_names=list(material.names), order=sweep.model_cfg["fc"]["order"])
print("Full loader proof:", loader_proof, flush=True)


""" ## 6. Карты попарных разностей и разрез при x=6000 м ## """

labels = ["Vp", "Vs", "Плотность"]
for name, title in (
    ("sgy_cell_minus_fc_nn", "SEG-Y (физические ячейки) − FC (таблицы, ближайшая точка слоя)"),
    ("sgy_node_linear_minus_fc_nn", "SEG-Y node_aligned (билинейно) − FC (ближайшая точка слоя)"),
    ("runtime_minus_fc_nn", "Фактический загрузчик SEMGPU − FC (таблицы, ближайшая точка слоя)"),
    ("runtime_minus_sgy_cell", "Фактический загрузчик SEMGPU − SEG-Y (физические ячейки)"),
):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), layout="constrained", sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        values = element_maps[name][:, k]
        limit = max(float(np.percentile(abs(values), 99)), 0.01)
        collection = PolyCollection(polygons, array=values, cmap="RdBu_r", clim=(-limit, limit),
                                    edgecolors="none", rasterized=True)
        ax.add_collection(collection)
        ax.set(xlim=(0, domain_max[0]), ylim=(domain_max[1], 0), ylabel="Глубина, м", title=labels[k])
        stats = quadrature_stats[name][k]
        ax.text(0.99, 0.05, f"MAE {stats['mae']:.2f}%; P95 |δ| {stats['p95']:.2f}%; max |δ| {stats['max_abs']:.2f}%",
                transform=ax.transAxes, ha="right", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        fig.colorbar(collection, ax=ax, label="δ, %; знаменатель — второе поле", extend="both")
    axes[-1].set_xlabel("x, м")
    fig.suptitle(title + "\nОбласть FC: 0–2650 м. Карта: среднее по элементу; статистика: квадратура 5×5, веса площади.", fontsize=12)
    fig.savefig(root / f"img/{prefix}_{name}.png", dpi=150)
    plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(15, 10), layout="constrained", sharex=True, sharey=True)
for k, ax in enumerate(axes):
    values = table_difference[:, common_z, k].T
    limit = max(float(np.percentile(abs(values), 99)), 0.01)
    im = ax.imshow(values, extent=(0, nx * 5, domain_max[1], 0), aspect="auto",
                   cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set(ylabel="Глубина, м", title=labels[k])
    fig.colorbar(im, ax=ax, label="100 × (SEG-Y − FC) / FC, %", extend="both")
axes[-1].set_xlabel("x, м")
fig.suptitle("Исходные значения SEG-Y − таблицы FC в точно одинаковых координатах\n"
             "Центры 5×5 м; общая область 0–2650 м. Без интерполяции и без влияния сетки элементов.", fontsize=12)
fig.savefig(root / f"img/{prefix}_table_pairs.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(16, 8), layout="constrained")
colors = {"sgy": "tab:blue", "fc_nn": "tab:orange", "runtime": "tab:red"}
names = {"sgy": "SEG-Y, ячейки", "fc_nn": "FC, таблицы NN", "runtime": "Загрузчик SEMGPU"}
for name, values in (("sgy", profile_sgy), ("fc_nn", profile_fc), ("runtime", profile_runtime)):
    axes[0].plot(values[:, 0] / 1000, depth, color=colors[name], label=names[name])
    axes[1].plot(values[:, 0] * values[:, 2] / 1e6, depth, color=colors[name])
    axes[2].plot(profile_times[name], depth_edges, color=colors[name])
axes[0].set(xlabel="Vp, км/с", ylabel="Глубина, м", title="Скорость P")
axes[0].set_xlim(2.6, 5.2)
axes[0].legend(fontsize=8)
axes[1].set(xlabel="ρVp, МПа·с/м", title="P-импеданс")
axes[1].set_xlim(5.5, 13)
axes[2].set(xlabel="Время PP с учётом источника, с", title="Время → глубина", xlim=(1.9, 2.3))
for ax in axes[:3]:
    ax.set_ylim(domain_max[1], 2100)
    for b in boundaries[72:74]:
        ax.axhline(b, color="0.5", ls=":", lw=0.8)
    ax.grid(alpha=0.2)
with np.load(root / "data/dev_2_3/x6000_data.npz") as shot:
    receiver = int(np.argmin(abs(shot["sensor_x"] - source_x)))
    axes[3].plot(shot["seismo_vy"][:, receiver], shot["seismo_times"], color="0.2")
for name, t in travel_times.items():
    axes[3].axhspan(t["boundary_73"], t["boundary_74"], color=colors[name], alpha=0.25, label=names[name]+": 73/74, 74/75")
axes[3].set(ylim=(2.24, 2.04), xlim=(-0.035, 0.035), xlabel="Vy, единицы NPZ", ylabel="Время, с", title="Наблюдаемый пакет")
axes[3].legend(fontsize=8, loc="lower left")
fig.suptitle("x=6000 м: загрузчик меняет глубинно-временную привязку отражений\n"
             "Источник: 5 м; Рикер 30 Гц, максимум 0.03333 с. Графики ограничены дном FC, 2650 м.", fontsize=13)
fig.savefig(root / f"img/{prefix}_x6000.png", dpi=170)
plt.close(fig)


""" ## 7. Сохранение численных результатов и выводы ## """

report = dict(manifest=manifest, fc_domain_min=domain_min.tolist(), fc_domain_max=domain_max.tolist(),
              raster_metadata=raster_metadata, table_stats_common=table_stats, per_layer=per_layer,
              quadrature_5=quadrature_stats, quadrature_3=check_stats, materials=material_audit,
              loader_proof=loader_proof, source=sweep.model_cfg["source"],
              fc_boundaries_x6000=boundaries.tolist(), travel_times_x6000=travel_times)
(root / f"data/{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
np.savez_compressed(root / f"data/{prefix}.npz", x_centers=x_centers, z_centers=z_centers,
                    sgy=sgy, fc_table_grid=fc_table_grid, table_layer_grid=table_layer_grid,
                    element_centers=element_centers, element_material=element_material,
                    **element_maps, profile_depth=depth, profile_depth_edges=depth_edges,
                    profile_sgy=profile_sgy, profile_fc_nn=profile_fc, profile_runtime=profile_runtime,
                    time_sgy=profile_times["sgy"], time_fc_nn=profile_times["fc_nn"],
                    time_runtime=profile_times["runtime"], fc_boundaries_x6000=boundaries)

"""
Проверены идентичность FC-файлов, геометрические области, значения всех
пространственных таблиц и оба комплекта SEG-Y. Попарные разности рассчитаны
в одинаковых координатах и на квадратуре внутри FC-элементов. Полный загрузчик
подтвердил потерю табличной зависимости: TABLE ошибочно обрабатывается как
константа из первой строки. Вычислена новая привязка отражений при x=6000 м.
Численные результаты сохранены в data/, графики — в img/. Исправление
production-загрузчика в данном исследовании не выполнялось.
"""
