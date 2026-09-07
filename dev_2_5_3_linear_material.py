""" # MILEN SEM 2D. Часть II. Материал при линейной интерполяции # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 3: послойная триангуляция таблиц FC ## """

"""
Проверим гипотетическое исправление загрузчика: линейно интерполируем
E, nu, rho по триангуляции Делоне табличных точек каждого материала.
Затем вычислим Vp, Vs, rho и Zp=rho*Vp. Сравним с двумя трактовками SEG-Y,
ближайшим соседом и ошибочным выбором первой строки. Геометрия и файлы
материала неизменны. Вне выпуклой оболочки используем ближайшего соседа
своего слоя и отдельно измеряем долю таких точек. Это явно заданная модель
интерполяции, а не доказательство всех внутренних правил Fidesys.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, KDTree

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(root / "fc-model/src"))
from fc_model import FCModel


""" ## 1. Повторно используем проверенные входы аудита II.5.2 ## """

prefix = "dev_2_5_3_linear_material"
fc_path = root / "data/dev_2_2_milen2Do5pore_full.fc"
previous_report = json.loads((root / "data/dev_2_5_2_fc_segy_audit.json").read_text())
with fc_path.open("rb") as stream:
    fc_hash = hashlib.file_digest(stream, "sha256").hexdigest()
assert fc_hash == next(value for path, value in previous_report["manifest"].items()
                       if path.endswith("MilenSem2D/data/dev_2_2_milen2Do5pore_full.fc"))
input_hashes = {str(fc_path): fc_hash}
for path, expected in previous_report["manifest"].items():
    if path.endswith(".sgy"):
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, f"SEG-Y изменён после исходного аудита: {path}"
        input_hashes[path] = actual
for name in ("dev_2_5_2_fc_segy_audit.npz", "dev_2_5_2_fc_segy_audit.json"):
    path = root / "data" / name
    with path.open("rb") as stream:
        input_hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
with np.load(root / "data/dev_2_5_2_fc_segy_audit.npz") as previous:
    sgy = previous["sgy"]
    table_layers = previous["table_layer_grid"]
    depth = previous["profile_depth"]
    depth_edges = previous["profile_depth_edges"]
    boundaries = previous["fc_boundaries_x6000"]
    profile_sgy = previous["profile_sgy"]
    profile_nn = previous["profile_fc_nn"]
    profile_bad = previous["profile_runtime"]
aligned = np.pad(sgy, ((0, 1), (0, 1), (0, 0)), mode="edge")
nx, nz = sgy.shape[:2]
fc = FCModel.load(str(fc_path))
nodes = fc.mesh.nodes_xyz[:, :2]
domain_max = np.round(nodes.max(axis=0), 6)
elements = sorted(fc.mesh.elements["QUAD8"].values(), key=lambda element: element.id)
node_indices = {int(nid): i for i, nid in enumerate(fc.mesh.nodes_ids)}
connectivity = np.array([[node_indices[int(nid)] for nid in element.nodes] for element in elements])
element_nodes = nodes[connectivity]
material_ids = np.array([fc.blocks[int(element.block)].material_id for element in elements])
polygons = element_nodes[:, [0, 4, 1, 5, 2, 6, 3, 7]]


""" ## 2. Физические свойства и квадратура QUAD8 ## """

def physical_properties(primitive):
    """Преобразует последние три столбца E, nu, rho в Vp, Vs, rho, Zp (СИ)."""
    young, poisson, density = np.moveaxis(primitive, -1, 0)
    vp = np.sqrt(young / density * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson)))
    vs = np.sqrt(young / (2 * density * (1 + poisson)))
    return np.stack((vp, vs, density, density * vp), axis=-1)


def with_impedance(values):
    """Добавляет Zp к массиву Vp, Vs, rho без изменения первых столбцов."""
    return np.concatenate((values, (values[..., 0] * values[..., 2])[..., None]), axis=-1)


def shape_quad8(r, s):
    """Функции формы восьмиузлового серендипова четырёхугольника."""
    return np.array([.25*(1-r)*(1-s)*(-r-s-1), .25*(1+r)*(1-s)*(r-s-1),
                     .25*(1+r)*(1+s)*(r+s-1), .25*(1-r)*(1+s)*(-r+s-1),
                     .5*(1-r*r)*(1-s), .5*(1+r)*(1-s*s),
                     .5*(1-r*r)*(1+s), .5*(1-r)*(1-s*s)])


gauss, gauss_weights = np.polynomial.legendre.leggauss(5)
point_parts, weight_parts = [], []
for j, s in enumerate(gauss):
    for i, r in enumerate(gauss):
        dr = shape_quad8(r + 1e-30j, s).imag / 1e-30
        ds = shape_quad8(r, s + 1e-30j).imag / 1e-30
        a = np.einsum("i,eij->ej", dr, element_nodes)
        b = np.einsum("i,eij->ej", ds, element_nodes)
        jacobian = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
        weight_parts.append(abs(jacobian)*gauss_weights[i]*gauss_weights[j])
        point_parts.append(np.einsum("i,eij->ej", shape_quad8(r, s), element_nodes))
points = np.stack(point_parts, axis=1)
weights = np.stack(weight_parts, axis=1)
assert np.isclose(weights.sum(), np.prod(domain_max), rtol=1e-8)


def sample_sgy(query, linear_nodes=False):
    """Читает физические ячейки SEG-Y или билинейно читает узловой вариант."""
    ix = np.clip(np.floor(query[..., 0]/5).astype(int), 0, nx-1)
    iz = np.clip(np.floor(query[..., 1]/5).astype(int), 0, nz-1)
    if not linear_nodes:
        return with_impedance(sgy[ix, iz])
    fx = np.clip(query[..., 0]/5 - ix, 0, 1)[..., None]
    fz = np.clip(query[..., 1]/5 - iz, 0, 1)[..., None]
    values = ((1-fx)*(1-fz)*aligned[ix, iz] + fx*(1-fz)*aligned[ix+1, iz]
              + (1-fx)*fz*aligned[ix, iz+1] + fx*fz*aligned[ix+1, iz+1])
    return with_impedance(values)


reference_cell = sample_sgy(points)
reference_nodes = sample_sgy(points, linear_nodes=True)
nearest_values = np.empty_like(reference_cell)
linear_values = np.empty_like(reference_cell)
bad_values = np.empty_like(reference_cell)
outside_hull = np.zeros(weights.shape, dtype=bool)
profile_points = np.column_stack((np.full_like(depth, 6000), depth))
profile_material = np.searchsorted(boundaries, depth) + 1
profile_linear = np.full((len(depth), 4), np.nan)
profile_outside = np.zeros(len(depth), dtype=bool)
layer_diagnostics = []


""" ## 3. Независимая триангуляция каждого материала ## """

for mid, material in fc.materials.items():
    properties = {prop.name: prop.data for groups in material.properties.values()
                  for group in groups for prop in group}
    data = [properties[name] for name in ("YOUNG_MODULE", "POISSON_RATIO", "DENSITY")]
    assert all(item.type == "TABLE" for item in data)
    columns = {column.type: column.value.data for column in data[0].table}
    xy = np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"]))
    primitive = np.column_stack([item.value.data for item in data])
    for item in data[1:]:
        other = {column.type: column.value.data for column in item.table}
        assert np.array_equal(xy, np.column_stack((other["TABULAR_X"], other["TABULAR_Y"])))
    # Перенос начала координат улучшает обусловленность для тонких слоёв;
    # масштаб осей одинаков и геометрия Делоне сохраняется.
    origin = xy.mean(axis=0)
    triangulation = Delaunay(xy-origin)
    interpolator = LinearNDInterpolator(triangulation, primitive, fill_value=np.nan)
    tree = KDTree(xy)
    mask = material_ids == mid
    query = points[mask].reshape(-1, 2)
    _, indices = tree.query(query)
    interpolated = interpolator(query-origin)
    outside = ~np.isfinite(interpolated).all(axis=1)
    interpolated[outside] = primitive[indices[outside]]
    linear_values[mask] = physical_properties(interpolated).reshape((-1, 25, 4))
    nearest_values[mask] = physical_properties(primitive[indices]).reshape((-1, 25, 4))
    bad_values[mask] = physical_properties(primitive[0])
    outside_hull[mask] = outside.reshape((-1, 25))
    profile_mask = profile_material == mid
    query_profile = profile_points[profile_mask]
    profile_primitive = interpolator(query_profile-origin)
    missing = ~np.isfinite(profile_primitive).all(axis=1)
    if missing.any():
        _, near = tree.query(query_profile[missing])
        profile_primitive[missing] = primitive[near]
    profile_linear[profile_mask] = physical_properties(profile_primitive)
    profile_outside[profile_mask] = missing
    # Интерполятор должен воспроизводить исходные значения в табличных узлах.
    check = np.unique(np.linspace(0, len(xy)-1, min(200, len(xy))).astype(int))
    assert np.allclose(interpolator(xy[check]-origin), primitive[check], rtol=1e-9, atol=1e-9)
    layer_diagnostics.append(dict(id=int(mid), triangles=len(triangulation.simplices),
                                  outside_area_pct=float(100*np.sum(weights[mask]*outside_hull[mask])/weights[mask].sum())))
    print(f"Layer {mid}: outside hull {layer_diagnostics[-1]['outside_area_pct']:.3f}% area", flush=True)
assert np.isfinite(linear_values).all() and np.isfinite(profile_linear).all()


""" ## 4. Разности: все свойства, включая акустический импеданс ## """

def statistics(difference, area_weights):
    """Статистика подписанной относительной разности (%) с весами площади."""
    d, w = difference.ravel(), area_weights.ravel()
    valid = w > 0
    d, w = d[valid], w[valid]
    absolute = abs(d)
    order = np.argsort(absolute)
    cumulative = np.cumsum(w[order])/w.sum()
    quantiles = np.interp([.95, .99], cumulative, absolute[order])
    return dict(mean=float(np.average(d, weights=w)), mae=float(np.average(absolute, weights=w)),
                p95=float(quantiles[0]), p99=float(quantiles[1]), max_abs=float(absolute.max()))


ix = np.clip(np.floor(points[..., 0]/5).astype(int), 0, nx-1)
iz = np.clip(np.floor(points[..., 1]/5).astype(int), 0, nz-1)
same_layer = table_layers[ix, iz] == material_ids[:, None]
unchanged_layers = (material_ids >= 6) & (material_ids <= 74)
masks = {"all": np.ones_like(weights), "inside_hull": ~outside_hull,
         "same_layer": same_layer,
         "layers_6_74_same_layer_inside": unchanged_layers[:, None] & same_layer & ~outside_hull}
results, element_maps = {}, {}
for reference_name, reference in (("cell", reference_cell), ("node_linear", reference_nodes)):
    for method, values in (("bad", bad_values), ("nearest", nearest_values), ("linear", linear_values)):
        difference = 100*(values-reference)/reference
        key = method+"_vs_"+reference_name
        results[key] = {region: [statistics(difference[..., k], weights*mask) for k in range(4)]
                        for region, mask in masks.items()}
        element_maps[key] = np.sum(difference*weights[..., None], axis=1)/weights.sum(axis=1)[:, None]
        print(key, results[key]["all"], flush=True)

times = {}
for method, values in (("sgy", profile_sgy), ("nearest", profile_nn),
                        ("bad", profile_bad), ("linear", profile_linear)):
    one_way = np.r_[0, np.cumsum(np.diff(depth_edges)/values[:, 0])]
    times[method] = 1/30+2*one_way-np.interp(5, depth_edges, one_way)
arrival_times = {method: {"boundary_73": float(np.interp(boundaries[72], depth_edges, values)),
                          "boundary_74": float(np.interp(boundaries[73], depth_edges, values)),
                          "depth_2495": float(np.interp(2495, depth_edges, values)),
                          "bottom": float(values[-1])} for method, values in times.items()}
print("Arrival times:", arrival_times, flush=True)


""" ## 5. Рисунки и сохранение результата ## """

labels = ["Vp", "Vs", "Плотность", "P-импеданс ρVp"]
for reference in ("cell", "node_linear"):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), layout="constrained", sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        values = element_maps["linear_vs_"+reference][:, k]
        limit = max(float(np.percentile(abs(values), 99)), .01)
        collection = PolyCollection(polygons, array=values, cmap="RdBu_r", clim=(-limit, limit), edgecolors="none", rasterized=True)
        ax.add_collection(collection)
        ax.set(xlim=(0, domain_max[0]), ylim=(domain_max[1], 0), ylabel="Глубина, м", title=labels[k])
        stat = results["linear_vs_"+reference]["all"][k]
        ax.text(.99, .06, f"MAE {stat['mae']:.3f}%; P95 {stat['p95']:.3f}%", transform=ax.transAxes,
                ha="right", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        fig.colorbar(collection, ax=ax, label="100 × (FC linear − SEG-Y) / SEG-Y, %", extend="both")
    axes[-1].set_xlabel("x, м")
    reference_title = "физические ячейки" if reference == "cell" else "узловой растр, билинейно"
    fig.suptitle("Послойная линейная интерполяция FC − SEG-Y: "+reference_title+
                 "\nОбщая область 0–2650 м; карта — среднее по элементу, статистика — квадратура 5×5 с весами площади.", fontsize=12)
    fig.savefig(root/f"img/{prefix}_{reference}.png", dpi=150)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 8), layout="constrained")
for method, values, color, label in (("sgy", profile_sgy, "tab:blue", "SEG-Y, ячейки"),
                                    ("nearest", profile_nn, "0.55", "FC, ближайший сосед"),
                                    ("linear", profile_linear, "tab:orange", "FC, линейно по слоям"),
                                    ("bad", profile_bad, "tab:red", "Ошибка: первая строка")):
    axes[0].plot(values[:, 0]/1000, depth, color=color, label=label)
    axes[1].plot(values[:, 0]*values[:, 2]/1e6, depth, color=color)
    axes[2].plot(times[method], depth_edges, color=color)
for ax in axes:
    ax.set_ylim(2650, 2100)
    ax.set_ylabel("Глубина, м")
    ax.grid(alpha=.2)
    for boundary in boundaries[72:74]:
        ax.axhline(boundary, color="0.5", ls=":", lw=.8)
axes[0].set(xlabel="Vp, км/с", xlim=(2.6, 5.2), title="Скорость P")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="ρVp, МПа·с/м", xlim=(5.5, 13), title="P-импеданс")
axes[2].set(xlabel="Время PP с учётом источника, с", xlim=(1.9, 2.3), title="Глубинно-временная привязка")
fig.suptitle("x=6000 м: сравнение способов чтения материала FC\nРикер 30 Гц; максимум 0.03333 с; источник на глубине 5 м.", fontsize=13)
fig.savefig(root/f"img/{prefix}_x6000.png", dpi=170)
plt.close(fig)

report = dict(fc_sha256=fc_hash, input_hashes=input_hashes,
              method="Delaunay/LinearNDInterpolator of E, nu, rho per material; nearest same material outside hull",
              quadrature_order=5, quadrature_points=int(weights.size), area_m2=float(weights.sum()),
              outside_hull_area_pct=float(100*np.sum(weights*outside_hull)/weights.sum()),
              properties=["Vp", "Vs", "rho", "Zp"], statistics=results, layers=layer_diagnostics,
              arrival_times=arrival_times, reference="denominator is SEG-Y for all comparisons")
(root/f"data/{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
np.savez_compressed(root/f"data/{prefix}.npz", **element_maps, profile_depth=depth,
                    profile_depth_edges=depth_edges, profile_linear=profile_linear,
                    profile_outside_hull=profile_outside, time_linear=times["linear"],
                    element_outside_area_fraction=np.sum(weights*outside_hull, axis=1)/weights.sum(axis=1))

""" ## Выводы ## """

"""
Вычислены поля при линейной интерполяции исходных упругих параметров внутри
каждого слоя. Сравнены Vp, Vs, плотность и P-импеданс с двумя трактовками
SEG-Y, ближайшим соседом и ошибочным выбором константы слоя. Измерено влияние
выхода за выпуклую оболочку таблиц; рассчитаны времена отражений при x=6000.
Численные результаты сохранены в data/, рисунки — в img/. Изменения
расчётного загрузчика и новый волновой расчёт не выполнялись.
"""
