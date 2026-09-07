""" # MILEN SEM 2D. Часть II. Согласованный с SEG-Y расчётный FC # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 5: новый материал для послойной интерполяции Делоне ## """

"""
Создадим копию расчётного FC с E, nu, rho, восстановленными из исходных
узловых SEG-Y. Сохраним исходную геометрию и остальные поля JSON.
Сохраним число и принадлежность точек каждому материалу; вне оболочки
применим ближайшего соседа своего слоя. Расширенные опорные области
рассмотрим лишь как сравнительный вариант сглаженного растра.
Проверим все локальные GLL-узлы порядка 7, точность
исходных отсчётов, разности полей и повторное чтение готового FC.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_dilation
from scipy.spatial import Delaunay, KDTree
import segyio

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(root / "fc-model/src"))
from fc_model import FCModel, FCValue

prefix = "dev_2_5_5"
input_name = "data/dev_2_2_milen2Do5pore_full.fc"
output_name = f"data/{prefix}_milen2Do5pore_full_node_aligned.fc"
# Сохраняем принадлежность опорных точек слоям. Расширенный вариант ниже
# служит сравнением с непрерывным билинейным растром, но в FC не попадает.
selected_method = "replaced_linear"
manifest = {}


""" ## 1. Загружаем и проверяем эталонные данные ## """

def input_path(name):
    """Возвращает путь и сохраняет контрольную сумму входа."""
    path = root / name
    with path.open("rb") as stream:
        manifest[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return path


source = json.loads(input_path(input_name).read_text())
output = copy.deepcopy(source)
model = FCModel.decode(source)
grid_data = np.load(input_path("data/dev_1_7_material_grids.npz"))
layers = grid_data["layer_indexes_grid"]
grid_xy = grid_data["coords_grid"]
sgy_columns = []
for name in ("Vp", "Vs", "Density"):
    with segyio.open(str(input_path(f"data/dev_2_5_legacy_{name}_model_node_aligned.sgy")),
                     "r", ignore_geometry=True) as stream:
        sgy_columns.append(np.asarray(stream.trace.raw[:], dtype=float))
sgy = np.stack(sgy_columns, axis=-1)
sgy[..., 2] *= 1000
assert sgy.shape == (layers.shape[0]+1, layers.shape[1]+1, 3)
assert np.allclose(grid_xy[..., 0], (np.arange(layers.shape[0])[:, None]+.5)*5)
assert np.allclose(grid_xy[..., 1], (np.arange(layers.shape[1])[None, :]+.5)*5)
assert np.isfinite(sgy).all() and (sgy > 0).all()
vp, vs, density = np.moveaxis(sgy, -1, 0)
poisson = (vp**2-2*vs**2)/(2*(vp**2-vs**2))
primitive_grid = np.stack((2*density*vs**2*(1+poisson), poisson, density), axis=-1)
assert ((poisson > -1) & (poisson < .5)).all()


def sample_nodes(query):
    """Билинейное восстановление Vp, Vs, rho узлового SEG-Y; края постоянны."""
    scaled = np.clip(query/5, [0, 0], np.array(sgy.shape[:2])-1)
    ij = np.minimum(np.floor(scaled).astype(int), np.array(sgy.shape[:2])-2)
    ix, iz = np.moveaxis(ij, -1, 0)
    a, b = np.moveaxis(scaled-ij, -1, 0)
    return ((1-a)*(1-b))[..., None]*sgy[ix, iz] + (a*(1-b))[..., None]*sgy[ix+1, iz] + ((1-a)*b)[..., None]*sgy[ix, iz+1] + (a*b)[..., None]*sgy[ix+1, iz+1]


def primitive_from_nodes(query):
    """Вычисляет E, nu, rho после билинейной интерполяции скоростей и плотности."""
    vp, vs, rho = np.moveaxis(sample_nodes(query), -1, 0)
    nu = (vp**2-2*vs**2)/(2*(vp**2-vs**2))
    return np.stack((2*rho*vs**2*(1+nu), nu, rho), axis=-1)


def physical(primitive):
    """Вычисляет Vp, Vs, rho, Zp из E, nu, rho в СИ."""
    young, nu, rho = np.moveaxis(primitive, -1, 0)
    vp = np.sqrt(young*(1-nu)/(rho*(1+nu)*(1-2*nu)))
    vs = np.sqrt(young/(2*rho*(1+nu)))
    return np.stack((vp, vs, rho, rho*vp), axis=-1)


def shape_quad8(r, s):
    """Функции формы QUAD8 в порядке углы, затем середины сторон."""
    return np.array([.25*(1-r)*(1-s)*(-r-s-1), .25*(1+r)*(1-s)*(r-s-1),
                     .25*(1+r)*(1+s)*(r+s-1), .25*(1-r)*(1+s)*(-r+s-1),
                     .5*(1-r*r)*(1-s), .5*(1+r)*(1-s*s),
                     .5*(1-r*r)*(1+s), .5*(1-r)*(1-s*s)])


""" ## 2. Строим квадратуру и профиль внутри расчётной геометрии ## """

nodes = model.mesh.nodes_xyz[:, :2]
elements = sorted(model.mesh.elements["QUAD8"].values(), key=lambda item: item.id)
lookup = {int(nid): i for i, nid in enumerate(model.mesh.nodes_ids)}
connectivity = np.array([[lookup[int(nid)] for nid in element.nodes] for element in elements])
element_nodes = nodes[connectivity]
material_ids = np.array([model.blocks[int(element.block)].material_id for element in elements])
domain = np.round(nodes.max(axis=0), 6)
assert np.array_equal(domain, [11750, 2650])
gauss, gauss_weights = np.polynomial.legendre.leggauss(5)
point_parts, weight_parts = [], []
for j, s in enumerate(gauss):
    for i, r in enumerate(gauss):
        dr = shape_quad8(r+1e-30j, s).imag/1e-30
        ds = shape_quad8(r, s+1e-30j).imag/1e-30
        a = np.einsum("i,eij->ej", dr, element_nodes)
        b = np.einsum("i,eij->ej", ds, element_nodes)
        weight_parts.append(abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])*gauss_weights[i]*gauss_weights[j])
        point_parts.append(np.einsum("i,eij->ej", shape_quad8(r, s), element_nodes))
points, weights = np.stack(point_parts, axis=1), np.stack(weight_parts, axis=1)
assert np.isclose(weights.sum(), np.prod(domain), rtol=1e-8)
ix = np.clip(np.floor(points[..., 0]/5).astype(int), 0, layers.shape[0]-1)
iz = np.clip(np.floor(points[..., 1]/5).astype(int), 0, layers.shape[1]-1)
reference = sample_nodes(points)
reference = np.concatenate((reference, (reference[..., 0]*reference[..., 2])[..., None]), axis=-1)
same_layer = layers[ix, iz] == material_ids[:, None]-1
gll = np.r_[-1, np.polynomial.legendre.Legendre.basis(7).deriv().roots(), 1]
gll_shapes = np.array([shape_quad8(r, s) for s in gll for r in gll])
audit = np.load(input_path("data/dev_2_5_2_fc_segy_audit.npz"))
depth = audit["profile_depth"]
depth_edges = audit["profile_depth_edges"]
boundaries = audit["fc_boundaries_x6000"]
profile_points = np.column_stack((np.full_like(depth, 6000), depth))
profile_ids = np.searchsorted(boundaries, depth)+1
assert depth_edges[-1] == 2650
values = {key: np.empty_like(reference) for key in ("old_linear", "replaced_linear", "new_linear")}
profiles = {key: np.empty((len(depth), 4)) for key in values}
layer_reports = []
output_materials = {item["id"]: item for item in output["materials"]}
replacement_keys = (("elasticity", 0, "YOUNG_MODULE"),
                    ("elasticity", 1, "POISSON_RATIO"), ("common", 0, "DENSITY"))


""" ## 3. Меняем только E, nu, rho; проверяем линейное восстановление ## """

for mid, material in model.materials.items():
    props = {prop.name: prop.data for groups in material.properties.values()
             for group in groups for prop in group}
    columns = {col.type: col.value.data for col in props["YOUNG_MODULE"].table}
    old_xy = np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"]))
    old = np.column_stack([props[name].value.data for _, _, name in replacement_keys])
    mask = layers == mid-1
    assert np.array_equal(old_xy, grid_xy[mask])
    base_xy, base = grid_xy[mask]-2.5, primitive_grid[:-1, :-1][mask]
    origin = base_xy.mean(axis=0)
    base_tri = Delaunay(base_xy-origin)
    old_tri = Delaunay(old_xy-origin)
    nearest = KDTree(base_xy)
    element_mask = material_ids == mid
    query = points[element_mask].reshape(-1, 2)
    profile_mask = profile_ids == mid
    combined_query = np.concatenate((query, profile_points[profile_mask]))
    gll_points = np.einsum("qi,eij->eqj", gll_shapes, element_nodes[element_mask]).reshape(-1, 2)
    # Ореол расширяется только для покрытия настоящих элементов слоя.
    # Одинаковые точки в разных таблицах имеют одинаковый эталонный материал.
    for halo_cells in (2, 4, 8, 16):
        padded = np.pad(mask, halo_cells)
        halo_indices = np.argwhere(binary_dilation(padded, structure=np.ones((3, 3)), iterations=halo_cells) & ~padded)-halo_cells
        halo_xy = halo_indices*5.0
        new_xy = np.concatenate((base_xy, halo_xy))
        new = primitive_from_nodes(new_xy)
        new_tri = Delaunay(new_xy-origin)
        interpolation = LinearNDInterpolator(new_tri, new)
        gll_values = interpolation(gll_points-origin)
        if np.isfinite(gll_values).all():
            break
    assert np.isfinite(gll_values).all(), f"Слой {mid}: GLL вне выпуклой оболочки"
    selected_xy, selected = (base_xy, base) if selected_method == "replaced_linear" else (new_xy, new)
    selected_interp = LinearNDInterpolator(base_tri, base) if selected_method == "replaced_linear" else interpolation
    selected_gll = selected_interp(gll_points-origin)
    selected_outside = ~np.isfinite(selected_gll).all(axis=1)
    if selected_outside.any():
        _, indices = KDTree(selected_xy).query(gll_points[selected_outside])
        selected_gll[selected_outside] = selected[indices]
    assert np.isfinite(selected_gll).all()
    assert (selected_gll[:, [0, 2]] > 0).all()
    assert ((selected_gll[:, 1] > -1) & (selected_gll[:, 1] < .5)).all()
    misses = {}
    for name, table in (("old_linear", old), ("replaced_linear", base), ("new_linear", new)):
        interp = interpolation if name == "new_linear" else LinearNDInterpolator(old_tri if name == "old_linear" else base_tri, table)
        evaluated = interp(combined_query-origin)
        outside = ~np.isfinite(evaluated).all(axis=1)
        misses[name] = outside[:len(query)].reshape((-1, 25))
        if name == "new_linear":
            assert not outside.any(), f"Слой {mid}: расширить опорные точки"
        elif outside.any():
            _, near = (KDTree(old_xy) if name == "old_linear" else nearest).query(combined_query[outside])
            evaluated[outside] = table[near]
        converted = physical(evaluated)
        values[name][element_mask] = converted[:len(query)].reshape((-1, 25, 4))
        profiles[name][profile_mask] = converted[len(query):]
    # Независимый контроль всех локальных GLL-узлов модели порядка 7.
    assert np.isfinite(gll_values).all(), f"Слой {mid}: GLL вне выпуклой оболочки"
    assert (gll_values[:, 0] > 0).all() and (gll_values[:, 2] > 0).all()
    assert ((gll_values[:, 1] > -1) & (gll_values[:, 1] < .5)).all()
    reproduced = interpolation(base_xy-origin)
    assert np.allclose(reproduced, base, rtol=1e-9, atol=1e-9)
    max_node_percent = 100*np.max(abs(physical(reproduced)[:, :3]/sgy[:-1, :-1][mask]-1), axis=0)
    # Редактируем только четыре поля записи каждого из трёх свойств.
    # Остальные JSON-данные не проходят через нормализацию FCModel.save.
    encoded_xy = [FCValue(selected_xy[:, k], "array").encode() for k in range(2)]
    for k, (group_name, property_code, _) in enumerate(replacement_keys):
        group = next(group for group in output_materials[mid][group_name] if property_code in group["const_names"])
        index = group["const_names"].index(property_code)
        group["constants"][index] = FCValue(selected[:, k], "array").encode()
        group["const_types"][index] = [1, 2]
        group["const_dep"][index] = encoded_xy.copy()
        group["const_dep_size"][index] = len(selected)
    layer_reports.append(dict(id=int(mid), original_points=len(base), halo_points=len(halo_xy), halo_cells=halo_cells,
        output_points=len(selected), gll_points=len(gll_points), outside_gll_count=int(selected_outside.sum()),
        original_outside_area_m2=float((weights[element_mask]*misses["replaced_linear"]).sum()),
        max_sample_relative_percent=max_node_percent.tolist()))
    print(f"Layer {mid}: {len(selected)} output points; {int(selected_outside.sum())}/{len(gll_points)} GLL use nearest; all finite", flush=True)


""" ## 4. Проверяем сохранение исходной модели и повторное чтение ## """

restored = copy.deepcopy(output)
for old_material, new_material in zip(source["materials"], restored["materials"]):
    assert old_material["id"] == new_material["id"]
    for group_name, property_code, _ in replacement_keys:
        for old_group, new_group in zip(old_material[group_name], new_material[group_name]):
            if property_code in old_group["const_names"]:
                index = old_group["const_names"].index(property_code)
                for field in ("constants", "const_types", "const_dep", "const_dep_size"):
                    new_group[field][index] = copy.deepcopy(old_group[field][index])
assert restored == source, "Изменено что-то помимо трёх свойств материала"
del restored
destination = root / output_name
assert destination != root / input_name
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
reloaded_raw = json.loads(destination.read_text())
assert reloaded_raw == output
reloaded = FCModel.decode(reloaded_raw)
for mid, material in reloaded.materials.items():
    props = {prop.name: prop.data for groups in material.properties.values() for group in groups for prop in group}
    for _, _, name in replacement_keys:
        assert props[name].type == "TABLE"
        assert np.isfinite(props[name].value.data).all()
    assert len(props["YOUNG_MODULE"].value.data) == layer_reports[mid-1]["output_points"]
    if selected_method == "replaced_linear":
        mask = layers == mid-1
        for k, (_, _, name) in enumerate(replacement_keys):
            assert np.array_equal(props[name].value.data, primitive_grid[:-1, :-1][mask, k])
            columns = {column.type: column.value.data for column in props[name].table}
            assert np.array_equal(np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"])), grid_xy[mask]-2.5)
with destination.open("rb") as stream:
    output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
print(f"Saved and reloaded: {destination}", flush=True)


def statistics(difference, area_weights):
    """Вычисляет MAE, P95 и максимум относительной разности с весами площади."""
    d, w = difference.ravel(), area_weights.ravel()
    valid = w > 0
    d, w = d[valid], w[valid]
    order = np.argsort(abs(d))
    return dict(mae=float(np.average(abs(d), weights=w)), mean=float(np.average(d, weights=w)),
                p95=float(np.interp(.95, np.cumsum(w[order])/w.sum(), abs(d)[order])),
                max_abs=float(abs(d).max()))


""" ## 5. Измеряем остаток и строим карты разностей ## """

results, maps, times, worst_points = {}, {}, {}, {}
for name, field in values.items():
    difference = 100*(field/reference-1)
    results[name] = {region: [statistics(difference[..., k], weights*mask) for k in range(4)]
                     for region, mask in (("all", np.ones_like(weights)), ("same_layer", same_layer))}
    maps[name] = np.sum(difference*weights[..., None], axis=1)/weights.sum(axis=1)[:, None]
    worst_points[name] = []
    for k in range(4):
        e, q = np.unravel_index(np.argmax(abs(difference[..., k])), weights.shape)
        worst_points[name].append(dict(x_m=float(points[e, q, 0]), depth_m=float(points[e, q, 1]),
            material_id=int(material_ids[e]), difference_percent=float(difference[e, q, k])))
    one_way = np.r_[0, np.cumsum(np.diff(depth_edges)/profiles[name][:, 0])]
    times[name] = 1/30+2*one_way-np.interp(5, depth_edges, one_way)
    print(name, results[name]["all"], flush=True)

polygons = element_nodes[:, [0, 4, 1, 5, 2, 6, 3, 7]]
labels = ["Vp", "Vs", "Плотность", "P-импеданс"]
fig, axes = plt.subplots(4, 2, figsize=(17, 12), layout="constrained", sharex=True, sharey=True)
for col, name in enumerate(("old_linear", selected_method)):
    for k, label in enumerate(labels):
        ax = axes[k, col]
        limit = 5 if k != 2 else 1.5
        pc = PolyCollection(polygons, array=maps[name][:, k], cmap="RdBu_r", clim=(-limit, limit), rasterized=True)
        ax.add_collection(pc)
        ax.set(xlim=(0, 11750), ylim=(2650, 0), ylabel="Глубина, м")
        ax.set_title(f"{'Исходный FC' if col == 0 else 'Новый FC'}: {label}; MAE={results[name]['all'][k]['mae']:.3f}%")
        fig.colorbar(pc, ax=ax, label="(FC − SEG-Y) / SEG-Y, %", extend="both")
for ax in axes[-1]:
    ax.set_xlabel("x, м")
fig.suptitle("Линейная интерполяция Делоне по слоям: до и после согласования материала\nЭталон — node_aligned SEG-Y, билинейно; карты — средняя разность по элементу")
fig.savefig(root / f"img/{prefix}_segy_matched_difference.png", dpi=160)
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(14, 8), layout="constrained")
sgy_profile = sample_nodes(profile_points)
sgy_one_way = np.r_[0, np.cumsum(np.diff(depth_edges)/sgy_profile[:, 0])]
sgy_time = 1/30+2*sgy_one_way-np.interp(5, depth_edges, sgy_one_way)
for name, label, color in (("old_linear", "Исходный FC, линейно", "tab:red"),
                           (selected_method, "Новый FC, линейно + nearest", "tab:green")):
    axes[0].plot(profiles[name][:, 0]/1000, depth, color=color, label=label)
    axes[1].plot(profiles[name][:, 3]/1e6, depth, color=color)
    axes[2].plot(times[name], depth_edges, color=color)
axes[0].plot(sgy_profile[:, 0]/1000, depth, color="tab:blue", ls="--", label="SEG-Y, узлы, билинейно")
axes[1].plot(sgy_profile[:, 0]*sgy_profile[:, 2]/1e6, depth, color="tab:blue", ls="--")
axes[2].plot(sgy_time, depth_edges, color="tab:blue", ls="--")
for ax in axes:
    ax.set_ylim(2650, 2100)
    ax.set_ylabel("Глубина, м")
    ax.grid(alpha=.2)
    for boundary in boundaries[72:74]:
        ax.axhline(boundary, color=".5", ls=":", lw=.8)
axes[0].set(xlabel="Vp, км/с", title="Скорость P")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="ρVp, МПа·с/м", title="P-импеданс")
axes[2].set(xlabel="Время PP с учётом источника, с", title="Вертикальная привязка", xlim=(1.9, 2.25))
fig.suptitle("x=6000 м: новый материал и эталон SEG-Y; дно расчётной модели 2650 м")
fig.savefig(root / f"img/{prefix}_segy_matched_x6000.png", dpi=160)
plt.close(fig)

report = dict(input_hashes=manifest, output_file=output_name, output_sha256=output_hash,
    selected_method=selected_method,
    properties=["Vp", "Vs", "rho", "Zp"], reference="node_aligned SEG-Y, bilinear Vp/Vs/rho then Zp=rho*Vp",
    domain_max_m=domain.tolist(), exact_json_preservation_except_225_properties=True,
    support="original layer membership and point count; coordinates shifted (-2.5,-2.5) m to node_aligned grid" if selected_method == "replaced_linear" else "node grid with expanded halo",
    interpolation="Delaunay linear E, nu, rho per layer; nearest of same layer outside hull; isotropic conversion afterwards",
    gll_order=7, gll_checked=sum(layer["gll_points"] for layer in layer_reports),
    gll_outside_count=sum(layer["outside_gll_count"] for layer in layer_reports), quadrature_order=5, quadrature_count=int(weights.size),
    area_m2=float(weights.sum()), different_layer_area_percent=float(100*np.sum(weights*~same_layer)/weights.sum()),
    layers=layer_reports, statistics=results, worst_points=worst_points,
    arrival_times={name: {"boundary_73":float(np.interp(boundaries[72], depth_edges, t)),
                          "boundary_74":float(np.interp(boundaries[73], depth_edges, t)),
                          "bottom":float(t[-1])} for name, t in {**times, "segy_node":sgy_time}.items()})
(root / f"data/{prefix}_segy_matched_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
np.savez_compressed(root / f"data/{prefix}_segy_matched_check.npz", **maps,
    profile_depth=depth, profile_depth_edges=depth_edges, profile_segy_node=sgy_profile, time_segy_node=sgy_time,
    **{f"profile_{name}": p for name, p in profiles.items()}, **{f"time_{name}":t for name, t in times.items()})
for name, expected in manifest.items():
    with (root / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected


""" ## Выводы ## """

"""
Создали новый расчётный FC, изменив только E, nu, rho и их координаты.
Исходные отсчёты узловых SEG-Y воспроизводятся; ближайший сосед своего
слоя обеспечивает конечные значения вне оболочки в локальных GLL-узлах
порядка 7. Расширенные таблицы исследованы отдельно. Остаточные разности
линейного поля и билинейного узлового растра измерены отдельно от смены материала.
Геометрия, нагрузки, демпфирование, настройки и исходные файлы сохранены.
"""
