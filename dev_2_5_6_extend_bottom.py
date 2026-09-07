""" # MILEN SEM 2D. Часть II. Наращивание нижнего слоя до 2750 м # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 6: согласование глубины FC и узловых SEG-Y ## """

"""
Добавим к согласованной модели четыре ряда QUAD8 по 25 м. Все существующие
узлы и элементы оставим на месте. Продолжим блок 75, перенесём нижнее
поглощение и продлим боковое. Материал уже задан на всю глубину; проверим
его линейное восстановление с ближайшим соседом в новой полосе.
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, KDTree
import segyio

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(root / "fc-model/src"))
from fc_model import FCModel
from fc_model.fc_value import decode, encode

prefix = "dev_2_5_6"
input_name = "data/dev_2_5_5_milen2Do5pore_full_node_aligned.fc"
output_name = f"data/{prefix}_milen2Do5pore_full_aligned_2750.fc"
old_bottom, new_bottom, row_height = 2650., 2750., 25.
edge_indices = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]])
manifest = {}


""" ## 1. Читаем исходную модель и определяем внешние границы ## """

def checked_path(name):
    """Фиксирует хеш входного файла и возвращает его путь."""
    path = root / name
    with path.open("rb") as stream:
        manifest[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return path


source = json.loads(checked_path(input_name).read_text())
output = copy.deepcopy(source)
model = FCModel.decode(source)
original_xyz = model.mesh.nodes_xyz
original_ids = model.mesh.nodes_ids
assert np.allclose(original_xyz.min(axis=0), [0, 0, 0])
assert np.allclose(original_xyz.max(axis=0), [11750, old_bottom, 0])
assert list(model.mesh.elements) == ["QUAD8"]
node_xyz = {int(nid): point for nid, point in zip(original_ids, original_xyz)}
elements = model.mesh.elements["QUAD8"]
boundary_loads = {}
for load in source["loads"]:
    assert load["type"] == 20, "Проверить неизвестное граничное условие"
    pairs = decode(load["apply_to"]).reshape(-1, 2)
    points = np.array([[node_xyz[elements[int(eid)].nodes[j]] for j in edge_indices[edge]]
                       for eid, edge in pairs])
    if np.allclose(points[..., 1], old_bottom):
        side = "bottom"
    elif np.allclose(points[..., 0], 0):
        side = "left"
    elif np.allclose(points[..., 0], 11750):
        side = "right"
    else:
        raise ValueError("Неоднозначная внешняя граница")
    boundary_loads[side] = (load["id"], pairs)
assert set(boundary_loads) == {"left", "right", "bottom"}
bottom_pairs = boundary_loads["bottom"][1]
bottom_nodes = np.unique([elements[int(eid)].nodes[j] for eid, edge in bottom_pairs for j in edge_indices[edge]])
bottom_nodes = bottom_nodes[np.argsort([node_xyz[int(nid)][0] for nid in bottom_nodes])]
bottom_x = np.array([node_xyz[int(nid)][0] for nid in bottom_nodes])
n_columns = len(bottom_pairs)
assert len(bottom_nodes) == 2*n_columns+1
assert np.allclose(bottom_x[1::2], .5*(bottom_x[:-2:2]+bottom_x[2::2]), atol=1e-8)
assert np.all(np.diff(bottom_x) > 0)
assert np.isclose(bottom_x[0], 0) and np.isclose(bottom_x[-1], 11750)
assert all(elements[int(eid)].block == 75 and elements[int(eid)].parent_id == 75 for eid, _ in bottom_pairs)
stored_order = int(elements[int(bottom_pairs[0, 0])].order)
assert all(elements[int(eid)].order == stored_order for eid, _ in bottom_pairs)


""" ## 2. Добавляем четыре ряда, используя прежние узлы стыка ## """

next_node_id = int(original_ids.max())+1
next_element_id = max(elements)+1
added_ids, added_xyz, added_eids, added_connectivity = [], [], [], []
new_left, new_right, new_floor = [], [], []
top_ids = bottom_nodes.copy()
n_rows = int(round((new_bottom-old_bottom)/row_height))
assert n_rows*row_height == new_bottom-old_bottom
for row in range(n_rows):
    z_top = old_bottom+row*row_height
    floor_ids = np.arange(next_node_id, next_node_id+len(bottom_x), dtype=np.int32)
    next_node_id += len(floor_ids)
    side_mid_ids = np.arange(next_node_id, next_node_id+n_columns+1, dtype=np.int32)
    next_node_id += len(side_mid_ids)
    added_ids.extend(floor_ids.tolist())
    added_xyz.extend(np.column_stack((bottom_x, np.full(len(bottom_x), z_top+row_height), np.zeros(len(bottom_x)))))
    added_ids.extend(side_mid_ids.tolist())
    added_xyz.extend(np.column_stack((bottom_x[::2], np.full(n_columns+1, z_top+row_height/2), np.zeros(n_columns+1))))
    for column in range(n_columns):
        eid = next_element_id
        next_element_id += 1
        added_eids.append(eid)
        added_connectivity.append([top_ids[2*column], top_ids[2*column+2],
            floor_ids[2*column+2], floor_ids[2*column], top_ids[2*column+1],
            side_mid_ids[column+1], floor_ids[2*column+1], side_mid_ids[column]])
        if column == 0:
            new_left.append([eid, 3])
        if column == n_columns-1:
            new_right.append([eid, 1])
        if row == n_rows-1:
            new_floor.append([eid, 2])
    top_ids = floor_ids
added_ids = np.array(added_ids, dtype=np.int32)
added_xyz = np.array(added_xyz, dtype=np.float64)
added_eids = np.array(added_eids, dtype=np.int32)
added_connectivity = np.array(added_connectivity, dtype=np.int32)
assert len(added_eids) == n_rows*n_columns
assert len(set(added_ids)) == len(added_ids)
assert not set(added_ids) & set(original_ids)
assert len(np.unique(np.vstack((original_xyz, added_xyz)), axis=0)) == len(original_xyz)+len(added_xyz)


def append_mesh_array(field, additional, dtype=np.int32):
    """Добавляет массив к FC-полю, сохраняя весь исходный префикс."""
    original = decode(source["mesh"][field], np.dtype(dtype))
    result = np.concatenate((original, np.asarray(additional, dtype=dtype).ravel()))
    assert np.array_equal(result[:len(original)], original)
    output["mesh"][field] = encode(result)


append_mesh_array("nodes", added_xyz, np.float64)
append_mesh_array("nids", added_ids)
append_mesh_array("elems", added_connectivity)
append_mesh_array("elemids", added_eids)
for field, value, dtype in (("elem_blocks", 75, np.int32), ("elem_parent_ids", 75, np.int32),
                            ("elem_orders", stored_order, np.int32),
                            ("elem_types", int(decode(source["mesh"]["elem_types"], np.dtype(np.uint8))[0]), np.uint8)):
    append_mesh_array(field, np.full(len(added_eids), value), dtype)
output["mesh"]["nodes_count"] += len(added_ids)
output["mesh"]["elems_count"] += len(added_eids)
print(f"Added {len(added_ids)} nodes and {len(added_eids)} QUAD8 in {n_rows} rows", flush=True)


""" ## 3. Переносим поглощение и расширяем набор всех узлов ## """

new_load_pairs = {}
for side, (load_id, original_pairs) in boundary_loads.items():
    if side == "bottom":
        pairs = np.asarray(new_floor, dtype=np.int32)
    else:
        extension = new_left if side == "left" else new_right
        pairs = np.vstack((original_pairs, np.asarray(extension, dtype=np.int32)))
    load = next(item for item in output["loads"] if item["id"] == load_id)
    load["apply_to"] = encode(pairs)
    load["apply_to_size"] = len(pairs)
    new_load_pairs[side] = pairs
updated_sets = []
for node_set in output.get("sets", {}).get("nodesets", []):
    members = decode(node_set["apply_to"])
    if len(members) == len(original_ids) and set(members) == set(original_ids):
        node_set["apply_to"] = encode(np.r_[members, added_ids].astype(np.int32))
        node_set["apply_to_size"] = len(members)+len(added_ids)
        updated_sets.append(node_set["id"])
assert updated_sets == [1]
# Разрешённые изменения ограничены сеткой, тремя ABC и набором всех узлов.
restored = copy.deepcopy(output)
restored["mesh"] = source["mesh"]
restored["loads"] = source["loads"]
restored["sets"] = source["sets"]
assert restored == source
for before, after in zip(source["loads"], output["loads"]):
    assert {k:v for k,v in before.items() if k not in ("apply_to", "apply_to_size")} == {k:v for k,v in after.items() if k not in ("apply_to", "apply_to_size")}
assert output["sets"]["nodesets"][1] == source["sets"]["nodesets"][1]
del restored


""" ## 4. Проверяем стык, внешние рёбра и качество элементов ## """

extended = FCModel.decode(output)
all_xyz = extended.mesh.nodes_xyz
all_ids = extended.mesh.nodes_ids
lookup = {int(nid): i for i, nid in enumerate(all_ids)}
all_elements = list(extended.mesh.elements["QUAD8"].values())
all_connectivity = np.array([element.nodes for element in all_elements])
all_edges = all_connectivity[:, edge_indices].reshape(-1, 3)
edge_counts = Counter(tuple(sorted(edge)) for edge in all_edges)
assert set(edge_counts.values()) == {1, 2}, "Неконформная сетка"
for eid, edge in bottom_pairs:
    key = tuple(sorted(elements[int(eid)].nodes[j] for j in edge_indices[edge]))
    assert edge_counts[key] == 2, "Разрыв на старом дне"
boundary_keys = {edge for edge, count in edge_counts.items() if count == 1}
abc_keys = set()
for side, pairs in new_load_pairs.items():
    for eid, edge in pairs:
        element = extended.mesh.elements["QUAD8"][int(eid)]
        key = tuple(sorted(element.nodes[j] for j in edge_indices[edge]))
        assert key in boundary_keys and key not in abc_keys
        coordinates = all_xyz[[lookup[int(nid)] for nid in key]]
        if side == "bottom":
            assert np.allclose(coordinates[:, 1], new_bottom)
        else:
            assert np.allclose(coordinates[:, 0], 0 if side == "left" else 11750)
        abc_keys.add(key)
free_keys = boundary_keys-abc_keys
assert all(np.allclose(all_xyz[[lookup[int(nid)] for nid in edge], 1], 0) for edge in free_keys)
assert np.array_equal(all_xyz[:len(original_xyz)], original_xyz)
assert np.array_equal(all_ids[:len(original_ids)], original_ids)
assert np.allclose(all_xyz.max(axis=0), [11750, new_bottom, 0])
new_element_xyz = all_xyz[np.array([[lookup[int(nid)] for nid in conn] for conn in added_connectivity]), :2]


def shape_quad8(r, s):
    """Функции формы QUAD8 в порядке углов и середин сторон."""
    return np.array([.25*(1-r)*(1-s)*(-r-s-1), .25*(1+r)*(1-s)*(r-s-1),
                     .25*(1+r)*(1+s)*(r+s-1), .25*(1-r)*(1+s)*(-r+s-1),
                     .5*(1-r*r)*(1-s), .5*(1+r)*(1-s*s),
                     .5*(1-r*r)*(1+s), .5*(1-r)*(1-s*s)])


gll = np.r_[-1, np.polynomial.legendre.Legendre.basis(7).deriv().roots(), 1]
gauss, gauss_weights = np.polynomial.legendre.leggauss(5)
jacobian_min, jacobian_max = np.inf, -np.inf
queries, weights = [], []
for rule, rule_weights in ((gll, None), (gauss, gauss_weights)):
    for j, s in enumerate(rule):
        for i, r in enumerate(rule):
            dr = shape_quad8(r+1e-30j, s).imag/1e-30
            ds = shape_quad8(r, s+1e-30j).imag/1e-30
            a = np.einsum("i,eij->ej", dr, new_element_xyz)
            b = np.einsum("i,eij->ej", ds, new_element_xyz)
            jac = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
            assert (jac > 0).all()
            jacobian_min = min(jacobian_min, float(jac.min()))
            jacobian_max = max(jacobian_max, float(jac.max()))
            if rule_weights is not None:
                queries.append(np.einsum("i,eij->ej", shape_quad8(r, s), new_element_xyz))
                weights.append(jac*rule_weights[i]*rule_weights[j])
queries = np.stack(queries, axis=1)
weights = np.stack(weights, axis=1)
assert np.isclose(weights.sum(), 11750*(new_bottom-old_bottom), rtol=1e-10)
print(f"Conforming interface; ABC exterior only; min Jacobian {jacobian_min:.6f}; added area {weights.sum():.1f}", flush=True)


""" ## 5. Сравниваем материал добавленной полосы с узловым SEG-Y ## """

fields = {prop.name: prop.data for groups in extended.materials[75].properties.values() for group in groups for prop in group}
tab = fields["YOUNG_MODULE"]
columns = {column.type: column.value.data for column in tab.table}
table_xy = np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"]))
primitive = np.column_stack([fields[name].value.data for name in ("YOUNG_MODULE", "POISSON_RATIO", "DENSITY")])
origin = table_xy.mean(axis=0)
interpolator = LinearNDInterpolator(Delaunay(table_xy-origin), primitive)
nearest = KDTree(table_xy)


def evaluate_fc(query):
    """Линейно читает слой 75, вне оболочки использует его ближайшую точку."""
    primitive = interpolator(query-origin)
    missing = ~np.isfinite(primitive).all(axis=-1)
    if missing.any():
        _, indices = nearest.query(query[missing])
        primitive[missing] = np.column_stack([fields[name].value.data[indices] for name in ("YOUNG_MODULE", "POISSON_RATIO", "DENSITY")])
    young, nu, rho = np.moveaxis(primitive, -1, 0)
    assert np.isfinite(primitive).all() and (young > 0).all() and (rho > 0).all()
    assert ((nu > -1) & (nu < .5)).all()
    vp = np.sqrt(young*(1-nu)/(rho*(1+nu)*(1-2*nu)))
    vs = np.sqrt(young/(2*rho*(1+nu)))
    return np.stack((vp, vs, rho, rho*vp), axis=-1), missing


sgy = []
for name in ("Vp", "Vs", "Density"):
    with segyio.open(str(checked_path(f"data/dev_2_5_legacy_{name}_model_node_aligned.sgy")), "r", ignore_geometry=True) as stream:
        assert len(stream.samples) == 551 and np.isclose(stream.samples[-1], new_bottom)
        sgy.append(np.asarray(stream.trace.raw[:], dtype=float))
sgy = np.stack(sgy, axis=-1)
sgy[..., 2] *= 1000


def sample_sgy(query):
    """Билинейно читает Vp, Vs, rho из узлового растра, затем вычисляет Zp."""
    q = np.clip(query/5, [0, 0], np.array(sgy.shape[:2])-1)
    ij = np.minimum(np.floor(q).astype(int), np.array(sgy.shape[:2])-2)
    ix, iz = np.moveaxis(ij, -1, 0)
    a, b = np.moveaxis(q-ij, -1, 0)
    result = ((1-a)*(1-b))[..., None]*sgy[ix, iz] + (a*(1-b))[..., None]*sgy[ix+1, iz] + ((1-a)*b)[..., None]*sgy[ix, iz+1] + (a*b)[..., None]*sgy[ix+1, iz+1]
    return np.concatenate((result, (result[..., 0]*result[..., 2])[..., None]), axis=-1)


actual, outside = evaluate_fc(queries)
reference = sample_sgy(queries)
difference = 100*(actual/reference-1)
gll_shapes = np.array([shape_quad8(r, s) for s in gll for r in gll])
gll_queries = np.einsum("qi,eij->eqj", gll_shapes, new_element_xyz)
gll_values, gll_outside = evaluate_fc(gll_queries)
metrics = []
for k in range(4):
    absolute = abs(difference[..., k]).ravel()
    order = np.argsort(absolute)
    metrics.append(dict(mae=float(np.average(absolute, weights=weights.ravel())),
        p95=float(np.interp(.95, np.cumsum(weights.ravel()[order])/weights.sum(), absolute[order])),
        max_abs=float(absolute.max())))
print("Added strip Vp, Vs, rho, Zp relative %:", metrics, flush=True)


""" ## 6. Сохраняем FC и доказательства ## """

destination = root / output_name
assert destination != root / input_name
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
reloaded_raw = json.loads(destination.read_text())
assert reloaded_raw == output
reloaded = FCModel.decode(reloaded_raw)
assert np.array_equal(reloaded.mesh.nodes_xyz, all_xyz)
assert reloaded_raw["materials"] == source["materials"]
with destination.open("rb") as stream:
    output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
for name, expected in manifest.items():
    with (root/name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
old_polygons = np.array([[node_xyz[int(nid)][:2] for nid in element.nodes[:4]] for element in elements.values()
                         if any(node_xyz[int(nid)][1] > 2550 for nid in element.nodes)])
new_polygons = new_element_xyz[:, :4]
axes[0].add_collection(PolyCollection(old_polygons, facecolor=".94", edgecolor=".5", linewidth=.3))
axes[0].add_collection(PolyCollection(new_polygons, facecolor="#e2f2ec", edgecolor="#19805c", linewidth=.4))
axes[0].axhline(old_bottom, color="#dd8800", ls="--", label="Прежнее дно / внутренний стык 2650 м")
axes[0].axhline(new_bottom, color="#145caa", lw=2, label="Новое дно и ABC 2750 м")
axes[0].set(xlim=(5500, 6500), ylim=(2770, 2550), xlabel="x, м", ylabel="Глубина, м", title="Добавлены четыре ряда QUAD8; существующие узлы сохранены")
axes[0].legend(fontsize=9)
element_diff = np.sum(difference[..., 0]*weights, axis=1)/weights.sum(axis=1)
pc = PolyCollection(new_polygons, array=element_diff, cmap="RdBu_r", clim=(-.2, .2), rasterized=True)
axes[1].add_collection(pc)
axes[1].set(xlim=(0, 11750), ylim=(2750, 2650), xlabel="x, м", ylabel="Глубина, м",
            title=f"Новая полоса: (Vp FC − Vp SEG-Y) / Vp SEG-Y; MAE={metrics[0]['mae']:.5f}%")
fig.colorbar(pc, ax=axes[1], label="Разность, %", extend="both")
fig.savefig(root/f"img/{prefix}_bottom_extension.png", dpi=170)
plt.close(fig)
profile_depth = np.arange(2600, 2750.0001, .25)
profile_xy = np.column_stack((np.full_like(profile_depth, 6000), profile_depth))
profile_fc, _ = evaluate_fc(profile_xy)
profile_sgy = sample_sgy(profile_xy)
fig, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained", sharey=True)
for k, label, scale, ax in ((0, "Vp, км/с", 1000, axes[0]), (3, "P-импеданс, МПа·с/м", 1e6, axes[1])):
    ax.plot(profile_fc[:, k]/scale, profile_depth, label="FC: linear + nearest", color="#19805c")
    ax.plot(profile_sgy[:, k]/scale, profile_depth, label="SEG-Y: узлы, билинейно", ls="--", color="#145caa")
    ax.axhline(2650, color="#dd8800", ls=":")
    ax.set(xlabel=label, ylabel="Глубина, м", ylim=(2750, 2600))
    ax.grid(alpha=.2)
axes[0].legend(fontsize=9)
fig.suptitle("x=6000 м: материал через прежнее дно и в добавленных 100 м")
fig.savefig(root/f"img/{prefix}_bottom_material_x6000.png", dpi=170)
plt.close(fig)

report = dict(input_hashes=manifest, output_file=output_name, output_sha256=output_sha,
    old_bottom_m=old_bottom, new_bottom_m=new_bottom, added_rows=n_rows, row_height_m=row_height,
    added_nodes=len(added_ids), added_elements=len(added_eids), nodes_count=len(all_ids), elements_count=len(all_elements),
    stored_element_order=stored_order, validation_gll_order=7, validation_gll_count=int(gll_outside.size),
    gll_nearest_count=int(gll_outside.sum()), quadrature_count=int(weights.size),
    added_area_m2=float(weights.sum()), total_area_m2=float(11750*new_bottom),
    min_jacobian=jacobian_min, max_jacobian=jacobian_max,
    old_nodes_and_elements_preserved=True, materials_preserved=True, interface_edge_count=len(bottom_pairs),
    abc_edge_counts={side:len(pairs) for side,pairs in new_load_pairs.items()}, all_abc_external=True,
    updated_all_node_sets=updated_sets, properties=["Vp", "Vs", "rho", "Zp"], statistics_added_strip=metrics,
    nearest_area_percent=float(100*np.sum(weights*outside)/weights.sum()),
    reference="node_aligned SEG-Y bilinear Vp, Vs, rho; FC layerwise linear E, nu, rho with nearest outside",
    damping_note="FC mass damping table unchanged: starts at 2500 m, now evaluated through 2750 m; external solver may build its own sponge")
(root/f"data/{prefix}_bottom_extension.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
np.savez_compressed(root/f"data/{prefix}_bottom_extension.npz", added_node_ids=added_ids,
    added_node_xyz=added_xyz, added_element_ids=added_eids, added_connectivity=added_connectivity,
    vp_element_difference_percent=element_diff, profile_depth=profile_depth, profile_fc=profile_fc, profile_sgy=profile_sgy)
print(f"Saved and verified {destination}", flush=True)


""" ## Выводы ## """

"""
Расширили 75-й слой до 2750 м четырьмя рядами элементов. Исходная сетка и
материал сохранены. Стык конформный, поглощение находится только на новой
внешней границе. Проверили якобианы, восстановление материала, повторное
чтение FC и совпадение глубины с узловыми SEG-Y. Полноволновой расчёт не
запускался; условия поглощения Tesseral не реконструировались.
"""
