""" # MILEN SEM 2D. Часть II. Происхождение различий материала # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 4: восстановление двух поколений материала ## """

"""
Сопоставим промежуточный и расчётный FC с источником SEG-Y.
В памяти восстановим слои 1–5 и 75 из текущих и исторических каротажей.
Исходные данные и расчётные файлы не перезаписываем. Историю Git читаем
только командой show. Геометрия материала до 2750 м используется лишь
для воспроизведения таблиц; расчётная область остаётся 0–2650 м.
"""

import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import CloughTocher2DInterpolator
import segyio

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(root / "fc-model/src"))
from fc_model import FCModel

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pykrige")
prefix = "dev_2_5_4_material_provenance"
historical_revision = "5d8ed0e"
well_paths = ["src/well1_Backus_Elast_Vp_Vs_rhob.txt",
              "src/well_2_Backus_Elast_param_Vp_Vs_rhob.txt"]
manifest = {}
report = {"historical_revision": subprocess.check_output(
    ["git", "rev-parse", historical_revision], cwd=root, text=True).strip()}


""" ## 1. Читаем таблицы, не интерполируя значения ## """

def record_hash(relative_path):
    """Фиксирует SHA-256 прочитанного входного файла."""
    path = root / relative_path
    with path.open("rb") as stream:
        manifest[relative_path] = hashlib.file_digest(stream, "sha256").hexdigest()
    return path


grid_data = np.load(record_hash("data/dev_1_7_material_grids.npz"))
grid = grid_data["material_grid"]
xy = grid_data["coords_grid"]
layers = grid_data["layer_indexes_grid"]
lagrange = np.load(record_hash("data/dev_1_6_model_material_new.npz"))
coords, properties = lagrange["coords"], lagrange["properties"]
geometry = np.load(record_hash("data/dev_1_5_2_layer_boundaries_quadratic.npz"))


def read_fc_grid(relative_path):
    """Раскладывает E, nu, rho из таблиц FC по их точным координатам 5×5 м."""
    model = FCModel.load(str(record_hash(relative_path)))
    values = np.full_like(grid, np.nan)
    labels = np.full(layers.shape, -1)
    for mid, material in model.materials.items():
        data = {prop.name: prop.data for groups in material.properties.values()
                for group in groups for prop in group}
        fields = [data[name] for name in ("YOUNG_MODULE", "POISSON_RATIO", "DENSITY")]
        for k, field in enumerate(fields):
            assert field.type == "TABLE"
            columns = {column.type: column.value.data for column in field.table}
            points = np.column_stack((columns["TABULAR_X"], columns["TABULAR_Y"]))
            indices = np.rint((points - 2.5) / 5).astype(int)
            ix, iz = indices.T
            assert np.array_equal(xy[ix, iz], points)
            values[ix, iz, k] = field.value.data
            labels[ix, iz] = mid - 1
    assert np.isfinite(values).all()
    assert np.array_equal(labels, layers)
    return values


cartesian = read_fc_grid("data/dev_1_7_model_material_cartes.fc")
production = read_fc_grid("data/dev_2_2_milen2Do5pore_full.fc")
report["cartesian_fc_primitive_bitwise_equal_npz"] = bool(np.array_equal(cartesian, grid))
assert report["cartesian_fc_primitive_bitwise_equal_npz"]


def physical(values):
    """Преобразует E, nu, rho в Vp, Vs, rho в единицах СИ."""
    young, poisson, density = np.moveaxis(values, -1, 0)
    return np.stack((np.sqrt(young*(1-poisson)/(density*(1+poisson)*(1-2*poisson))),
                     np.sqrt(young/(2*density*(1+poisson))), density), axis=-1)


sgy = []
for name in ("Vp", "Vs", "Density"):
    with segyio.open(str(record_hash(f"data/dev_2_1_{name}_model.sgy")),
                     "r", ignore_geometry=True) as stream:
        sgy.append(np.asarray(stream.trace.raw[:], dtype=float))
sgy = np.stack(sgy, axis=-1)
sgy[..., 2] *= 1000
report["sgy_max_relative_percent_vs_npz"] = (
    100*np.max(abs(sgy/physical(grid)-1), axis=(0, 1))).tolist()
print("FC cartes == NPZ:", report["cartesian_fc_primitive_bitwise_equal_npz"], flush=True)
print("SEG-Y max relative %, Vp/Vs/rho:", report["sgy_max_relative_percent_vs_npz"], flush=True)


""" ## 2. Восстанавливаем исходные значения по двум версиям каротажей ## """

source_path = record_hash("dev_1_6_material_update.py")
tree = ast.parse(source_path.read_text())
interpolation_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                              and node.name == "interpolate_layer_properties")
namespace = dict(np=np, OrdinaryKriging=OrdinaryKriging, distances=geometry["distances"],
                 WELL1_DISTANCE=4250, WELL2_DISTANCE=7500)
# Исполняем только определение чистого интерполятора; генератор файлов не запускаем.
exec(compile(ast.Module(body=[interpolation_function], type_ignores=[]), str(source_path), "exec"), namespace)
record_hash("dev_1_7_material_euler.py")
starts = np.r_[0, np.where(np.diff(coords[0, :, 1]) == 0)[0] + 1]
ends = np.r_[starts[1:], coords.shape[1]]
assert len(starts) == 75
report["reconstruction"] = []
report["well_versions"] = []

for revision, smoothing in ((None, .75), (historical_revision, 0.0)):
    wells = []
    for name in well_paths:
        if revision is None:
            content = record_hash(name).read_text()
        else:
            content = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root, text=True)
        values = np.loadtxt(io.StringIO(content), skiprows=1)
        wells.append(values)
        report["well_versions"].append(dict(revision=revision or "working_tree", path=name,
            row_count=len(values), last_file_depth=float(values[-1, 0]),
            last_parser_depth=float(10*len(values)), last_vp=float(values[-1, 1]),
            text_sha256=hashlib.sha256(content.encode()).hexdigest()))
    mean_head = .5*(wells[0][:10, 1:] + wells[1][:10, 1:])
    for values in wells:
        values[:10, 1:] = smoothing*mean_head + (1-smoothing)*values[:10, 1:]
    for layer in (0, 1, 2, 3, 4, 74):
        info = {}
        for number, values in enumerate(wells, 1):
            depths = np.arange(10, 10*len(values)+1, 10)
            boundaries = np.r_[0, geometry[f"well{number}_depths"]]
            top, bottom = boundaries[layer:layer+2]
            mask = (depths >= top) & (depths <= bottom)
            info.update({f"well{number}_depths": depths[mask], f"well{number}_data": values[mask, 1:],
                         f"well{number}_top_depth": top, f"well{number}_bottom_depth": bottom})
        indices = np.arange(starts[layer], ends[layer])
        values = namespace["interpolate_layer_properties"](info, len(indices))
        vp, density, vs = np.moveaxis(values, -1, 0)
        density = density*1000
        poisson = (vp**2-2*vs**2)/(2*(vp**2-vs**2))
        primitive = np.stack((2*density*vs**2*(1+poisson), poisson, density), axis=-1)
        mask = layers == layer
        points = coords[:, indices].reshape(-1, 2)
        nodal = primitive.reshape(-1, 3)
        raster = np.column_stack([CloughTocher2DInterpolator(points, nodal[:, k],
            fill_value=nodal[:, k].mean())(xy[mask]) for k in range(3)])
        reference = grid[mask] if revision is None else production[mask]
        entry = dict(revision=revision or "working_tree", smoothing=smoothing, layer=layer+1,
            max_primitive_relative_percent=(100*np.max(abs(raster/reference-1), axis=0)).tolist(),
            max_physical_relative_percent=(100*np.max(abs(physical(raster)/physical(reference)-1), axis=0)).tolist())
        if revision is None:
            entry["max_nodal_relative_percent"] = (100*np.max(abs(primitive/properties[:, indices]-1), axis=(0, 1))).tolist()
        assert np.isfinite(raster).all()
        assert np.allclose(raster, reference, rtol=1e-10, atol=0)
        report["reconstruction"].append(entry)
        print(entry, flush=True)


""" ## 3. Сохраняем численное доказательство и историю изменений ## """

report["manifest"] = manifest
for layer in (1, 6, 75):
    mask = (layers == layer-1) & (xy[..., 1] < 2650)
    report[f"layer_{layer}_production_vs_cartesian_max_percent"] = (
        100*np.max(abs(physical(production[mask])/physical(grid[mask])-1), axis=0)).tolist()
(root / "data" / f"{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
history = subprocess.check_output(["git", "diff", historical_revision, "HEAD", "--", *well_paths], cwd=root, text=True)
(root / "data" / f"{prefix}_well_history.diff").write_text(history)


""" ## Выводы ## """

"""
Проверили точное соответствие промежуточного FC и NPZ, округление SEG-Y,
а также воспроизведение изменённых слоёв из двух поколений исходных данных.
Численные невязки восстановления записаны в JSON, изменения каротажей —
в отдельный diff. Источники, расчётные FC и SEG-Y оставлены без изменения.
"""
