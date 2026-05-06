import sys
import os
import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from semgpu.model.receiver import rec_steps_from_dt

from semgpu.model import SGModel, SGConditionModel, SGReceiverModel
from semgpu.solvers import SGLinElast2DNewmarkKimHybridNBSolver as SGSolver
from semgpu.solvers.lin_elast_2d_newmark_kim_hybrid_nb.constants import (
    # conditions
    COND_FAM_P1_DMP,
    COND_FAM_NN_DMP,
    COND_FAM_DN,
    COND_CODE_NN_DMP_MASS_DAMPING,
    COND_CODE_P1_DMP_ABC,
    COND_CODE_DN_UX,
    STATE_CODE_X,
    STATE_CODE_Y,
    STATE_CODE_VX,
    STATE_CODE_VY,
    STATE_CODE_KIN_ENERGY,
    STATE_CODE_F_INT_POT_ENERGY,
    STATE_CODE_F_EXT_POT_ENERGY,

    # receiver codes (fast GPU integration / derived)
)

from semgpu.tools.material import build_homogeneous, build_from_elements
from semgpu.model import SGPETBLineModel, SGPETBTriModel
from semgpu.tools.build_mesh import build_structured_eqtri_rectangle_mesh, build_structured_rectangle_mesh, build_structured_tri_rectangle_mesh
from semgpu.tools.lamb_conditions_common import add_source_condition_nearest_node

# visualization library (project-local)
from semgpu.tools.utils import drain_queue_until_solver_stops, fft_from_time_window, compose_status
from semgpu.tools.visualization import VisualizationSetup, FieldPlotConfig, TimeSeriesPlotConfig


"""
SEMGPU Test Card

id
    solver_suite: lin_elast_2d_newmark_kim_hybrid_nb
    geometry_suite: simple_rect
    test_name: test_plot_lamb_wave_layers
    class: plot

    Цель
        Распространение волнового пакета в прямоугольной области, состоящей из двух горизонтальных слоев.
        Наблюдаем отражение и преломление волн на границе раздела.
        Используется симметрия задачи (моделируется только правая половина).

    Модель
        Геометрия: прямоугольник (вытянутый), структурированная quad-сетка.
        Координаты: (0,0) - левый верхний угол. Ось Y направлена вверх (область y <= 0).
        Материал: двухслойный линейно-упругий (верхний слой "мягче")
        Параметры материала задаются через (rho, cp, cs) или (rho, E, nu).
        Настраивается из конфигурации

    Постановка
        Источник (один из типов, задается в конфигурации):
            - NN_POINT_FORCE_NEAREST / NN_POINT_FORCE_SAMPLING (узловая сила),
            - DN_VY_NEAREST / DN_VY_SAMPLING (закрепление скорости v_y),
            - DN_UY_NEAREST / DN_UY_SAMPLING (закрепление перемещения u_y),
            - P2_POINT_PRESSURE_NEAREST / P2_POINT_PRESSURE_SAMPLING (узловое давление в P2).
          Источник прикладывается в точке (0,0) (левый верхний угол).
        Граничные условия:
            - Левая граница (x=0): Симметрия (DN_UX = 0).
            - Правая и нижняя границы: ABC + массовое демпфирование (sponge).
            - Верхняя граница: Свободная.

Приемники 
  1) field:
     - X,Y,v_x,v_y (в стенде рисуется |v| на исходной сетке, без деформации).

Стенд (визуализация)
  Окно: поле |v|(x,t) без деформации + L(t), H(t) и их FFT.

На что смотреть
  - пространственную волновую картину |v|
  - уменьшение H(t) за счёт демпфирования и ABC
  - отсутствие неустойчивости при выбранном dt_safety

Артефакты
  save-mode: velocity_magnitude.png в output.dir

"""


# ============================================================
# CONFIG
# ============================================================

Lx = 300.0
Ly = 60.0
e_per_x = 0.5
order = 3
source_type = "NN_POINT_FORCE_NEAREST"

interface_y = 20

model_cfg: Dict[str, Any] = {
    "mesh": {"Lx": Lx, "Ly": Ly, "Ex":int(Lx*e_per_x), "Ey":int(Ly*e_per_x), "order": order, "distortion": 0.00},

    "material_top": {"rho": 1600, "cp": 1100, "cs": 330},
    # "material_top": {"rho": 2000, "cp": 1800, "cs": 540},
    "material_bottom": {"rho": 2000, "cp": 1800, "cs": 540},
    "interface_y": -interface_y,
    "solver": {"kim_beta": 0, "kim_gamma": 0.5},
    "time": {"t_begin": 0.0, "t_end": 0.15, "dt_safety": 0.1, "dt": 0.0},
    "source": {
        "type": source_type,  # NN_POINT_FORCE_NEAREST | NN_POINT_FORCE_SAMPLING | DN_VY_NEAREST | DN_VY_SAMPLING | DN_UY_NEAREST | DN_UY_SAMPLING | P2_POINT_PRESSURE_NEAREST | P2_POINT_PRESSURE_SAMPLING
        "amp": 1e7,
        "f0": 30,
        "duration": 0.5,
        "steps": 256,
    },
    "damping": {"enabled": True, "layer": 30.0, "reflection": 4.586801e-01, "power": 0.8498},
    "abc": {"enabled": True},

    "receivers": {
        "graph": {"dt_frame": 0.0001, "qmax": 100},
        "field": {
            "dt_frame": 0.001, "qmax": 1,
            "grid": {"Nx": 420, "Ny": 84}
        },
        "seismo": {
            "dt_frame": 0.0001, "qmax": 100,
            "grid": {"n_sensors": 400, "y": "top"}
        },
    },
}

stand_cfg: Dict[str, Any] = {
    "mode": "live",  # "live" | "save"

    "output": {
        "dir": os.path.join(os.path.dirname(__file__), f"plot_lamb_wave_layers_{order}_{e_per_x}_{interface_y}"),
        "velocity_magnitude_png": "velocity_magnitude.png",
        "lagrangian_png": "lagrangian.png",
        "hamiltonian_png": "hamiltonian.png",
    },

    "live": {
        "interval_ms": 20,
        "field_res": 420,
    },

    "axes": {
        "seismo_x": {
            "cmap": "RdBu",
            "scale_min": -0.1,
            "scale_max": 0.1,
            "scale_from_history": True,
        },
        "seismo_y": {
            "cmap": "RdBu",
            "scale_min": -0.1,
            "scale_max": 0.1,
            "scale_from_history": True,
        },
        "field_v": {
            "cmap": "viridis",
            "scale_min": 0.0,
            "scale_max": 0.1,
            "scale_from_history": False,
        },
        "dispersion_x": {
            "cmap": "jet",
            "scale_min": 0.0,
            "scale_max": None,
        },
        "dispersion_y": {
            "cmap": "jet",
            "scale_min": 0.0,
            "scale_max": None,
        },
        "L": {"window_min": 0.0, "window_max": 0.1},
        "H": {"window_min": 0.0, "window_max": 0.1},
    },
    "dispersion": {
        "enabled": True,
        "v_min": 0.0,
        "v_max": 2000.0,
        "v_steps": 512,
        "f_max": 150.0,
        "update_interval_frames": 20, # Update every N frames
    },
}


# ============================================================
# STATE (collector as dict)
# ============================================================

def build_state(model_cfg: Dict[str, Any], stand_cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = stand_cfg['mode']
    receivers_cfg = model_cfg.get("receivers", {})
    qmax_graph = int(receivers_cfg.get("graph", {}).get("qmax", 100))
    qmax_field = int(receivers_cfg.get("field", {}).get("qmax", 1))
    qmax_seismo = int(receivers_cfg.get("seismo", {}).get("qmax", 10))
    return {
        "model_cfg": model_cfg, # store for post-processing
        # invariants (filled in build_model)
        "nodes_coords": np.array([]),  # (2,N)
        "N": 0,
        "dt": 0,
        "steps": 0,
        "t_begin": 0,
        "t_end": 0,

        # graph history (used in save and live)
        "hist_graph": [],  # list[(t, E_kin, E_int, E_ext)]

        # field history (used in save and live)
        "hist_field": [],  # list[(t, |v|)]
        "hist_seismo": [],  # list[(t, vx_top, vy_top)]

        # latest frames + queues for live
        "q_graph": Queue(maxsize=qmax_graph) if mode == "live" else None,
        "q_field": Queue(maxsize=qmax_field) if mode == "live" else None,
        "q_seismo": Queue(maxsize=qmax_seismo) if mode == "live" else None,

        "last_field_time": 0,   # for frame_dt in status
        "last_field_frame": 0,  # for status
        "seismo_times": [],
        "seismo_values_x": [],
        "seismo_values_y": [],
    }




def _get_elastic_properties(props: Dict[str, float]) -> Dict[str, float]:
    """
    Returns dictionary with {rho, E, nu, cp, cs} derived from input props.
    Input can be (rho, cp, cs) or (rho, E, nu).
    """
    rho = float(props["rho"])
    
    if "cp" in props and "cs" in props:
        cp = float(props["cp"])
        cs = float(props["cs"])
        
        # Lame parameters
        mu = rho * cs * cs
        lam = rho * cp * cp - 2 * mu
        
        # Engineering constants
        if abs(lam + mu) < 1e-14:
            E = 0.0
            nu = 0.0
        else:
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            nu = lam / (2 * (lam + mu))
            
    elif "E" in props and "nu" in props:
        E = float(props["E"])
        nu = float(props["nu"])
        
        # Lame parameters
        mu = E / (2.0 * (1.0 + nu))
        # Avoid division by zero for nu=0.5 (incompressible)
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        if abs(denom) < 1e-14:
            lam = 1e14 # Large value
        else:
            lam = E * nu / denom
            
        cp = np.sqrt((lam + 2.0 * mu) / rho)
        cs = np.sqrt(mu / rho)
        
    else:
        raise ValueError(f"Material props must contain (rho, cp, cs) or (rho, E, nu). Got: {list(props.keys())}")
        
    return {"rho": rho, "E": E, "nu": nu, "cp": cp, "cs": cs}


def _build_mass_damping_layer_condition(
    *,
    mesh,
    material,
    layer: float,
    reflection: float,
    t_begin: float,
    t_end: float,
) -> Optional[SGConditionModel]:
    """
    Демпфирующий слой (sponge) как узловое условие mass damping:
        F = -alpha * M * v

    Закон (как в tests/old/seismo/test_work_milen_full.py):
        alpha = alpha_max * (l/L)^3
        alpha_max = (2*c_p/L) * ln(1/R)

    Слой ставится на правой и нижней границе. Верх свободный, левая граница под дирихле
    """
    L = float(layer)
    R = float(reflection)
    if L <= 0.0:
        return None
    if not (0.0 < R < 1.0):
        raise ValueError(f"reflection должен быть в (0,1), получено {R}")

    nodes_coords = np.asarray(mesh.p_nodes_coords, dtype=float)
    x = nodes_coords[:, 0]
    y = nodes_coords[:, 1]

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))

    l_right = np.clip(x - (x_max - L), 0.0, L)
    l_bottom = np.clip((y_min + L) - y, 0.0, L)

    # Symmetry on left boundary (x=0), so no damping there.
    in_layer = (l_right > 0.0) | (l_bottom > 0.0)
    node_ids = np.flatnonzero(in_layer).astype(int)
    if node_ids.size == 0:
        return None

    # Материал в SGMaterialModel задан на EN. Для каждого nid берем первый enid.
    n_nodes = int(mesh.p_nodes_coords.shape[0])
    e2_nodes = np.asarray(mesh.elements[(2, 2, 2)]["nodes"], dtype=int).ravel()
    first_enid = np.full(n_nodes, -1, dtype=int)
    for enid, nid in enumerate(e2_nodes):
        if first_enid[int(nid)] < 0:
            first_enid[int(nid)] = int(enid)

    enid_sel = first_enid[node_ids]
    if np.any(enid_sel < 0):
        raise RuntimeError("Damping layer: не удалось сопоставить nid -> enid (first_enid содержит -1).")

    names = material.names
    try:
        idx_rho = names.index("rho")
        idx_E = names.index("E")
        idx_nu = names.index("nu")
    except ValueError as exc:
        raise RuntimeError(f"Damping layer: материал должен содержать rho, E, nu. names={names}") from exc

    rho = material.values[enid_sel, idx_rho].astype(float)
    E = material.values[enid_sel, idx_E].astype(float)
    nu = np.clip(material.values[enid_sel, idx_nu].astype(float), 0.0, 0.49)

    c_p = np.sqrt(np.maximum(E / rho * ((1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)), 0.0))
    alpha_max = (2.0 * c_p / L) * float(np.log(1.0 / R))

    rr = l_right[node_ids] / L
    bb = l_bottom[node_ids] / L

    alpha_vals = alpha_max * (rr ** model_cfg['damping']['power'] + bb ** model_cfg['damping']['power'])
    good = np.isfinite(alpha_vals) & (alpha_vals > 0.0)
    if not np.any(good):
        return None

    node_ids = node_ids[good]
    alpha_vals = alpha_vals[good]

    data = np.zeros((node_ids.size, 1, 1), dtype=float)
    data[:, 0, 0] = alpha_vals

    return SGConditionModel(
        "boundary_mass_damping_layer",
        family_code=COND_FAM_NN_DMP,
        type_code=COND_CODE_NN_DMP_MASS_DAMPING,
        begin=t_begin,
        end=t_end,
        domain=node_ids,
        nodes_space="N",
        data=data,
        interp_degree=0,
    )

# ============================================================
# PRE: build_model
# ============================================================

def build_model(cfg: Dict[str, Any], state: Dict[str, Any]) -> SGModel:
    # unpack
    t_begin = float(cfg["time"]["t_begin"])
    t_end = float(cfg["time"]["t_end"])

    mcfg = cfg["mesh"]
    pet_line = SGPETBLineModel(name=f"Line{int(mcfg['order'])}", order=int(mcfg["order"]))
    pet_tri = SGPETBTriModel(name=f"Tri{int(mcfg['order'])}", order=int(mcfg["order"]), generatrix=(pet_line,))
    mesh = build_structured_tri_rectangle_mesh(
        Lx=float(mcfg["Lx"]),
        Ly=float(mcfg["Ly"]),
        Ex=int(mcfg["Ex"]),
        Ey=int(mcfg["Ey"]),
        cx=float(mcfg["Lx"]) / 2.0,
        cy=-float(mcfg["Ly"]) / 2.0,
        order=int(mcfg["order"]),
        pets=[pet_line, pet_tri],
        distortion=float(mcfg["distortion"]),
    )

    nodes_coords = mesh.p_nodes_coords.T  # (2,N)
    N = int(mesh.p_nodes_coords.shape[0])

    # Two layers material
    props_top = _get_elastic_properties(cfg["material_top"])
    props_bot = _get_elastic_properties(cfg["material_bottom"])
    
    # Pre-calculate values for performance
    vals_top = np.array([props_top["rho"], props_top["E"], props_top["nu"]], dtype=float)
    vals_bot = np.array([props_bot["rho"], props_bot["E"], props_bot["nu"]], dtype=float)
    
    print(f"[lamb_wave_layers] Top:    rho={props_top['rho']:.2f}, cp={props_top['cp']:.2f}, cs={props_top['cs']:.2f} => E={props_top['E']:.2f}, nu={props_top['nu']:.2f}")
    print(f"[lamb_wave_layers] Bottom: rho={props_bot['rho']:.2f}, cp={props_bot['cp']:.2f}, cs={props_bot['cs']:.2f} => E={props_bot['E']:.2f}, nu={props_bot['nu']:.2f}")

    y_min = float(np.min(mesh.p_nodes_coords[:, 1]))
    y_max = float(np.max(mesh.p_nodes_coords[:, 1]))
    
    if "interface_y" in cfg:
        y_interface = float(cfg["interface_y"])
    elif "layer_bottom_height" in cfg:
        y_interface = y_min + float(cfg["layer_bottom_height"])
    else:
        y_interface = (y_min + y_max) / 2.0
    
    print(f"[lamb_wave_layers] mesh Y range: [{y_min:.2f}, {y_max:.2f}]")
    print(f"[lamb_wave_layers] interface_y={y_interface:.2f}")

    def element_law(dim, eid, nodes_indices, nodes_coords):
        # nodes_coords: (K, 2)
        center_y = np.mean(nodes_coords[:, 1])
        vals = vals_top if center_y >= y_interface else vals_bot
        return np.tile(vals, (nodes_indices.shape[0], 1))

    material = build_from_elements(mesh, ("rho", "E", "nu"), element_law)

    scfg = cfg["solver"]
    solver = SGSolver(config=scfg)

    t_compile_start = time.perf_counter()
    model = SGModel(mesh, material, solver)
    t_compile_end = time.perf_counter()
    state["t_compile_start"] = t_compile_start
    state["t_compile_end"] = t_compile_end

    # dt and steps (for status)

    if cfg['time']["dt"]:
        model.dt = cfg['time']["dt"]
    elif cfg['time']["dt_safety"]:
        model.solver.compute_optimal_dt(
            dividend=t_end - t_begin, 
            safety=cfg['time']['dt_safety']
        )

    dt = model.dt

    steps_total = int(round((t_end - t_begin) / dt))

    state["nodes_coords"] = nodes_coords.copy()
    state["N"] = N
    state["dt"] = dt
    state["steps"] = steps_total
    state["t_begin"] = t_begin
    state["t_end"] = t_end

    # source: Ricker impulse in one node at top-center boundary
    source_cfg = cfg["source"]
    xs = mesh.p_nodes_coords[:, 0]
    ys = mesh.p_nodes_coords[:, 1]
    y_top = float(np.max(ys))
    y_tol = 1e-9 + 1e-6 * max(1.0, abs(y_top))
    top_nodes = np.where(np.abs(ys - y_top) <= y_tol)[0]
    if top_nodes.size == 0:
        top_nodes = np.arange(N, dtype=int)
    # Source at top-left corner (x=0, y=0)
    x_center_top = 0.0
    local_idx = int(np.argmin((xs[top_nodes] - x_center_top) ** 2))
    src_nid = int(top_nodes[local_idx])
    src_nodes = np.array([src_nid], dtype=int)
    
    # sensors
    all_nodes = np.arange(N, dtype=int)
    weights = np.ones(N, dtype=float)

    # framing
    receivers_cfg = cfg.get("receivers", {})
    dt_graph_frame = float(receivers_cfg.get("graph", {}).get("dt_frame", 0.0))
    dt_field_frame = float(receivers_cfg.get("field", {}).get("dt_frame", 0.0))
    dt_seismo_frame = float(receivers_cfg.get("seismo", {}).get("dt_frame", dt_field_frame))
    graph_steps = rec_steps_from_dt(t_begin, t_end, dt_graph_frame, dt)
    field_steps = rec_steps_from_dt(t_begin, t_end, dt_field_frame, dt)
    seismo_steps = rec_steps_from_dt(t_begin, t_end, dt_seismo_frame, dt)

    # NEW: Uniform receiver logic
    # 1. Seismo (Top Boundary)
    seismo_cfg = receivers_cfg.get("seismo", {})
    seismo_grid = seismo_cfg.get("grid", {})
    n_sensors = int(seismo_grid.get("n_sensors", 400))
    
    x_min = np.min(xs)
    x_max = np.max(xs)
    
    target_x = np.linspace(x_min, x_max, n_sensors)
    target_coords_seismo = np.zeros((n_sensors, 2))
    target_coords_seismo[:, 0] = target_x
    target_coords_seismo[:, 1] = y_top
    
    # Grid bounds (undistorted logical grid)
    Lx_val = float(mcfg["Lx"])
    Ly_val = float(mcfg["Ly"])
    Ex_val = int(mcfg["Ex"])
    Ey_val = int(mcfg["Ey"])
    cx_val = Lx_val / 2.0
    cy_val = -Ly_val / 2.0 # Mesh is generated with cy=-Ly/2 (top at 0)
    
    dx = Lx_val / Ex_val
    dy = Ly_val / Ey_val
    
    # Seismo Element Finding (triangular mesh: 2 triangles per quad cell)
    # Triangle 0 = (p00,p10,p11) lower-right; Triangle 1 = (p00,p11,p01) upper-left
    # Top edge belongs to triangle 1 of each cell in the top row.
    x_local = target_x - (cx_val - Lx_val/2.0)
    ix = np.floor(x_local / dx).astype(int)
    ix = np.clip(ix, 0, Ex_val - 1)
    iy = Ey_val - 1 # Top row
    
    eids_seismo = 2 * (iy * Ex_val + ix) + 1  # triangle 1 (upper-left) has the top edge
    
    elements_ids_seismo = np.zeros((n_sensors, 2), dtype=int)
    elements_ids_seismo[:, 0] = 2
    elements_ids_seismo[:, 1] = eids_seismo
    
    refs_seismo = mesh.phys_to_refs(elements_ids_seismo, target_coords_seismo)
    weights_seismo = mesh.sampling_shape(elements_ids_seismo, refs_seismo)
    
    nodes_flat = mesh.elements[(2, 2, 2)]['nodes']
    offsets = mesh.elements[(2, 2, 2)]['nodes_offsets']
    S = weights_seismo.shape[1]
    
    start_indices = offsets[eids_seismo]
    gather_indices = start_indices[:, None] + np.arange(S)[None, :]
    element_nodes_seismo = nodes_flat[gather_indices]
    
    sens_nodes_seismo = element_nodes_seismo.flatten()
    sens_weights_seismo = weights_seismo.flatten()
    sens_frame_seismo = np.repeat(np.arange(n_sensors), S)
    
    state["top_nodes"] = None 
    state["top_x"] = target_x
    
    # 2. Field (Regular Grid)
    field_cfg = receivers_cfg.get("field", {})
    field_grid = field_cfg.get("grid", {})
    f_Nx = int(field_grid.get("Nx", 100))
    f_Ny = int(field_grid.get("Ny", 20))
    
    f_x = np.linspace(x_min, x_max, f_Nx)
    f_y_min = float(np.min(ys))
    f_y_max = float(np.max(ys))
    f_y = np.linspace(f_y_min, f_y_max, f_Ny)
    
    f_X, f_Y = np.meshgrid(f_x, f_y) # (Ny, Nx)
    f_coords = np.column_stack((f_X.ravel(), f_Y.ravel())) # (Ny*Nx, 2)
    n_field_points = f_coords.shape[0]
    
    # Field Element Finding (triangular mesh: 2 triangles per quad cell)
    # Diagonal from p00(bottom-left) to p11(top-right) splits each cell:
    #   Triangle 0 (p00,p10,p11): local_y <= local_x (lower-right)
    #   Triangle 1 (p00,p11,p01): local_y >  local_x (upper-left)
    fx_local = f_coords[:, 0] - (cx_val - Lx_val/2.0)
    fy_local = f_coords[:, 1] - (cy_val - Ly_val/2.0)
    
    fix = np.floor(fx_local / dx).astype(int)
    fiy = np.floor(fy_local / dy).astype(int)
    fix = np.clip(fix, 0, Ex_val - 1)
    fiy = np.clip(fiy, 0, Ey_val - 1)
    
    # Determine which triangle within the cell
    cell_lx = (fx_local - fix * dx) / dx  # local x in [0, 1]
    cell_ly = (fy_local - fiy * dy) / dy  # local y in [0, 1]
    tri_idx = (cell_ly > cell_lx).astype(int)  # 0 = lower-right, 1 = upper-left
    
    eids_field = 2 * (fiy * Ex_val + fix) + tri_idx
    
    elements_ids_field = np.zeros((n_field_points, 2), dtype=int)
    elements_ids_field[:, 0] = 2
    elements_ids_field[:, 1] = eids_field
    
    refs_field = mesh.phys_to_refs(elements_ids_field, f_coords)
    weights_field = mesh.sampling_shape(elements_ids_field, refs_field)
    
    start_indices_f = offsets[eids_field]
    gather_indices_f = start_indices_f[:, None] + np.arange(S)[None, :]
    element_nodes_field = nodes_flat[gather_indices_f]
    
    sens_nodes_field = element_nodes_field.flatten()
    sens_weights_field = weights_field.flatten()
    sens_frame_field = np.repeat(np.arange(n_field_points), S)
    
    state["field_grid"] = {
        "Nx": f_Nx, "Ny": f_Ny, 
        "x": f_x, "y": f_y,
        "coords": f_coords
    }
    
    src_cfg = dict(source_cfg)
    src_cfg["x"] = float(x_center_top)
    src_cfg["y"] = float(y_top)
    src_info = add_source_condition_nearest_node(
        model=model,
        mesh=mesh,
        source_cfg=src_cfg,
        t_begin=t_begin,
        dt=dt,
        x_min=float(np.min(xs)),
        x_max=float(np.max(xs)),
        y_min=float(np.min(ys)),
        y_max=float(np.max(ys)),
    )
    src_nid = int(src_info["src_nid"])
    src_nodes = np.asarray(src_info["src_nodes"], dtype=int)

    # Symmetry on left boundary (tag 4)
    # nodes_tags: (N, 3) [mesh_tag, geom_tag, entity_tag]
    # entity_tag=4 is Left boundary
    left_nodes_mask = (mesh.p_nodes_tags[:, 2] == 4)
    left_nodes = np.flatnonzero(left_nodes_mask).astype(int)
    if left_nodes.size > 0:
        model.conditions.append(
            SGConditionModel(
                "symmetry_left",
                family_code=COND_FAM_DN,
                type_code=COND_CODE_DN_UX,
                begin=t_begin,
                end=t_end,
                domain=left_nodes,
                nodes_space="N",
                data=np.zeros((left_nodes.size, 1, 1), dtype=float),
                interp_degree=0,
            )
        )

    # damping layer near right/bottom boundaries (left is symmetry)
    damping_cfg = cfg.get("damping", {})
    if bool(damping_cfg.get("enabled", True)):
        cond_layer = _build_mass_damping_layer_condition(
            mesh=mesh,
            material=material,
            layer=float(damping_cfg.get("layer", 0.0)),
            reflection=float(damping_cfg.get("reflection", 1e-6)),
            t_begin=t_begin,
            t_end=t_end,
        )
        if cond_layer is not None:
            model.conditions.append(cond_layer)

    # ABC on right/bottom boundaries (left is symmetry)
    # Using bottom material for reference (approximation for sides)
    abc_data_value = np.array([props_bot["rho"] * props_bot["cp"], props_bot["rho"] * props_bot["cs"]], dtype=float)

    abc_cfg = cfg.get("abc", {})
    if bool(abc_cfg.get("enabled", True)):
        for tag_id, edge_name in ((2, "right"), (1, "bottom")):
            edge_elems = np.asarray(mesh.elements_filter(dim=1, geom_tag=tag_id), dtype=int)
            if edge_elems.size == 0:
                continue
            e1_offsets = np.asarray(mesh.elements[(2, 2, 1)]["nodes_offsets"], dtype=int)
            enids = []
            for eid in edge_elems:
                if eid < 0 or eid + 1 >= e1_offsets.size:
                    continue
                start = e1_offsets[eid]
                end = e1_offsets[eid + 1]
                enids.extend(range(start, end))
            edge_enids = np.asarray(enids, dtype=int)
            if edge_enids.size == 0:
                continue
            abc_data = np.tile(abc_data_value.reshape(1, 1, 2), (edge_enids.size, 1, 1))
            model.conditions.append(
                SGConditionModel(
                    f"abc_{edge_name}",
                    family_code=COND_FAM_P1_DMP,
                    type_code=COND_CODE_P1_DMP_ABC,
                    begin=t_begin,
                    end=t_end,
                    domain=edge_enids,
                    nodes_space="P221",
                    data=abc_data,
                    interp_degree=0,
                )
            )

    # sensors
    all_nodes = np.arange(N, dtype=int)
    weights = np.ones(N, dtype=float)

    # framing
    receivers_cfg = cfg.get("receivers", {})
    dt_graph_frame = float(receivers_cfg.get("graph", {}).get("dt_frame", 0.0))
    dt_field_frame = float(receivers_cfg.get("field", {}).get("dt_frame", 0.0))
    dt_seismo_frame = float(receivers_cfg.get("seismo", {}).get("dt_frame", dt_field_frame))
    graph_steps = rec_steps_from_dt(t_begin, t_end, dt_graph_frame, dt)
    field_steps = rec_steps_from_dt(t_begin, t_end, dt_field_frame, dt)
    seismo_steps = rec_steps_from_dt(t_begin, t_end, dt_seismo_frame, dt)

    state["seismo_steps"] = seismo_steps
    state["_last_progress_pct"] = -1
    state["_run_start"] = None

    # Receiver: integrated energies
    def on_graph(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        item = (float(frame_time), float(frame[0]), float(frame[1]), float(frame[2]))
        state["hist_graph"].append(item)
        q = state.get("q_graph")
        if q is not None:
            q.put(item, block=True)

    rec_graph = SGReceiverModel(
        "graph",
        begin=t_begin,
        end=t_end,
        steps=graph_steps,
        components=np.array([STATE_CODE_KIN_ENERGY, STATE_CODE_F_INT_POT_ENERGY, STATE_CODE_F_EXT_POT_ENERGY], dtype=int),
        sens_nodes=all_nodes,
        sens_frame=np.zeros(N, dtype=int),
        sens_weights=weights,
        onframe=on_graph,
    )

    # Receiver: FIELD |v| on interpolated grid
    def on_field(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        # frame is (4, n_field_points) -> X, Y, VX, VY
        data = frame.reshape(4, n_field_points)
        vx = data[2, :].astype(float)
        vy = data[3, :].astype(float)
        vmag = np.sqrt(vx * vx + vy * vy)
        
        item = (frame_time, vmag)
        state["hist_field"].append(item)
        q = state.get("q_field")
        if q is not None:
            q.put(item, block=True)
        # print(frame_time) # verbose
        state["last_field_time"] = float(frame_time)
        state["last_field_frame"] = int(frame_id)

    rec_field = SGReceiverModel(
        "field",
        begin=t_begin,
        end=t_end,
        steps=field_steps,
        components=np.array([STATE_CODE_X, STATE_CODE_Y, STATE_CODE_VX, STATE_CODE_VY], dtype=int),
        sens_nodes=sens_nodes_field,
        sens_frame=sens_frame_field,
        sens_weights=sens_weights_field,
        onframe=on_field,
    )

    def on_seismo(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        data = frame.reshape(2, n_sensors)
        vx_top = data[0, :].astype(float)
        vy_top = data[1, :].astype(float)
        item = (float(frame_time), vx_top, vy_top)
        state["hist_seismo"].append(item)
        q = state.get("q_seismo")
        if q is not None:
            q.put(item, block=True)
        else:
            # save mode: print progress
            if state["_run_start"] is None:
                state["_run_start"] = time.perf_counter()
            t_span = state["t_end"] - state["t_begin"]
            t_progress = (frame_time - state["t_begin"]) / t_span if t_span > 0 else 0.0
            pct = int(t_progress * 100)
            if pct >= state["_last_progress_pct"] + 5:
                state["_last_progress_pct"] = pct
                elapsed = time.perf_counter() - state["_run_start"]
                eta = elapsed / t_progress * (1.0 - t_progress) if t_progress > 1e-6 else 0.0
                print(f"[lamb_wave] {pct:3d}%  t={frame_time:.4f}s  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    rec_seismo = SGReceiverModel(
        "seismo",
        begin=t_begin,
        end=t_end,
        steps=seismo_steps,
        components=np.array([STATE_CODE_VX, STATE_CODE_VY], dtype=int),
        sens_nodes=sens_nodes_seismo,
        sens_frame=sens_frame_seismo,
        sens_weights=sens_weights_seismo,
        onframe=on_seismo,
    )

    model.receivers.append(rec_graph)
    model.receivers.append(rec_field)
    model.receivers.append(rec_seismo)

    print(f"[lamb_wave] built: N={N}, dt={dt:.6e}, steps={steps_total}, field_steps={field_steps}")
    return model


# ============================================================
# POST: build_stand
# ============================================================


def _compute_dispersion_image(times, data_vals, top_x, disp_cfg):
    """
    Helper to compute f-v image from time-space data.
    data_vals: (Nt, Nx)
    """
    if len(times) < 2 or data_vals.size == 0:
        return None, None, None

    # Remove DC
    data = data_vals - np.mean(data_vals, axis=0)
    
    Nt, Nx = data.shape
    dt = times[1] - times[0]
    if dt <= 1e-9:
        return None, None, None

    x = top_x - top_x[0]
    
    v_min = float(disp_cfg.get("v_min", 0.0))
    v_max = float(disp_cfg.get("v_max", 2000.0))
    nv = int(disp_cfg.get("v_steps", 200))
    f_max = float(disp_cfg.get("f_max", 100.0))
    
    # FFT
    freqs = np.fft.rfftfreq(Nt, dt)
    spec = np.fft.rfft(data, axis=0) # (Nf, Nx)
    
    mask_f = (freqs > 0) & (freqs <= f_max)
    freqs = freqs[mask_f]
    spec = spec[mask_f, :]
    
    vs = np.linspace(v_min, v_max, nv)
    image = np.zeros((freqs.size, nv))
    
    # Phase Shift
    # Vectorized loop
    for i, v in enumerate(vs):
        if v < 1e-3: continue
        arg = 2.0 * np.pi * freqs[:, None] * (x[None, :] / v)
        shift = np.exp(1j * arg)
        stack = np.sum(spec * shift, axis=1)
        image[:, i] = np.abs(stack)
        
    if np.max(image) > 0:
        image = image / np.max(image)
        
    return freqs, vs, image


def _compute_and_save_dispersion(state, disp_cfg, out_dir, model_cfg):
    print("[lamb_wave_layers/dispersion] Computing f-v diagram for save...")
    hist = state.get("hist_seismo", [])
    if not hist:
        print("[lamb_wave_layers/dispersion] No seismo history found.")
        return
    
    # Check if we have data
    if len(hist) < 2:
        return

    times = np.array([h[0] for h in hist])
    top_x = state.get("top_x")
    if top_x is None or len(top_x) < 2:
        return

    # Compute for both components if present in history
    # hist item: (t, vx, vy)
    for comp_idx, comp_name in [(1, "vx"), (2, "vy")]:
        data = np.vstack([h[comp_idx] for h in hist])
        freqs, vs, image = _compute_dispersion_image(times, data, top_x, disp_cfg)
        
        if image is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(image.T, origin='lower', aspect='auto',
                       extent=[freqs[0], freqs[-1], vs[0], vs[-1]],
                       cmap='jet', interpolation='bilinear')
            
            plt.colorbar(label="Amplitude")
            plt.title(f"Dispersion Image ({comp_name.upper()}) | Phase Shift Method")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase Velocity (m/s)")
            
            # Theoretical lines
            props = _get_elastic_properties(model_cfg["material_top"])
            cs = props["cs"]
            cr = cs * (0.87 + 1.12*props["nu"]) / (1+props["nu"])
            plt.axhline(cr, color='white', linestyle='--', linewidth=2, label=f"Rayleigh Top ({cr:.0f})")
            plt.axhline(cs, color='white', linestyle=':', linewidth=1, label=f"Shear Top ({cs:.0f})")
            
            props_bot = _get_elastic_properties(model_cfg["material_bottom"])
            plt.axhline(props_bot["cs"], color='gray', linestyle='-.', linewidth=1, label=f"Shear Bot ({props_bot['cs']:.0f})")
            plt.legend(loc='upper right')
            
            fname = f"dispersion_{comp_name}.png"
            path = os.path.join(out_dir, fname)
            plt.savefig(path)
            plt.close()
            print(f"[lamb_wave_layers/dispersion] Saved: {path}")


def _save_segy(filepath: str, data: np.ndarray, times: np.ndarray, x_coords: np.ndarray) -> None:
    """Save (Nt, Nx) seismogram array as SEG-Y. times in seconds, x_coords in meters."""
    import segyio
    Nt, Nx = data.shape
    dt_us = max(1, int(round((times[1] - times[0]) * 1e6))) if Nt > 1 else 1000
    t0_ms = int(round(float(times[0]) * 1000.0)) if times.size > 0 else 0
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1  # IBM 4-byte float
    spec.samples = np.arange(Nt, dtype=np.float32)
    spec.tracecount = Nx
    with segyio.create(filepath, spec) as f:
        f.bin.update(hdt=dt_us, dto=dt_us)
        for i in range(Nx):
            f.trace[i] = data[:, i].astype(np.float32)
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: i + 1,
                segyio.TraceField.GroupX: int(round(x_coords[i] * 1000)),
                segyio.TraceField.SourceGroupScalar: -1000,
                segyio.TraceField.DelayRecordingTime: t0_ms,
            }


def build_stand(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(cfg["mode"]).lower()
    assert mode in ("live", "save")
    if mode == "save":
        out_dir = os.makedirs(cfg["output"]["dir"], exist_ok=True)
        out_dir = cfg["output"]["dir"]

        def start_stand(_handle):
            return None

        def finalize_stand():
            hist_graph = state["hist_graph"]
            hist_field = state["hist_field"]
            hist_seismo = state["hist_seismo"]
            
            if not hist_field and not hist_graph and not hist_seismo:
                print("[lamb_wave/save] no frames collected")
                return {}

            # 1. Velocity Magnitude (Scatter)
            if hist_field:
                field_grid = state.get("field_grid")
                plt.figure(figsize=(8, 6))
                field_v_cfg = cfg.get("axes", {}).get("field_v", {})
                cmap_v = str(field_v_cfg.get("cmap", "viridis"))
                v_scale_min = field_v_cfg.get("scale_min")
                v_scale_max = field_v_cfg.get("scale_max")
                imshow_kwargs: Dict[str, Any] = {}
                if v_scale_min is not None:
                    imshow_kwargs["vmin"] = float(v_scale_min)
                if v_scale_max is not None:
                    imshow_kwargs["vmax"] = float(v_scale_max)
                
                if field_grid:
                    # Use imshow for regular grid
                    f_Nx = field_grid["Nx"]
                    f_Ny = field_grid["Ny"]
                    f_x = field_grid["x"]
                    f_y = field_grid["y"]
                    extent = [f_x[0], f_x[-1], f_y[0], f_y[-1]]
                    
                    _, vmag = hist_field[-1]
                    # vmag is flat, reshape
                    if vmag.size == f_Nx * f_Ny:
                        grid_data = vmag.reshape(f_Ny, f_Nx)
                        plt.imshow(grid_data, origin='lower', extent=extent, aspect='equal', cmap=cmap_v, **imshow_kwargs)
                    else:
                        print(f"[lamb_wave/save] vmag size mismatch: {vmag.size} != {f_Nx}*{f_Ny}")
                else:
                    # Fallback to scatter
                    nodes_coords = state["nodes_coords"]
                    _, vmag = hist_field[-1]
                    x_node = nodes_coords[0, :]
                    y_node = nodes_coords[1, :]
                    plt.scatter(x_node, y_node, c=vmag, s=4, cmap=cmap_v, **imshow_kwargs)
                    plt.gca().set_aspect("equal", "box")
                
                plt.colorbar(label="|v|")
                plt.title("Velocity Magnitude |v|(x,t_last)")
                plt.xlabel("x")
                plt.ylabel("y")
                p = os.path.join(out_dir, str(cfg["output"]["velocity_magnitude_png"]))
                plt.tight_layout()
                plt.savefig(p)
                plt.close()
                print(f"[lamb_wave/save] saved: {p}")
            
            # 2. Energy Graphs (L and H)
            if hist_graph:
                times = [item[0] for item in hist_graph]
                ekin = np.array([item[1] for item in hist_graph])
                eint = np.array([item[2] for item in hist_graph])
                eext = np.array([item[3] for item in hist_graph])
                
                epi = eint + eext
                L = ekin - epi
                H = ekin + epi
                
                # Save L
                plt.figure(figsize=(10, 6))
                plt.plot(times, L, color='purple', label='L = T - Pi')
                plt.grid(True, alpha=0.3)
                plt.xlabel("Time [s]")
                plt.ylabel("Energy")
                plt.title("Lagrangian L(t)")
                plt.legend()
                p_l = os.path.join(out_dir, "energy_L.png")
                plt.savefig(p_l)
                plt.close()
                print(f"[lamb_wave/save] saved: {p_l}")
                
                # Save H
                plt.figure(figsize=(10, 6))
                plt.plot(times, H, color='green', label='H = T + Pi')
                plt.grid(True, alpha=0.3)
                plt.xlabel("Time [s]")
                plt.ylabel("Energy")
                plt.title("Hamiltonian H(t)")
                plt.legend()
                p_h = os.path.join(out_dir, "energy_H.png")
                plt.savefig(p_h)
                plt.close()
                print(f"[lamb_wave/save] saved: {p_h}")

            # 3. Seismograms (Vx, Vy)
            if hist_seismo:
                # hist_seismo: list of (t, vx[], vy[])
                times_s = np.array([h[0] for h in hist_seismo])
                vx_data = np.vstack([h[1] for h in hist_seismo]) # (Nt, Nx)
                vy_data = np.vstack([h[2] for h in hist_seismo]) # (Nt, Nx)
                
                top_x = state.get("top_x")
                if top_x is not None and top_x.size > 0:
                    x_min, x_max = float(np.min(top_x)), float(np.max(top_x))
                    t_min, t_max = float(times_s[0]), float(times_s[-1])
                    extent = [x_min, x_max, t_min, t_max]
                    
                    # Save Seismo Vx
                    plt.figure(figsize=(10, 8))
                    seis_x_cfg = cfg.get("axes", {}).get("seismo_x", {})
                    cmap = str(seis_x_cfg.get("cmap", "RdBu"))
                    sx_min = seis_x_cfg.get("scale_min")
                    sx_max = seis_x_cfg.get("scale_max")
                    if sx_min is not None and sx_max is not None:
                        vm_min_x, vm_max_x = float(sx_min), float(sx_max)
                    else:
                        vm = np.max(np.abs(vx_data)) if np.max(np.abs(vx_data)) > 0 else 1.0
                        vm_min_x, vm_max_x = -vm, vm
                    plt.imshow(vx_data, origin='lower', aspect='auto', extent=extent, cmap=cmap, vmin=vm_min_x, vmax=vm_max_x)
                    plt.colorbar(label="Vx")
                    plt.title("Seismogram Vx(x, t)")
                    plt.xlabel("x")
                    plt.ylabel("t")
                    p_vx = os.path.join(out_dir, "seismogram_vx.png")
                    plt.savefig(p_vx)
                    plt.close()
                    print(f"[lamb_wave/save] saved: {p_vx}")
                    
                    # Save Seismo Vy
                    plt.figure(figsize=(10, 8))
                    seis_y_cfg = cfg.get("axes", {}).get("seismo_y", {})
                    cmap = str(seis_y_cfg.get("cmap", "RdBu"))
                    sy_min = seis_y_cfg.get("scale_min")
                    sy_max = seis_y_cfg.get("scale_max")
                    if sy_min is not None and sy_max is not None:
                        vm_min_y, vm_max_y = float(sy_min), float(sy_max)
                    else:
                        vm = np.max(np.abs(vy_data)) if np.max(np.abs(vy_data)) > 0 else 1.0
                        vm_min_y, vm_max_y = -vm, vm
                    plt.imshow(vy_data, origin='lower', aspect='auto', extent=extent, cmap=cmap, vmin=vm_min_y, vmax=vm_max_y)
                    plt.colorbar(label="Vy")
                    plt.title("Seismogram Vy(x, t)")
                    plt.xlabel("x")
                    plt.ylabel("t")
                    p_vy = os.path.join(out_dir, "seismogram_vy.png")
                    plt.savefig(p_vy)
                    plt.close()
                    print(f"[lamb_wave/save] saved: {p_vy}")

                    # Save SEG-Y files
                    p_segy_vx = os.path.join(out_dir, "seismogram_vx.segy")
                    _save_segy(p_segy_vx, vx_data, times_s, top_x)
                    print(f"[lamb_wave/save] saved: {p_segy_vx}")

                    p_segy_vy = os.path.join(out_dir, "seismogram_vy.segy")
                    _save_segy(p_segy_vy, vy_data, times_s, top_x)
                    print(f"[lamb_wave/save] saved: {p_segy_vy}")

            # 4. Dispersion
            disp_cfg = cfg.get("dispersion", {})
            if disp_cfg.get("enabled", False):
                 _compute_and_save_dispersion(state, disp_cfg, out_dir, state["model_cfg"])
            
            return {"frames_graph": len(hist_graph), "frames_field": len(hist_field)}

        return {"start": start_stand, "finalize": finalize_stand}

    nodes_coords = state["nodes_coords"]
    dt = float(state["dt"])
    steps_total = int(state["steps"])
    t_begin = float(state["t_begin"])
    t_end = float(state["t_end"])
    field_res = int(cfg["live"]["field_res"])

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 0.8], hspace=0.35, wspace=0.20)
    
    # Top: Field (full width)
    ax_field = fig.add_subplot(gs[0, :])
    
    # Middle: Seismo X and Seismo Y
    ax_seis_x = fig.add_subplot(gs[1, 0])
    ax_seis_y = fig.add_subplot(gs[1, 1])
    
    # Bottom: Dispersion X and Dispersion Y
    ax_disp_x = fig.add_subplot(gs[2, 0])
    ax_disp_y = fig.add_subplot(gs[2, 1])
    
    # Energy: L and H
    ax_l = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])
    
    plt.subplots_adjust(bottom=0.05, top=0.95)
    vis = VisualizationSetup(
        fig=fig,
        axes=np.array([[ax_field], [ax_seis_x, ax_seis_y], [ax_disp_x, ax_disp_y], [ax_l, ax_h]], dtype=object),
        nodes_coords=nodes_coords,
        shape=None,
    )

    top_x = state.get("top_x", np.array([], dtype=float))
    
    # Seismo X
    seis_cfg_x = cfg.get("axes", {}).get("seismo_x", {})
    seis_data_x = np.zeros((1, max(1, top_x.size)), dtype=float)
    seis_im_x = ax_seis_x.imshow(
        seis_data_x,
        origin="lower",
        aspect="auto",
        cmap=str(seis_cfg_x.get("cmap", "RdBu")),
        extent=[float(np.min(top_x)) if top_x.size else 0.0,
                float(np.max(top_x)) if top_x.size else 1.0,
                t_begin,
                t_begin + 1e-6],
    )
    ax_seis_x.set_title("Seismogram V_x(x, t)")
    ax_seis_x.set_xlabel("x")
    ax_seis_x.set_ylabel("t")
    plt.colorbar(seis_im_x, ax=ax_seis_x)

    # Seismo Y
    seis_cfg_y = cfg.get("axes", {}).get("seismo_y", {})
    seis_data_y = np.zeros((1, max(1, top_x.size)), dtype=float)
    seis_im_y = ax_seis_y.imshow(
        seis_data_y,
        origin="lower",
        aspect="auto",
        cmap=str(seis_cfg_y.get("cmap", "RdBu")),
        extent=[float(np.min(top_x)) if top_x.size else 0.0,
                float(np.max(top_x)) if top_x.size else 1.0,
                t_begin,
                t_begin + 1e-6],
    )
    ax_seis_y.set_title("Seismogram V_y(x, t)")
    ax_seis_y.set_xlabel("x")
    ax_seis_y.set_ylabel("t")
    plt.colorbar(seis_im_y, ax=ax_seis_y)
    
    # Dispersion Config
    disp_cfg = cfg.get("dispersion", {})
    disp_enabled = bool(disp_cfg.get("enabled", False))
    v_min = float(disp_cfg.get("v_min", 0.0))
    v_max = float(disp_cfg.get("v_max", 2000.0))
    f_max = float(disp_cfg.get("f_max", 150.0))
    disp_update_interval = int(disp_cfg.get("update_interval_frames", 20))
    
    cs = 0.0
    cr = 0.0
    # Dispersion X
    disp_im_x = None
    if disp_enabled:
        ax_disp_x.set_title("Dispersion V_x (f-v)")
        ax_disp_x.set_xlabel("Frequency (Hz)")
        ax_disp_x.set_ylabel("Phase Velocity (m/s)")
        # Empty placeholder
        disp_im_x = ax_disp_x.imshow(
            np.zeros((10, 10)),
            origin="lower",
            aspect="auto",
            cmap=str(cfg.get("axes", {}).get("dispersion_x", {}).get("cmap", "jet")),
            extent=[0, f_max, v_min, v_max],
            interpolation='bilinear'
        )
        plt.colorbar(disp_im_x, ax=ax_disp_x)
        
        # Theoretical lines
        props_top = _get_elastic_properties(state["model_cfg"]["material_top"])
        cs = props_top["cs"]
        cr = cs * (0.87 + 1.12*props_top["nu"]) / (1+props_top["nu"])
        ax_disp_x.axhline(cr, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label="Rayleigh Top")
        ax_disp_x.axhline(cs, color='white', linestyle=':', linewidth=1.5, alpha=0.7, label="Shear Top")
        ax_disp_x.legend(fontsize='x-small', loc='upper right')

    # Dispersion Y
    disp_im_y = None
    if disp_enabled:
        ax_disp_y.set_title("Dispersion V_y (f-v)")
        ax_disp_y.set_xlabel("Frequency (Hz)")
        ax_disp_y.set_ylabel("Phase Velocity (m/s)")
        disp_im_y = ax_disp_y.imshow(
            np.zeros((10, 10)),
            origin="lower",
            aspect="auto",
            cmap=str(cfg.get("axes", {}).get("dispersion_y", {}).get("cmap", "jet")),
            extent=[0, f_max, v_min, v_max],
            interpolation='bilinear'
        )
        plt.colorbar(disp_im_y, ax=ax_disp_y)
        
        # Theoretical lines same as X
        ax_disp_y.axhline(cr, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_disp_y.axhline(cs, color='white', linestyle=':', linewidth=1.5, alpha=0.7)

    field_grid = state.get("field_grid")
    field_im = None
    field_v = None

    if field_grid:
        # Use imshow for regular grid from receiver
        f_Nx = field_grid["Nx"]
        f_Ny = field_grid["Ny"]
        f_x = field_grid["x"]
        f_y = field_grid["y"]
        extent = [f_x[0], f_x[-1], f_y[0], f_y[-1]]
        
        ax_field.set_title("|v|(x,t)")
        ax_field.set_xlabel("x")
        ax_field.set_ylabel("y")
        
        field_im = ax_field.imshow(
            np.zeros((f_Ny, f_Nx)),
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=str(cfg.get("axes", {}).get("field_v", {}).get("cmap", "viridis")),
            vmin=float(cfg.get("axes", {}).get("field_v", {}).get("scale_min", 0.0)),
            vmax=float(cfg.get("axes", {}).get("field_v", {}).get("scale_max", 0.1)),
            interpolation='bilinear'
        )
        plt.colorbar(field_im, ax=ax_field)
    else:
        # Fallback to FieldPlot (scatter/interpolation)
        field_v = vis.add_field_plot(
            ax_field,
            width=field_res,
            height=field_res,
            config=FieldPlotConfig(
                title="|v|(x,t)",
                cmap=str(cfg.get("axes", {}).get("field_v", {}).get("cmap", "viridis")),
                scale_min=cfg.get("axes", {}).get("field_v", {}).get("scale_min"),
                scale_max=cfg.get("axes", {}).get("field_v", {}).get("scale_max"),
                scale_from_history=bool(cfg.get("axes", {}).get("field_v", {}).get("scale_from_history", False)),
            ),
        )

    # --- 4. Energy L, H ---
    axis_l = cfg.get("axes", {}).get("L", {})
    axis_h = cfg.get("axes", {}).get("H", {})
    
    plot_l = vis.add_time_series(
        ax_l,
        config=TimeSeriesPlotConfig(
            title=r"$L(t) = T - \Pi$",
            xlabel="Time [s]",
            ylabel="Energy",
            window_min=axis_l.get("window_min", 0.0),
            window_max=axis_l.get("window_max", 0.2),
        )
    )
    plot_l.add_series("L", color="purple")
    
    plot_h = vis.add_time_series(
        ax_h,
        config=TimeSeriesPlotConfig(
            title=r"$H(t) = T + \Pi$",
            xlabel="Time [s]",
            ylabel="Energy",
            window_min=axis_h.get("window_min", 0.0),
            window_max=axis_h.get("window_max", 0.2),
        )
    )
    plot_h.add_series("H", color="green")
    
    vis.set_time_text(text="", x=0.5, y=0.02, fontsize=10)
    runtime = {
        "wall_start": time.time(),
        "ui_prev": None,
        "last_t": t_begin,
        "last_frame_dt": 0.0,
        "frame_counter": 0,
    }

    def update(_):
        runtime["frame_counter"] += 1
        field_changed = False
        last_data = None
        
        # Graph (Energy)
        graph_batch = []
        qg = state.get("q_graph")
        if qg is not None:
            try:
                while True:
                    graph_batch.append(qg.get_nowait())
            except Empty:
                pass

        if graph_batch:
            last_t = 0.0
            for t, e_kin, e_int, e_ext in graph_batch:
                val_pi = e_int + e_ext
                val_l = e_kin - val_pi
                val_h = e_kin + val_pi
                plot_l.update_series("L", t, val_l)
                plot_h.update_series("H", t, val_h)
                last_t = float(t)
            plot_l.update_limits(last_t, t_begin, t_end)
            plot_h.update_limits(last_t, t_begin, t_end)
                
        qf = state["q_field"]
        if qf is not None:
            try:
                while True:
                    last_data = qf.get_nowait()
                    field_changed = True
            except Empty:
                pass
                
        qs = state.get("q_seismo")
        seis_changed = False
        if qs is not None:
            try:
                while True:
                    t_s, vx_top, vy_top = qs.get_nowait()
                    state["seismo_times"].append(float(t_s))
                    state["seismo_values_x"].append(vx_top)
                    state["seismo_values_y"].append(vy_top)
                    seis_changed = True
            except Empty:
                pass

        if field_changed and last_data is not None:
            t, vmag = last_data
            if field_v:
                field_v.update(state["nodes_coords"], vmag)
            elif field_im:
                # vmag is flat (Ny*Nx)
                # reshape to (Ny, Nx)
                f_Nx = state["field_grid"]["Nx"]
                f_Ny = state["field_grid"]["Ny"]
                if vmag.size == f_Nx * f_Ny:
                    grid_data = vmag.reshape(f_Ny, f_Nx)
                    field_im.set_data(grid_data)
            
            runtime["last_t"] = float(t)
            vis.update_time_text(compose_status(runtime, t_begin, t_end, dt, steps_total, float(t)))
            
        if seis_changed:
            times = np.asarray(state["seismo_times"], dtype=float)
            vals_x = np.asarray(state["seismo_values_x"], dtype=float)
            vals_y = np.asarray(state["seismo_values_y"], dtype=float)
            
            if times.size and vals_x.size and vals_y.size:
                x_min = float(np.min(top_x)) if top_x.size else 0.0
                x_max = float(np.max(top_x)) if top_x.size else 1.0
                extent = [x_min, x_max, float(times[0]), float(times[-1])]
                
                # Update X
                seis_im_x.set_array(vals_x)
                seis_im_x.set_extent(extent)
                
                # Update Y
                seis_im_y.set_array(vals_y)
                seis_im_y.set_extent(extent)
                
                # Rescale X
                s_min_x = seis_cfg_x.get("scale_min")
                s_max_x = seis_cfg_x.get("scale_max")
                if s_min_x is None and s_max_x is None:
                    if bool(seis_cfg_x.get("scale_from_history", False)):
                        vmin = float(np.nanmin(vals_x))
                        vmax = float(np.nanmax(vals_x))
                    else:
                        vmin = float(np.nanmin(vals_x[-1]))
                        vmax = float(np.nanmax(vals_x[-1]))
                else:
                    vmin = float(s_min_x) if s_min_x is not None else float(np.nanmin(vals_x))
                    vmax = float(s_max_x) if s_max_x is not None else float(np.nanmax(vals_x))
                if vmax <= vmin: vmax = vmin + 1e-12
                seis_im_x.set_clim(vmin=vmin, vmax=vmax)

                # Rescale Y
                s_min_y = seis_cfg_y.get("scale_min")
                s_max_y = seis_cfg_y.get("scale_max")
                if s_min_y is None and s_max_y is None:
                    if bool(seis_cfg_y.get("scale_from_history", False)):
                        vmin = float(np.nanmin(vals_y))
                        vmax = float(np.nanmax(vals_y))
                    else:
                        vmin = float(np.nanmin(vals_y[-1]))
                        vmax = float(np.nanmax(vals_y[-1]))
                else:
                    vmin = float(s_min_y) if s_min_y is not None else float(np.nanmin(vals_y))
                    vmax = float(s_max_y) if s_max_y is not None else float(np.nanmax(vals_y))
                if vmax <= vmin: vmax = vmin + 1e-12
                seis_im_y.set_clim(vmin=vmin, vmax=vmax)
        
        # Dispersion Update
        if disp_enabled and (runtime["frame_counter"] % disp_update_interval == 0):
            # Try to update dispersion images
            hist_times = np.asarray(state["seismo_times"], dtype=float)
            hist_x = np.asarray(state["seismo_values_x"], dtype=float)
            hist_y = np.asarray(state["seismo_values_y"], dtype=float)
            
            if hist_times.size > 100: # Need some history
                # Compute X
                # hist_x is 1D array of floats? Wait, seismo_values_x stores arrays?
                # on_seismo: state["seismo_values_x"].append(vx_top) where vx_top is array.
                # So hist_x should be list of arrays.
                # Let's fix this access.
                pass
                
                # Re-access list directly to avoid copy/stack overhead every frame if possible
                # But we need stack.
                # Only stack if we gonna compute
                
                # Check data structure:
                # state["seismo_values_x"] is list of (Nx,) arrays.
                # We need (Nt, Nx)
                
                try:
                    # Limit history size for performance? Maybe last N samples?
                    # Phase shift works best with long time window. Let's take all.
                    # Warning: this grows. But Nt ~ 1000-2000 is fine.
                    
                    data_x = np.vstack(state["seismo_values_x"]) # (Nt, Nx)
                    data_y = np.vstack(state["seismo_values_y"]) # (Nt, Nx)
                    times = np.array(state["seismo_times"])
                    
                    # Compute X
                    freqs, vs, img_x = _compute_dispersion_image(times, data_x, top_x, disp_cfg)
                    if img_x is not None and disp_im_x is not None:
                        # image is (Nf, Nv). Transpose for plot (Nv, Nf) -> (Y, X)
                        # extent=[freqs[0], freqs[-1], vs[0], vs[-1]]
                        disp_im_x.set_data(img_x.T)
                        disp_im_x.set_extent([freqs[0], freqs[-1], vs[0], vs[-1]])
                        disp_im_x.autoscale()
                    
                    # Compute Y
                    freqs, vs, img_y = _compute_dispersion_image(times, data_y, top_x, disp_cfg)
                    if img_y is not None and disp_im_y is not None:
                        disp_im_y.set_data(img_y.T)
                        disp_im_y.set_extent([freqs[0], freqs[-1], vs[0], vs[-1]])
                        disp_im_y.autoscale()
                        
                except Exception as e:
                    # e.g. different shapes in stack or memory
                    print(f"Dispersion update error: {e}")

        artists = vis.get_all_artists() + [seis_im_x, seis_im_y]
        if disp_im_x: artists.append(disp_im_x)
        if disp_im_y: artists.append(disp_im_y)
        if field_im: artists.append(field_im)
        return artists

    def start(handle):
        def on_close(_evt):
            # Keep draining external queues after UI closes,
            # so receiver callbacks with blocking put() can finish.
            for key in ("q_graph", "q_field"):
                q = state.get(key)
                if q is not None:
                    threading.Thread(
                        target=drain_queue_until_solver_stops,
                        args=(q, handle),
                        daemon=True,
                    ).start()
            try:
                # Force solver to stop on window close.
                handle.model.events.abort.set()
            except Exception:
                try:
                    threading.Thread(target=handle.terminate, daemon=True).start()
                except Exception:
                    pass

        fig.canvas.mpl_connect("close_event", on_close)
        state["_ani"] = FuncAnimation(
            fig,
            update,
            interval=int(cfg["live"]["interval_ms"]),
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def finalize():
        return {"frames_graph": len(state["hist_graph"]), "frames_field": len(state["hist_field"])}

    return {"start": start, "finalize": finalize}


# ============================================================
# MAIN
# ============================================================

def main(model_cfg: Dict[str, Any], stand_cfg: Dict[str, Any]) -> Dict[str, Any]:

    mode = stand_cfg["mode"]
    assert mode in ("live", "save")

    t_start = time.perf_counter()
    state = build_state(model_cfg, stand_cfg)
    state["t_start"] = t_start
    model = build_model(model_cfg, state)
    stand = build_stand(stand_cfg, state)
    t_before_run = time.perf_counter()

    t_end = float(model_cfg["time"]["t_end"])
    t_compile_start = float(state.get("t_compile_start", t_before_run))
    t_compile_end = float(state.get("t_compile_end", t_before_run))
    t_precompile = max(0.0, t_compile_start - t_start)
    t_compile = max(0.0, t_compile_end - t_compile_start)
    t_postcompile = max(0.0, t_before_run - t_compile_end)
    print(
        f"[lamb_wave/timing] to_compile={t_precompile:.3f}s | "
        f"compile={t_compile:.3f}s | to_run={t_postcompile:.3f}s"
    )


    if mode == "save":
        print("[lamb_wave] run sync (save)")
        model.run(time_end=t_end, sync=True)
        t_total = time.perf_counter() - t_start
        print(f"[lamb_wave] finished in {t_total:.2f}s")
        return stand["finalize"]()

    print("[lamb_wave] run async (live)")
    handle = model.run(time_end=t_end, sync=False)
    stand["start"](handle)

    # after closing window
    try:
        if not handle.poll():
            handle.terminate()
    except Exception:
        pass

    t_total = time.perf_counter() - t_start
    print(f"[lamb_wave] finished in {t_total:.2f}s")
    return stand["finalize"]()


if __name__ == "__main__":
    # CLI:
    #   python test_lamb_wave_energy.py live
    #   python test_lamb_wave_energy.py save
    if len(sys.argv) >= 2:
        stand_cfg["mode"] = sys.argv[1].strip().lower()

    main(model_cfg, stand_cfg)
