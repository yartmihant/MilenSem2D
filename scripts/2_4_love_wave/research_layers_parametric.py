"""
SEMGPU Research Script

research_layers_parametric.py

Параметрическое исследование влияния порядка элемента (order),
размера элемента (element_size) и толщины верхнего слоя (interface_depth)
на волновую картину в двухслойной среде.

Для каждого набора (order, element_size, interface_depth) запускается расчёт
в save-режиме. Генерируются: сейсмограммы (png + segy), поле |v|,
энергии L/H, дисперсионные диаграммы, а также сырые данные (npz).

Запуск:
    python research_layers_parametric.py
"""

import sys
import os
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semgpu.model.receiver import rec_steps_from_dt
from semgpu.model import SGModel, SGConditionModel, SGReceiverModel
from semgpu.solvers import SGLinElast2DNewmarkKimHybridNBSolver as SGSolver
from semgpu.solvers.lin_elast_2d_newmark_kim_hybrid_nb.constants import (
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
)

from semgpu.tools.material import build_from_elements
from semgpu.model import SGPETBLineModel
from semgpu.tools.build_mesh import build_structured_rectangle_mesh
from semgpu.tools.lamb_conditions_common import add_source_condition_nearest_node

# ============================================================
# CONFIG
# ============================================================

research_cfg: Dict[str, Any] = {
    # Parametric grid: list of (order, element_size) pairs
    # element_size — размер элемента в метрах (e_per_x = 1/element_size)
    "order_es_pairs": [
        (9, 10),
        (8, 10),
        (7, 10),
        (6, 10),
        (5, 10),
        (4, 10),
        (3, 10),
        (2, 10),
        (1, 10),

        # (9, 15),
        # (9, 12),
        # (9, 10),
        # (9, 7.5),
        # (9, 5),
        # (9, 2.5),
        # (9, 2),
        # (8, 10),
        # (8, 5), 
        # (8, 2.5), 
        # (8, 2), 
        # (7, 5),
        # (7, 4), 
        # (7, 2.5), 
        # (7, 1.5), 
        # (7, 1), 
        # (7, 0.75), 
        # (6, 1),
        # (6, 0.8),
        # (6, 0.6),
        # (5, 0.5),
        # (5, 0.4),
        # (4, 0.25),
        # (4, 0.2),
        # (4, 0.15),
        # (3, 0.15),
        # (3, 0.1),
        # (2, 0.08),
        # (2, 0.04),
        # (1, 0.05),
        # (1, 0.02),
    ],

    # Interface depth variants (thickness of top layer in meters, measured from y=0 downward)
    "interface_depths": [20],

    # Output root directory
    "output_root": os.path.join(os.path.dirname(__file__), "parametric_layers2"),

    # Common model parameters
    "Lx": 300.0,
    "Ly": 60.0,

    "material_top": {"rho": 1600, "cp": 1100, "cs": 330},
    "material_bottom": {"rho": 2000, "cp": 1800, "cs": 540},

    "solver": {"kim_beta": 1 / 27, "kim_gamma": 0.6},
    "time": {"t_begin": 0.0, "t_end": 0.3, "dt_safety": 0.1, "dt": 0.0},

    "source": {
        "type": "NN_POINT_FORCE_NEAREST",
        "amp": 1e7,
        "f0": 30,
        "duration": 0.5,
        "steps": 256,
        "y_depth": 2
    },

    "damping": {"enabled": True, "layer": 30.0, "reflection": 4.586801e-01, "power": 0.8498},
    "abc": {"enabled": True},

    "receivers": {
        "graph": {"dt_frame": 0.0001, "qmax": 100},
        "field": {
            "dt_frame": 0.001, "qmax": 1,
            "grid": {"Nx": 420, "Ny": 84},
        },
        "seismo": {
            "dt_frame": 0.0001, "qmax": 100,
            "grid": {"n_sensors": 400, "y": "top"},
        },
    },

    # Visualization axes
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
    },

    "dispersion": {
        "enabled": True,
        "v_min": 0.0,
        "v_max": 2000.0,
        "v_steps": 512,
        "f_max": 150.0,
    },
}


# ============================================================
# HELPERS (from test_plot_lamb_wave_layers, standalone)
# ============================================================

def _get_elastic_properties(props: Dict[str, float]) -> Dict[str, float]:
    rho = float(props["rho"])
    if "cp" in props and "cs" in props:
        cp = float(props["cp"])
        cs = float(props["cs"])
        mu = rho * cs * cs
        lam = rho * cp * cp - 2 * mu
        if abs(lam + mu) < 1e-14:
            E = 0.0
            nu = 0.0
        else:
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            nu = lam / (2 * (lam + mu))
    elif "E" in props and "nu" in props:
        E = float(props["E"])
        nu = float(props["nu"])
        mu = E / (2.0 * (1.0 + nu))
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        if abs(denom) < 1e-14:
            lam = 1e14
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
    power: float,
    t_begin: float,
    t_end: float,
) -> Optional[SGConditionModel]:
    L = float(layer)
    R = float(reflection)
    if L <= 0.0:
        return None
    if not (0.0 < R < 1.0):
        raise ValueError(f"reflection must be in (0,1), got {R}")

    nodes_coords = np.asarray(mesh.nodes_coords, dtype=float)
    x = nodes_coords[:, 0]
    y = nodes_coords[:, 1]

    x_max = float(np.max(x))
    y_min = float(np.min(y))

    l_right = np.clip(x - (x_max - L), 0.0, L)
    l_bottom = np.clip((y_min + L) - y, 0.0, L)

    in_layer = (l_right > 0.0) | (l_bottom > 0.0)
    node_ids = np.flatnonzero(in_layer).astype(int)
    if node_ids.size == 0:
        return None

    n_nodes = int(mesh.nodes_coords.shape[0])
    e2_nodes = np.asarray(mesh.elements[(2, 2, 2)]["nodes"], dtype=int).ravel()
    first_enid = np.full(n_nodes, -1, dtype=int)
    for enid, nid in enumerate(e2_nodes):
        if first_enid[int(nid)] < 0:
            first_enid[int(nid)] = int(enid)

    enid_sel = first_enid[node_ids]
    if np.any(enid_sel < 0):
        raise RuntimeError("Damping layer: failed nid->enid mapping.")

    names = material.names
    idx_rho = names.index("rho")
    idx_E = names.index("E")
    idx_nu = names.index("nu")

    rho_v = material.values[enid_sel, idx_rho].astype(float)
    E_v = material.values[enid_sel, idx_E].astype(float)
    nu_v = np.clip(material.values[enid_sel, idx_nu].astype(float), 0.0, 0.49)

    c_p = np.sqrt(np.maximum(E_v / rho_v * ((1.0 - nu_v) / (1.0 + nu_v) / (1.0 - 2.0 * nu_v)), 0.0))
    alpha_max = (2.0 * c_p / L) * float(np.log(1.0 / R))

    rr = l_right[node_ids] / L
    bb = l_bottom[node_ids] / L
    alpha_vals = alpha_max * (rr ** power + bb ** power)
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
# BUILD single run configuration
# ============================================================

def _make_model_cfg(rcfg: Dict[str, Any], order: int, element_size: float, interface_depth: float) -> Dict[str, Any]:
    Lx = float(rcfg["Lx"])
    Ly = float(rcfg["Ly"])
    e_per_x = 1.0 / element_size
    return {
        "mesh": {
            "Lx": Lx, "Ly": Ly,
            "Ex": int(Lx * e_per_x), "Ey": int(Ly * e_per_x),
            "order": order, "distortion": 0.0,
        },
        "material_top": dict(rcfg["material_top"]),
        "material_bottom": dict(rcfg["material_bottom"]),
        "interface_y": -interface_depth,
        "solver": dict(rcfg["solver"]),
        "time": dict(rcfg["time"]),
        "source": dict(rcfg["source"]),
        "damping": dict(rcfg["damping"]),
        "abc": dict(rcfg["abc"]),
        "receivers": rcfg["receivers"],
    }


def _make_stand_cfg(rcfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    return {
        "mode": "save",
        "output": {
            "dir": out_dir,
            "velocity_magnitude_png": "velocity_magnitude.png",
            "lagrangian_png": "lagrangian.png",
            "hamiltonian_png": "hamiltonian.png",
        },
        "axes": rcfg.get("axes", {}),
        "dispersion": rcfg.get("dispersion", {}),
    }


# ============================================================
# BUILD MODEL (adapted from test_plot_lamb_wave_layers)
# ============================================================

def build_state(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_cfg": model_cfg,
        "nodes_coords": np.array([]),
        "N": 0, "dt": 0, "steps": 0,
        "t_begin": 0, "t_end": 0,
        "hist_graph": [],
        "hist_field": [],
        "hist_seismo": [],
        "q_graph": None, "q_field": None, "q_seismo": None,
        "last_field_time": 0, "last_field_frame": 0,
        "seismo_times": [], "seismo_values_x": [], "seismo_values_y": [],
    }


def build_model(cfg: Dict[str, Any], state: Dict[str, Any]) -> SGModel:
    t_begin = float(cfg["time"]["t_begin"])
    t_end = float(cfg["time"]["t_end"])

    mcfg = cfg["mesh"]
    pet_line = SGPETBLineModel(name=f"Line{int(mcfg['order'])}", order=int(mcfg["order"]))
    mesh = build_structured_rectangle_mesh(
        Lx=float(mcfg["Lx"]), Ly=float(mcfg["Ly"]),
        Ex=int(mcfg["Ex"]), Ey=int(mcfg["Ey"]),
        cx=float(mcfg["Lx"]) / 2.0, cy=-float(mcfg["Ly"]) / 2.0,
        order=int(mcfg["order"]),
        pets=[pet_line],
        distortion=float(mcfg["distortion"]),
    )

    nodes_coords = mesh.nodes_coords.T
    N = int(mesh.nodes_coords.shape[0])

    props_top = _get_elastic_properties(cfg["material_top"])
    props_bot = _get_elastic_properties(cfg["material_bottom"])
    vals_top = np.array([props_top["rho"], props_top["E"], props_top["nu"]], dtype=float)
    vals_bot = np.array([props_bot["rho"], props_bot["E"], props_bot["nu"]], dtype=float)

    y_interface = float(cfg["interface_y"])

    def element_law(dim, eid, nodes_indices, nodes_coords_el):
        center_y = np.mean(nodes_coords_el[:, 1])
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

    if cfg["time"]["dt"]:
        model.dt = cfg["time"]["dt"]
    elif cfg["time"]["dt_safety"]:
        model.solver.compute_optimal_dt(
            dividend=t_end - t_begin,
            safety=cfg["time"]["dt_safety"],
        )

    dt = model.dt
    steps_total = int(round((t_end - t_begin) / dt))

    state["nodes_coords"] = nodes_coords.copy()
    state["N"] = N
    state["dt"] = dt
    state["steps"] = steps_total
    state["t_begin"] = t_begin
    state["t_end"] = t_end

    xs = mesh.nodes_coords[:, 0]
    ys = mesh.nodes_coords[:, 1]
    y_top = float(np.max(ys) - cfg["y_depth"])

    # Source
    src_cfg = dict(cfg["source"])
    src_cfg["x"] = 0.0
    src_cfg["y"] = float(y_top)
    add_source_condition_nearest_node(
        model=model, mesh=mesh, source_cfg=src_cfg,
        t_begin=t_begin, dt=dt,
        x_min=float(np.min(xs)), x_max=float(np.max(xs)),
        y_min=float(np.min(ys)), y_max=float(np.max(ys)),
    )

    # Symmetry left
    left_nodes = np.flatnonzero(mesh.nodes_tags[:, 2] == 4).astype(int)
    if left_nodes.size > 0:
        model.conditions.append(SGConditionModel(
            "symmetry_left", family_code=COND_FAM_DN, type_code=COND_CODE_DN_UX,
            begin=t_begin, end=t_end, domain=left_nodes, nodes_space="N",
            data=np.zeros((left_nodes.size, 1, 1), dtype=float), interp_degree=0,
        ))

    # Damping
    damping_cfg = cfg.get("damping", {})
    if bool(damping_cfg.get("enabled", True)):
        cond_layer = _build_mass_damping_layer_condition(
            mesh=mesh, material=material,
            layer=float(damping_cfg.get("layer", 0.0)),
            reflection=float(damping_cfg.get("reflection", 1e-6)),
            power=float(damping_cfg.get("power", 3.0)),
            t_begin=t_begin, t_end=t_end,
        )
        if cond_layer is not None:
            model.conditions.append(cond_layer)

    # ABC
    abc_data_value = np.array([props_bot["rho"] * props_bot["cp"], props_bot["rho"] * props_bot["cs"]], dtype=float)
    if bool(cfg.get("abc", {}).get("enabled", True)):
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
            model.conditions.append(SGConditionModel(
                f"abc_{edge_name}", family_code=COND_FAM_P1_DMP, type_code=COND_CODE_P1_DMP_ABC,
                begin=t_begin, end=t_end, domain=edge_enids, nodes_space="P221",
                data=abc_data, interp_degree=0,
            ))

    # Receivers setup
    all_nodes = np.arange(N, dtype=int)
    weights = np.ones(N, dtype=float)
    receivers_cfg = cfg.get("receivers", {})
    dt_graph_frame = float(receivers_cfg.get("graph", {}).get("dt_frame", 0.0))
    dt_field_frame = float(receivers_cfg.get("field", {}).get("dt_frame", 0.0))
    dt_seismo_frame = float(receivers_cfg.get("seismo", {}).get("dt_frame", dt_field_frame))
    graph_steps = rec_steps_from_dt(t_begin, t_end, dt_graph_frame, dt)
    field_steps = rec_steps_from_dt(t_begin, t_end, dt_field_frame, dt)
    seismo_steps = rec_steps_from_dt(t_begin, t_end, dt_seismo_frame, dt)

    # Seismo receivers (uniform on top boundary)
    seismo_cfg = receivers_cfg.get("seismo", {})
    n_sensors = int(seismo_cfg.get("grid", {}).get("n_sensors", 400))
    x_min_v, x_max_v = float(np.min(xs)), float(np.max(xs))
    target_x = np.linspace(x_min_v, x_max_v, n_sensors)
    target_coords_seismo = np.zeros((n_sensors, 2))
    target_coords_seismo[:, 0] = target_x
    target_coords_seismo[:, 1] = y_top

    Lx_val = float(mcfg["Lx"])
    Ly_val = float(mcfg["Ly"])
    Ex_val = int(mcfg["Ex"])
    Ey_val = int(mcfg["Ey"])
    cx_val = Lx_val / 2.0
    cy_val = -Ly_val / 2.0
    dx = Lx_val / Ex_val
    dy = Ly_val / Ey_val

    x_local = target_x - (cx_val - Lx_val / 2.0)
    ix = np.clip(np.floor(x_local / dx).astype(int), 0, Ex_val - 1)
    iy = Ey_val - 1
    eids_seismo = iy * Ex_val + ix

    elements_ids_seismo = np.zeros((n_sensors, 2), dtype=int)
    elements_ids_seismo[:, 0] = 2
    elements_ids_seismo[:, 1] = eids_seismo

    refs_seismo = mesh.phys_to_refs(elements_ids_seismo, target_coords_seismo)
    weights_seismo = mesh.sampling_shape(elements_ids_seismo, refs_seismo)

    nodes_flat = mesh.elements[(2, 2, 2)]["nodes"]
    offsets = mesh.elements[(2, 2, 2)]["nodes_offsets"]
    S = weights_seismo.shape[1]

    start_indices = offsets[eids_seismo]
    gather_indices = start_indices[:, None] + np.arange(S)[None, :]
    element_nodes_seismo = nodes_flat[gather_indices]

    sens_nodes_seismo = element_nodes_seismo.flatten()
    sens_weights_seismo = weights_seismo.flatten()
    sens_frame_seismo = np.repeat(np.arange(n_sensors), S)

    state["top_x"] = target_x

    # Field receivers (regular grid)
    field_cfg = receivers_cfg.get("field", {})
    field_grid = field_cfg.get("grid", {})
    f_Nx = int(field_grid.get("Nx", 100))
    f_Ny = int(field_grid.get("Ny", 20))
    f_x = np.linspace(x_min_v, x_max_v, f_Nx)
    f_y_min, f_y_max = float(np.min(ys)), float(np.max(ys))
    f_y = np.linspace(f_y_min, f_y_max, f_Ny)
    f_X, f_Y = np.meshgrid(f_x, f_y)
    f_coords = np.column_stack((f_X.ravel(), f_Y.ravel()))
    n_field_points = f_coords.shape[0]

    fx_local = f_coords[:, 0] - (cx_val - Lx_val / 2.0)
    fy_local = f_coords[:, 1] - (cy_val - Ly_val / 2.0)
    fix = np.clip(np.floor(fx_local / dx).astype(int), 0, Ex_val - 1)
    fiy = np.clip(np.floor(fy_local / dy).astype(int), 0, Ey_val - 1)
    eids_field = fiy * Ex_val + fix

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

    state["field_grid"] = {"Nx": f_Nx, "Ny": f_Ny, "x": f_x, "y": f_y, "coords": f_coords}

    state["_last_progress_pct"] = -1
    state["_run_start"] = None

    # Callbacks
    def on_graph(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        state["hist_graph"].append((float(frame_time), float(frame[0]), float(frame[1]), float(frame[2])))

    def on_field(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        data = frame.reshape(4, n_field_points)
        vx = data[2, :].astype(float)
        vy = data[3, :].astype(float)
        vmag = np.sqrt(vx * vx + vy * vy)
        state["hist_field"].append((frame_time, vmag))
        state["last_field_time"] = float(frame_time)
        state["last_field_frame"] = int(frame_id)

    def on_seismo(frame: np.ndarray, frame_id: int, frame_time: float, receiver: SGReceiverModel, model_: SGModel):
        data = frame.reshape(2, n_sensors)
        vx_top = data[0, :].astype(float)
        vy_top = data[1, :].astype(float)
        state["hist_seismo"].append((float(frame_time), vx_top, vy_top))
        # progress bar
        if state["_run_start"] is None:
            state["_run_start"] = time.perf_counter()
        t_span = state["t_end"] - state["t_begin"]
        frac = (frame_time - state["t_begin"]) / t_span if t_span > 0 else 0.0
        pct = int(frac * 100)
        if pct > state["_last_progress_pct"]:
            state["_last_progress_pct"] = pct
            elapsed = time.perf_counter() - state["_run_start"]
            eta = elapsed / frac * (1.0 - frac) if frac > 1e-6 else 0.0
            bar_len = 30
            filled = int(bar_len * frac)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  [{bar}] {pct:3d}%  t={frame_time:.4f}s  {elapsed:.0f}s/{elapsed+eta:.0f}s", end="", flush=True)

    model.receivers.append(SGReceiverModel(
        "graph", begin=t_begin, end=t_end, steps=graph_steps,
        components=np.array([STATE_CODE_KIN_ENERGY, STATE_CODE_F_INT_POT_ENERGY, STATE_CODE_F_EXT_POT_ENERGY], dtype=int),
        sens_nodes=all_nodes, sens_frame=np.zeros(N, dtype=int), sens_weights=weights,
        onframe=on_graph,
    ))
    model.receivers.append(SGReceiverModel(
        "field", begin=t_begin, end=t_end, steps=field_steps,
        components=np.array([STATE_CODE_X, STATE_CODE_Y, STATE_CODE_VX, STATE_CODE_VY], dtype=int),
        sens_nodes=sens_nodes_field, sens_frame=sens_frame_field, sens_weights=sens_weights_field,
        onframe=on_field,
    ))
    model.receivers.append(SGReceiverModel(
        "seismo", begin=t_begin, end=t_end, steps=seismo_steps,
        components=np.array([STATE_CODE_VX, STATE_CODE_VY], dtype=int),
        sens_nodes=sens_nodes_seismo, sens_frame=sens_frame_seismo, sens_weights=sens_weights_seismo,
        onframe=on_seismo,
    ))

    print(f"  built: N={N}, dt={dt:.6e}, steps={steps_total}")
    return model


# ============================================================
# SAVE ROUTINES
# ============================================================

def _compute_dispersion_image(times: np.ndarray, data_vals: np.ndarray, top_x: np.ndarray, disp_cfg: Dict[str, Any]):
    if len(times) < 2 or data_vals.size == 0:
        return None, None, None
    data = data_vals - np.mean(data_vals, axis=0)
    Nt, Nx = data.shape
    dt_s = times[1] - times[0]
    if dt_s <= 1e-9:
        return None, None, None

    x = top_x - top_x[0]
    v_min = float(disp_cfg.get("v_min", 0.0))
    v_max = float(disp_cfg.get("v_max", 2000.0))
    nv = int(disp_cfg.get("v_steps", 200))
    f_max = float(disp_cfg.get("f_max", 100.0))

    freqs = np.fft.rfftfreq(Nt, dt_s)
    spec = np.fft.rfft(data, axis=0)
    mask_f = (freqs > 0) & (freqs <= f_max)
    freqs = freqs[mask_f]
    spec = spec[mask_f, :]

    vs = np.linspace(v_min, v_max, nv)
    image = np.zeros((freqs.size, nv))
    for i, v in enumerate(vs):
        if v < 1e-3:
            continue
        arg = 2.0 * np.pi * freqs[:, None] * (x[None, :] / v)
        shift = np.exp(1j * arg)
        stack = np.sum(spec * shift, axis=1)
        image[:, i] = np.abs(stack)
    if np.max(image) > 0:
        image = image / np.max(image)
    return freqs, vs, image


def _save_segy(filepath: str, data: np.ndarray, times: np.ndarray, x_coords: np.ndarray) -> None:
    import segyio
    Nt, Nx = data.shape
    dt_us = max(1, int(round((times[1] - times[0]) * 1e6))) if Nt > 1 else 1000
    t0_ms = int(round(float(times[0]) * 1000.0)) if times.size > 0 else 0
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
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


def save_results(state: Dict[str, Any], stand_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> None:
    out_dir = stand_cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    hist_graph = state["hist_graph"]
    hist_field = state["hist_field"]
    hist_seismo = state["hist_seismo"]
    axes_cfg = stand_cfg.get("axes", {})

    if not hist_field and not hist_graph and not hist_seismo:
        print("  [save] no frames collected")
        return

    # --- 1. Velocity Magnitude ---
    if hist_field:
        field_grid = state.get("field_grid")
        plt.figure(figsize=(8, 6))
        fv_cfg = axes_cfg.get("field_v", {})
        cmap_v = str(fv_cfg.get("cmap", "viridis"))
        imshow_kw: Dict[str, Any] = {}
        if fv_cfg.get("scale_min") is not None:
            imshow_kw["vmin"] = float(fv_cfg["scale_min"])
        if fv_cfg.get("scale_max") is not None:
            imshow_kw["vmax"] = float(fv_cfg["scale_max"])

        if field_grid:
            f_Nx = field_grid["Nx"]
            f_Ny = field_grid["Ny"]
            f_x = field_grid["x"]
            f_y = field_grid["y"]
            extent = [f_x[0], f_x[-1], f_y[0], f_y[-1]]
            _, vmag = hist_field[-1]
            if vmag.size == f_Nx * f_Ny:
                grid_data = vmag.reshape(f_Ny, f_Nx)
                plt.imshow(grid_data, origin="lower", extent=extent, aspect="equal", cmap=cmap_v, **imshow_kw)
        else:
            nc = state["nodes_coords"]
            _, vmag = hist_field[-1]
            plt.scatter(nc[0, :], nc[1, :], c=vmag, s=4, cmap=cmap_v, **imshow_kw)
            plt.gca().set_aspect("equal", "box")
        plt.colorbar(label="|v|")
        plt.title("Velocity Magnitude |v|(x,t_last)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "velocity_magnitude.png"))
        plt.close()

    # --- 2. Energy ---
    if hist_graph:
        times_g = [it[0] for it in hist_graph]
        ekin = np.array([it[1] for it in hist_graph])
        eint = np.array([it[2] for it in hist_graph])
        eext = np.array([it[3] for it in hist_graph])
        epi = eint + eext
        L_arr = ekin - epi
        H_arr = ekin + epi

        for arr, label, color, fname in [
            (L_arr, "L = T - Pi", "purple", "energy_L.png"),
            (H_arr, "H = T + Pi", "green", "energy_H.png"),
        ]:
            plt.figure(figsize=(10, 6))
            plt.plot(times_g, arr, color=color, label=label)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Time [s]"); plt.ylabel("Energy")
            plt.title(f"{label}(t)")
            plt.legend()
            plt.savefig(os.path.join(out_dir, fname))
            plt.close()

    # --- 3. Seismograms ---
    top_x = state.get("top_x")
    if hist_seismo and top_x is not None and top_x.size > 0:
        times_s = np.array([h[0] for h in hist_seismo])
        vx_data = np.vstack([h[1] for h in hist_seismo])
        vy_data = np.vstack([h[2] for h in hist_seismo])
        x_min, x_max = float(np.min(top_x)), float(np.max(top_x))
        t_min, t_max = float(times_s[0]), float(times_s[-1])
        extent = [x_min, x_max, t_min, t_max]

        for data_arr, comp, cfg_key in [(vx_data, "vx", "seismo_x"), (vy_data, "vy", "seismo_y")]:
            s_cfg = axes_cfg.get(cfg_key, {})
            cmap = str(s_cfg.get("cmap", "RdBu"))
            s_min = s_cfg.get("scale_min")
            s_max = s_cfg.get("scale_max")
            if s_min is not None and s_max is not None:
                vmin_s, vmax_s = float(s_min), float(s_max)
            else:
                vm = np.max(np.abs(data_arr)) if np.max(np.abs(data_arr)) > 0 else 1.0
                vmin_s, vmax_s = -vm, vm

            plt.figure(figsize=(10, 8))
            plt.imshow(data_arr, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin_s, vmax=vmax_s)
            plt.colorbar(label=comp.upper())
            plt.title(f"Seismogram {comp.upper()}(x, t)")
            plt.xlabel("x"); plt.ylabel("t")
            plt.savefig(os.path.join(out_dir, f"seismogram_{comp}.png"))
            plt.close()

            # SEG-Y
            _save_segy(os.path.join(out_dir, f"seismogram_{comp}.segy"), data_arr, times_s, top_x)

        # --- 4. NPZ ---
        npz_path = os.path.join(out_dir, "data.npz")
        npz_data: Dict[str, Any] = {
            "seismo_times": times_s,
            "seismo_vx": vx_data,
            "seismo_vy": vy_data,
            "top_x": top_x,
        }
        if hist_graph:
            npz_data["graph_times"] = np.array([it[0] for it in hist_graph])
            npz_data["graph_ekin"] = np.array([it[1] for it in hist_graph])
            npz_data["graph_eint"] = np.array([it[2] for it in hist_graph])
            npz_data["graph_eext"] = np.array([it[3] for it in hist_graph])
        if hist_field:
            npz_data["field_times"] = np.array([it[0] for it in hist_field])
            npz_data["field_vmag"] = np.vstack([it[1] for it in hist_field])
        np.savez_compressed(npz_path, **npz_data)
        print(f"  [save] saved: {npz_path}")

    # --- 5. Dispersion ---
    disp_cfg = stand_cfg.get("dispersion", {})
    if disp_cfg.get("enabled", False) and hist_seismo and top_x is not None:
        times_s = np.array([h[0] for h in hist_seismo])
        for comp_idx, comp_name in [(1, "vx"), (2, "vy")]:
            data_c = np.vstack([h[comp_idx] for h in hist_seismo])
            freqs, vs, image = _compute_dispersion_image(times_s, data_c, top_x, disp_cfg)
            if image is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(image.T, origin="lower", aspect="auto",
                           extent=[freqs[0], freqs[-1], vs[0], vs[-1]],
                           cmap="jet", interpolation="bilinear")
                plt.colorbar(label="Amplitude")
                plt.title(f"Dispersion ({comp_name.upper()}) | Phase Shift")
                plt.xlabel("Frequency (Hz)"); plt.ylabel("Phase Velocity (m/s)")
                props_t = _get_elastic_properties(model_cfg["material_top"])
                cs_t = props_t["cs"]
                cr_t = cs_t * (0.87 + 1.12 * props_t["nu"]) / (1 + props_t["nu"])
                plt.axhline(cr_t, color="white", linestyle="--", linewidth=2, label=f"Rayleigh Top ({cr_t:.0f})")
                plt.axhline(cs_t, color="white", linestyle=":", linewidth=1, label=f"Shear Top ({cs_t:.0f})")
                props_b = _get_elastic_properties(model_cfg["material_bottom"])
                plt.axhline(props_b["cs"], color="gray", linestyle="-.", linewidth=1, label=f"Shear Bot ({props_b['cs']:.0f})")
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(out_dir, f"dispersion_{comp_name}.png"))
                plt.close()

    print(f"  [save] all results saved to {out_dir}")


# ============================================================
# RUN SINGLE CASE
# ============================================================

def run_single(model_cfg: Dict[str, Any], stand_cfg: Dict[str, Any]) -> None:
    state = build_state(model_cfg)
    model = build_model(model_cfg, state)
    t_end = float(model_cfg["time"]["t_end"])
    t0 = time.perf_counter()
    model.run(time_end=t_end, sync=True)
    elapsed = time.perf_counter() - t0
    print(f"\r  solver finished in {elapsed:.1f}s" + " " * 40)
    save_results(state, stand_cfg, model_cfg)


# ============================================================
# MAIN
# ============================================================

def main(rcfg: Dict[str, Any]) -> None:
    pairs: List[Tuple[int, float]] = rcfg["order_es_pairs"]
    depths: List[float] = rcfg["interface_depths"]
    output_root = rcfg["output_root"]

    total = len(pairs) * len(depths)
    idx = 0
    t_research_start = time.perf_counter()
    for order, element_size in pairs:
        for depth in depths:
            idx += 1
            es_int = int(round(element_size * 100))
            tag = f"o{order}_es{es_int}_d{depth}"
            out_dir = os.path.join(output_root, tag)
            elapsed_total = time.perf_counter() - t_research_start
            if idx > 1:
                avg_per_case = elapsed_total / (idx - 1)
                eta_total = avg_per_case * (total - idx + 1)
                eta_str = f"  ETA total: {eta_total:.0f}s"
            else:
                eta_str = ""
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] order={order}  element_size={element_size}  depth={depth}m  -> {tag}{eta_str}")
            print(f"{'='*60}")

            model_cfg = _make_model_cfg(rcfg, order, element_size, depth)
            stand_cfg = _make_stand_cfg(rcfg, out_dir)

            try:
                run_single(model_cfg, stand_cfg)
            except Exception:
                print(f"  FAILED: {tag}")
                traceback.print_exc()
                continue

    elapsed_total = time.perf_counter() - t_research_start
    print(f"\n{'='*60}")
    print(f"All {total} cases completed in {elapsed_total:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(research_cfg)
