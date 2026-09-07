""" # Численное моделирование сейсмических волн MILEN SEM 2D. Часть вторая # """

""" Исторический этап главы II.5. Основной актуальный комплект строится в
dev_2_5_material_consistency.py. Данный этап использует legacy-растр;
результаты диагностики загрузчика относятся к его версии на момент аудита. """

""" ## Глава II.5. Задача 1: глубина отражения для источника x = 6000 м ## """

"""
ВНИМАНИЕ: предварительная интерпретация по SEG-Y отозвана после аудита
dev_2_5_2_fc_segy_audit.py. Глубина расчётного FC — 2650 м; ниже старый
расчёт использует всю область растра 2750 м. Текущий загрузчик превращает
таблицы в константы слоёв; его отражения от 2401.84 и 2411.84 м приходят
на 2.1546 и 2.1604 с. Для интерпретации расчёта использовать новый аудит.
"""

"""
Оценим глубину яркого события Vy на 2.1–2.2 с по вертикальному профилю
материала SEG-Y. Сопоставим время пробега с импедансом, геологическими
слоями и исходными сейсмическими горизонтами. Это локальная оценка PP,
а не идентификация волны по полному двумерному расчёту.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy.signal import find_peaks


""" ## 1. Исходные данные и параметры источника ## """

# Параметры проверены в TrySEMGPU/research/seismic/test_exp_seismic_sweep.py
# и semgpu/src/semgpu/solvers/tools/builders/conditions/lin_elast_2d/lamb.py.
source_x = 6000.0  # горизонтальная координата источника, м
source_depth = 5.0  # глубина центра объёмного источника давления, м
source_frequency = 30.0  # частота импульса Рикера, Гц
source_peak_time = 1.0 / source_frequency  # максимум Рикера, с
observed_window = np.array([2.1, 2.2])  # исходное окно поиска, с
output_prefix = "dev_2_5_1_reflection_depth"  # префикс результатов задачи
data_dir = Path("data")
image_dir = Path("img")
data_dir.mkdir(exist_ok=True)
image_dir.mkdir(exist_ok=True)

with np.load(data_dir / "dev_2_3/x6000_data.npz") as shot:
    assert np.isclose(float(shot["x_center"]), source_x)
    receiver_x = shot["sensor_x"]
    times = shot["seismo_times"]
    seismo_vy = shot["seismo_vy"]
    seismo_vx = shot["seismo_vx"]

with np.load(data_dir / "dev_1_7_material_grids.npz") as grid:
    coords = grid["coords_grid"]
    layer_indexes = grid["layer_indexes_grid"]


""" ## 2. Профиль материала и физические координаты ## """

"""
Экспорт главы II.1 записал центры ячеек 2.5, 7.5, ... м с заголовками
0, 5, ... м. Используем именно физические границы ячеек 0, 5, ... 2750 м.
Трасса с CDP_X=6000 относится к центру x=6002.5 м. Отдельно проверим
соседний столбец x=5997.5 м, чтобы оценить чувствительность привязки.
Плотность в SEG-Y дана в г/см³; переводим её в кг/м³.
"""

profiles = {}
for name in ("Vp", "Vs", "Density"):
    with segyio.open(str(data_dir / f"dev_2_1_{name}_model.sgy"),
                     "r", ignore_geometry=True) as model:
        header_x = model.attributes(segyio.TraceField.CDP_X)[:]
        trace_index = int(np.argmin(np.abs(header_x - source_x)))
        assert header_x[trace_index] == source_x
        profiles[name] = model.trace[trace_index].astype(float)
        profiles[name + "_left"] = model.trace[trace_index - 1].astype(float)
        assert len(profiles[name]) == coords.shape[1]

vp = profiles["Vp"]
vs = profiles["Vs"]
rho = 1000.0 * profiles["Density"]
depth_centers = coords[trace_index, :, 1]
depth_step = float(np.diff(depth_centers)[0])
assert np.allclose(np.diff(depth_centers), depth_step)
depth_edges = np.r_[depth_centers - depth_step / 2,
                    depth_centers[-1] + depth_step / 2]
assert np.isclose(depth_edges[0], 0.0)
assert np.all(np.isfinite(vp)) and np.all(vp > vs) and np.all(vs > 0)
assert np.all(np.isfinite(rho)) and np.all(rho > 0)
layer_numbers = layer_indexes[trace_index] + 1
impedance = rho * vp
reflection_coefficient = np.diff(impedance) / (impedance[1:] + impedance[:-1])


""" ## 3. Интегрируем медленность и учитываем источник ## """

r"""
Интегрируем **обратную скорость**, а не скорость:

$$T_P(z)=\int_0^z\frac{d\zeta}{V_P(x,\zeta)},\qquad
t_{PP}(z)=t_{\rm Ricker}+2T_P(z)-T_P(z_s).$$

Приёмники находятся на поверхности, центр источника — на глубине $z_s=5$ м.
Суммирование $\Delta z/V_P$ точно для кусочно-постоянного растра.
Время $t_{PP}$ привязано к максимуму исходного давления: отдельный экстремум
Vy может иметь другой фазовый сдвиг. Нормальный акустический коэффициент
$R=(Z_2-Z_1)/(Z_2+Z_1)$ служит индикатором контраста, не прогнозом амплитуды Vy.

Метод: https://www.eoas.ubc.ca/courses/eosc350/content/methods/meth_10d/normal.html
"""

one_way_p = np.r_[0.0, np.cumsum(np.diff(depth_edges) / vp)]
one_way_s = np.r_[0.0, np.cumsum(np.diff(depth_edges) / vs)]
source_leg = float(np.interp(source_depth, depth_edges, one_way_p))
two_way_p = 2.0 * one_way_p
arrival_pp = source_peak_time + two_way_p - source_leg
# Дополнительные ориентиры PS и SS для тех же глубин, без расчёта амплитуд.
arrival_ps = source_peak_time + one_way_p + one_way_s - source_leg
arrival_ss = (source_peak_time + 2.0 * one_way_s
              - np.interp(source_depth, depth_edges, one_way_s))
left_one_way = np.r_[0.0, np.cumsum(np.diff(depth_edges) / profiles["Vp_left"])]
left_arrival_pp = (source_peak_time + 2.0 * left_one_way
                   - np.interp(source_depth, depth_edges, left_one_way))
depth_window = np.interp(observed_window, arrival_pp, depth_edges)
uncorrected_depth_window = np.interp(observed_window, two_way_p, depth_edges)
print("Глубины для окна 2.1–2.2 с, с поправкой источника:", depth_window)
print("Без поправки источника:", uncorrected_depth_window)


""" ## 4. Привязываем последние слои и исходные горизонты ## """

with np.load(data_dir / "dev_1_5_2_layer_boundaries_quadratic.npz") as geometry:
    boundaries = np.array([np.interp(source_x, geometry["distances"], row)
                           for row in geometry["layer_boundaries_array"]])
    formations = geometry["formations"]

with np.load(data_dir / "dev_1_2_profile_sections_depths.npz") as horizons:
    horizon_depths = np.array([
        np.interp(source_x, horizons["distances"], column)
        for column in horizons["profile_depths"].T
    ])
    horizon_names = horizons["surface_names"]

candidate_depths = np.array([boundaries[72], boundaries[73],
                             horizon_depths[np.argmax(horizon_depths)],
                             2490.0, 2495.0, 2500.0, 2670.0, depth_edges[-1]])
candidate_names = ["Граница 73/74", "Граница 74/75", "Горизонт A",
                   "Контраст внутри слоя 75: 2490 м",
                   "Контраст внутри слоя 75: 2495 м",
                   "Низ перехода: 2500 м", "Переход: 2670 м", "Дно модели"]
candidate_times = np.interp(candidate_depths, depth_edges, arrival_pp)
for name, depth, arrival in zip(candidate_names, candidate_depths, candidate_times):
    print(f"{name}: z={depth:.2f} м, t_PP={arrival:.6f} с")

receiver_index = int(np.argmin(np.abs(receiver_x - source_x)))
window_indices = np.flatnonzero((times >= 2.12) & (times <= 2.19))
peak_indices = window_indices[find_peaks(np.abs(seismo_vy[window_indices, receiver_index]))[0]]
peak_indices = peak_indices[np.argsort(-np.abs(seismo_vy[peak_indices, receiver_index]))]
for index in peak_indices[:4]:
    print(f"Экстремум Vy: {times[index]:.6f} с, "
          f"формальная глубина {np.interp(times[index], arrival_pp, depth_edges):.1f} м")


""" ## 5. Сохраняем численные результаты ## """

np.savez_compressed(
    data_dir / f"{output_prefix}.npz", depth_centers=depth_centers,
    depth_edges=depth_edges, vp=vp, vs=vs, rho=rho, impedance=impedance,
    reflection_coefficient=reflection_coefficient, layer_numbers=layer_numbers,
    two_way_p=two_way_p, arrival_pp=arrival_pp, arrival_ps=arrival_ps,
    arrival_ss=arrival_ss, left_arrival_pp=left_arrival_pp,
    source_depth=source_depth, source_peak_time=source_peak_time,
    profile_x=coords[trace_index, 0, 0], source_x=source_x,
    candidate_depths=candidate_depths, candidate_times=candidate_times,
    candidate_names=np.array(candidate_names), boundaries=boundaries,
    formations=formations, horizon_names=horizon_names, horizon_depths=horizon_depths,
    times=times, trace_vy=seismo_vy[:, receiver_index],
    trace_vx=seismo_vx[:, receiver_index], peak_times=times[peak_indices[:4]],
)
report = {
    "source_peak_time_s": source_peak_time,
    "source_depth_m": source_depth,
    "source_leg_s": source_leg,
    "depth_window_corrected_m": depth_window.tolist(),
    "depth_window_uncorrected_m": uncorrected_depth_window.tolist(),
    "candidates": [dict(name=name, depth_m=float(depth), arrival_pp_s=float(arrival))
                   for name, depth, arrival in zip(candidate_names, candidate_depths, candidate_times)],
    "left_column_max_time_difference_s": float(np.max(np.abs(left_arrival_pp - arrival_pp))),
    "interpretation": "Локальный кандидат около 2490–2500 м внутри слоя 75; "
                      "одномерная оценка не доказывает происхождение события.",
}
(data_dir / f"{output_prefix}.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


""" ## 6. Визуальное сопоставление профиля и сейсмограммы ## """

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 4, figsize=(16, 8), layout="constrained")
axes[0].plot(vp / 1000, depth_centers, label="Vp")
axes[0].plot(vs / 1000, depth_centers, label="Vs", color="0.5")
axes[0].set(xlabel="Скорость, км/с", ylabel="Глубина, м", title="Скорости")
axes[0].legend()
axes[1].plot(impedance / 1e6, depth_centers, color="tab:blue")
axes[1].set(xlabel="ρVp, МПа·с/м", title="P-импеданс")
axes[2].plot(two_way_p, depth_edges, color="0.6", ls="--", label="2∫ dz/Vp")
axes[2].plot(arrival_pp, depth_edges, color="tab:blue", label="С учётом источника")
axes[2].axvspan(2.1, 2.2, color="0.9")
axes[2].set(xlabel="Время, с", title="Время → глубина", xlim=(1.96, 2.29))
axes[2].legend(fontsize=8, loc="upper left")
for ax in axes[:3]:
    ax.set_ylim(2750, 2200)
    ax.axhspan(2485, 2505, color="tab:orange", alpha=0.17)
    for depth in boundaries[72:74]:
        ax.axhline(depth, color="0.5", ls=":", lw=0.9)
    ax.grid(alpha=0.18)
axes[1].annotate("73/74", (7.5, boundaries[72]), xytext=(7.0, 2370),
                 arrowprops={"arrowstyle": "-", "color": "0.4"})
axes[1].annotate("74/75", (9.3, boundaries[73]), xytext=(9.8, 2440),
                 arrowprops={"arrowstyle": "-", "color": "0.4"})
axes[1].text(7.1, 2518, "Переход внутри\nслоя 75", fontsize=9)
trace = seismo_vy[:, receiver_index]
axes[3].plot(trace, times, color="0.15", lw=1.1, label="Vy, x = 6000 м")
axes[3].axhspan(candidate_times[3], candidate_times[5], color="tab:orange", alpha=0.22,
                label="PP: переход 2490–2500 м")
axes[3].axhspan(candidate_times[0], candidate_times[1], color="tab:blue", alpha=0.16,
                label="PP: границы 73/74, 74/75")
axes[3].axhline(candidate_times[-1], color="0.5", ls="--", label="PP: дно 2750 м")
axes[3].set(ylim=(2.28, 2.04), xlim=(-0.035, 0.035), xlabel="Vy, единицы исходного NPZ",
            ylabel="Время записи, с", title="Трасса у источника")
axes[3].legend(fontsize=8, loc="lower left")
axes[3].grid(alpha=0.18)
fig.suptitle("x = 6000 м: кандидат на отражение — переход около 2.50 км внутри слоя 75\n"
             "Рикер: 30 Гц, максимум 0.03333 с; источник: 5 м. "
             "Полосы PP относятся к максимуму давления, а не к экстремуму Vy.", fontsize=13)
fig.savefig(image_dir / f"{output_prefix}.png", dpi=170)
plt.close(fig)


""" ## Выводы ## """

"""
Выполнено интегрирование медленности P и S, рассчитан P-импеданс и
сопоставлены глубины со слоями и горизонтами. После учёта источника окно
2.1–2.2 с соответствует примерно 2415–2628 м. Главные экстремумы Vy у
источника на 2.146 и 2.160 с формально дают 2511 и 2541 м; они не являются
двумя отдельными отражателями. В этой части разреза подходящий локальный
контраст расположен несколько выше, около 2490–2500 м внутри слоя 75.
Его расчётная привязка 2.136–2.141 с отличается от экстремумов Vy на
5–24 мс: требуются учёт фазы отклика и двумерной кинематики для подтверждения.
Границы 73/74 и 74/75 дают около 2.093–2.098 с, техническое дно — 2.254 с.
Численные профили, параметры привязки и рисунок сохранены в data/ и img/.
"""
