""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть третья. """

""" ## Глава V: SEG-Y модель для Tesseral """

"""
### Задача

Конвертировать декартову сетку материала модели с разломом в формат SEG-Y
для загрузки в программу Tesseral (конечно-разностное моделирование).

Аналог Главы II.1, но для модели с разломом (расширенная глубина до 2800 м).

Входные данные:
- `data/dev_3_2_fault_material.npz` — материал на декартовой сетке 5×5 м

Выходные данные:
- `data/dev_3_5_fault_Vp.sgy` — скорость P-волн (м/с)
- `data/dev_3_5_fault_Vs.sgy` — скорость S-волн (м/с)
- `data/dev_3_5_fault_Density.sgy` — плотность (г/см³)
"""

import numpy as np
import segyio
from pathlib import Path


""" ## Загрузка данных ## """

mat_data = np.load('data/dev_3_2_fault_material.npz')
material_grid = mat_data['material_grid']   # (2350, 560, 3) — E, nu, rho
coords_grid = mat_data['coords_grid']       # (2350, 560, 2) — x, y
fault_x = float(mat_data['fault_x'])
fault_throw = float(mat_data['fault_throw'])
model_bottom_depth = float(mat_data['model_bottom_depth'])

# Извлекаем свойства
E_grid = material_grid[:, :, 0]   # Модуль Юнга (Па)
nu_grid = material_grid[:, :, 1]  # Коэффициент Пуассона
rho_grid = material_grid[:, :, 2] # Плотность (кг/м³)

# Вычисляем скорости волн
Vp_grid = np.sqrt(E_grid / rho_grid * (1 - nu_grid) /
                  ((1 + nu_grid) * (1 - 2 * nu_grid)))
Vs_grid = np.sqrt(E_grid / (2 * rho_grid * (1 + nu_grid)))

# Плотность в г/см³ (стандарт для геофизики)
density_grid = rho_grid / 1000.0

# Параметры сетки
x_coords = coords_grid[:, 0, 0]
y_coords = coords_grid[0, :, 1]
nx, ny = len(x_coords), len(y_coords)
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]

print(f"Материал загружен: {material_grid.shape}")
print(f"  Nx={nx}, Ny={ny}, dx={dx:.1f} м, dy={dy:.1f} м")
print(f"  X: {x_coords[0]:.1f} .. {x_coords[-1]:.1f} м")
print(f"  Y: {y_coords[0]:.1f} .. {y_coords[-1]:.1f} м")
print(f"\nДиапазоны:")
print(f"  Vp: {Vp_grid.min():.0f} .. {Vp_grid.max():.0f} м/с")
print(f"  Vs: {Vs_grid.min():.0f} .. {Vs_grid.max():.0f} м/с")
print(f"  ρ:  {density_grid.min():.3f} .. {density_grid.max():.3f} г/см³")


""" ## Функция записи SEG-Y ## """

def write_segy_2d(filename, data, dx, dy, x_origin=0.0):
    """
    Записывает 2D массив в формат SEG-Y (совместимый с Tesseral).

    Args:
        filename: путь к выходному файлу
        data: 2D массив [nx, ny] — трассы по X, сэмплы по Y (глубина)
        dx: шаг по X (м)
        dy: шаг по Y (м)
        x_origin: начало координат по X (м)
    """
    nx, ny = data.shape

    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = range(ny)
    spec.tracecount = nx
    spec.interval = int(dy * 1000)  # мкс (условно)

    with segyio.create(filename, spec) as f:
        f.bin = {
            segyio.BinField.JobID: 1,
            segyio.BinField.Samples: ny,
            segyio.BinField.Interval: spec.interval,
            segyio.BinField.Format: 5,
        }

        for i in range(nx):
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.INLINE_3D: i + 1,
                segyio.TraceField.CROSSLINE_3D: 1,
                segyio.TraceField.CDP_X: int(x_origin + i * dx),
                segyio.TraceField.CDP_Y: 0,
                segyio.TraceField.TRACE_SAMPLE_COUNT: ny,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: spec.interval,
            }
            f.trace[i] = data[i, :].astype(np.float32)

    file_size = Path(filename).stat().st_size / 1024 / 1024
    print(f"  {filename} — {nx}×{ny}, [{data.min():.1f}..{data.max():.1f}], {file_size:.1f} МБ")


""" ## Запись SEG-Y файлов ## """

print(f"\nЗапись SEG-Y файлов:")

write_segy_2d('data/dev_3_5_fault_Vp.sgy', Vp_grid, dx=dx, dy=dy, x_origin=x_coords[0])
write_segy_2d('data/dev_3_5_fault_Vs.sgy', Vs_grid, dx=dx, dy=dy, x_origin=x_coords[0])
write_segy_2d('data/dev_3_5_fault_Density.sgy', density_grid, dx=dx, dy=dy, x_origin=x_coords[0])


""" ## Проверка записанных файлов ## """

print(f"\nПроверка:")
for name, path in [('Vp', 'data/dev_3_5_fault_Vp.sgy'),
                   ('Vs', 'data/dev_3_5_fault_Vs.sgy'),
                   ('Density', 'data/dev_3_5_fault_Density.sgy')]:
    with segyio.open(path, 'r', ignore_geometry=True) as f:
        data_check = segyio.tools.collect(f.trace[:])
        header0 = f.header[0]
        print(f"  {name}: shape={data_check.shape}, "
              f"range=[{data_check.min():.2f}..{data_check.max():.2f}], "
              f"CDP_X[0]={header0[segyio.TraceField.CDP_X]}")


""" ## Визуализация ## """

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

for ax, data, title, label in zip(
    axes,
    [Vp_grid, Vs_grid, density_grid],
    ['Vp', 'Vs', 'Плотность'],
    ['м/с', 'м/с', 'г/см³']
):
    c = ax.pcolormesh(X, Y, data, cmap='rainbow', shading='auto')
    plt.colorbar(c, ax=ax, label=label)
    ax.axvline(fault_x, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Глубина (м)')
    ax.set_title(f'{title} (SEG-Y)')
    ax.set_ylim(model_bottom_depth, 0)

plt.tight_layout()
plt.savefig('img/dev_3_5_segy_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nСохранено: img/dev_3_5_segy_overview.png")


""" ## Сводка ## """

print(f"\n{'='*60}")
print(f"=== Сводка Главы V ===")
print(f"{'='*60}")
print(f"Модель: {nx}×{ny} (шаг {dx:.0f}×{dy:.0f} м)")
print(f"Глубина: 0 .. {model_bottom_depth:.0f} м")
print(f"Разлом: x={fault_x:.0f} м, throw={fault_throw:.0f} м")
print(f"\nФайлы SEG-Y:")
print(f"  data/dev_3_5_fault_Vp.sgy      — Vp [{Vp_grid.min():.0f}..{Vp_grid.max():.0f}] м/с")
print(f"  data/dev_3_5_fault_Vs.sgy      — Vs [{Vs_grid.min():.0f}..{Vs_grid.max():.0f}] м/с")
print(f"  data/dev_3_5_fault_Density.sgy — ρ  [{density_grid.min():.3f}..{density_grid.max():.3f}] г/см³")


""" ## Выводы ## """

"""
Выполнена конвертация декартовой сетки материала модели с разломом в формат SEG-Y:

1. Загружен материал из Главы III.2 (E, ν, ρ на сетке 5×5 м)
2. Вычислены сейсмические скорости Vp и Vs
3. Плотность конвертирована из кг/м³ в г/см³
4. Записаны 3 файла SEG-Y (формат IEEE float, трассы по X, сэмплы по глубине)
5. Файлы совместимы с Tesseral для конечно-разностного моделирования

Формат идентичен Главе II.1 — расширена только область по глубине (2800 м вместо 2750 м).
"""
