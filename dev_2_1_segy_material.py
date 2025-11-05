""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть вторая. """

""" ## Глава I: Модель для геофизиков """

"""
### Научная задача

Наше исследование заключается в построении конечноэлементной и конечноразностной моделей с последующим совместным их решением.

Мы уже построили конечноэлементную модель для программы CAE-Fidesys
Теперь нужно построить аналогичную конечно-разностную двумерную модель задачи в файле формата, совместимого с tesseral.
"""

import numpy as np
import segyio
from pathlib import Path

""" #### Задача 1: Анализ формата данных Tesseral #### """

# Анализируем пример из Tesseral
example_vp_path = 'src/tesseral_example/Vp_model.sgy'

print("="*60)
print("АНАЛИЗ ФОРМАТА SEG-Y ИЗ TESSERAL")
print("="*60)

with segyio.open(example_vp_path, 'r', ignore_geometry=True) as f:
    print(f"\nФайл: {example_vp_path}")
    print(f"Количество трасс: {f.tracecount}")
    print(f"Количество сэмплов на трассу: {len(f.trace[0])}")
    print(f"Формат данных: {f.format}")
    
    # Читаем заголовок первой трассы
    print(f"\nЗаголовок первой трассы:")
    header = f.header[0]
    print(f"  Inline: {header[segyio.TraceField.INLINE_3D]}")
    print(f"  Crossline: {header[segyio.TraceField.CROSSLINE_3D]}")
    print(f"  CDP X: {header[segyio.TraceField.CDP_X]}")
    print(f"  CDP Y: {header[segyio.TraceField.CDP_Y]}")
    print(f"  Sample interval: {header[segyio.TraceField.TRACE_SAMPLE_INTERVAL]}")
    
    # Читаем бинарный заголовок
    print(f"\nБинарный заголовок:")
    bin_header = f.bin
    print(f"  Job ID: {bin_header[segyio.BinField.JobID]}")
    print(f"  Samples: {bin_header[segyio.BinField.Samples]}")
    print(f"  Interval: {bin_header[segyio.BinField.Interval]}")
    print(f"  Format: {bin_header[segyio.BinField.Format]}")
    
    # Читаем данные первой трассы
    trace_0 = f.trace[0]
    print(f"\nДанные первой трассы:")
    print(f"  Минимум: {trace_0.min():.2f}")
    print(f"  Максимум: {trace_0.max():.2f}")
    print(f"  Среднее: {trace_0.mean():.2f}")
    print(f"  Первые 5 значений: {trace_0[:5]}")

# Анализируем все три файла
for param_name in ['Vp', 'Vs', 'Density']:
    param_path = f'src/tesseral_example/{param_name}_model.sgy'
    with segyio.open(param_path, 'r', ignore_geometry=True) as f:
        data = segyio.tools.collect(f.trace[:])
        print(f"\n{param_name}:")
        print(f"  Форма данных: {data.shape}")
        print(f"  Диапазон: [{data.min():.2f}, {data.max():.2f}]")

""" #### Задача 2: Подготовка и выгрузка наших данных #### """

print("\n" + "="*60)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ TESSERAL")
print("="*60)

# Загружаем наши декартовы данные
data_path = Path('data/dev_1_7_material_grids.npz')
data = np.load(data_path)

material_grid = data['material_grid']      # [2350, 550, 3]
coords_grid = data['coords_grid']          # [2350, 550, 2]
layer_indexes_grid = data['layer_indexes_grid']  # [2350, 550]

print(f"\nЗагружены данные из {data_path}")
print(f"Размер сетки материала: {material_grid.shape}")
print(f"Размер сетки координат: {coords_grid.shape}")

# Извлекаем свойства
E_grid = material_grid[:, :, 0]   # Модуль Юнга (Па)
nu_grid = material_grid[:, :, 1]  # Коэффициент Пуассона
rho_grid = material_grid[:, :, 2] # Плотность (кг/м³)

# Вычисляем скорости волн
Vp_grid = np.sqrt(E_grid / rho_grid * (1 - nu_grid) / 
                  ((1 + nu_grid) * (1 - 2*nu_grid)))
Vs_grid = np.sqrt(E_grid / (2 * rho_grid * (1 + nu_grid)))

# Плотность в г/см³ (стандарт для геофизики)
density_grid = rho_grid / 1000.0

print(f"\nДиапазоны параметров:")
print(f"  Vp: {Vp_grid.min():.1f} - {Vp_grid.max():.1f} м/с")
print(f"  Vs: {Vs_grid.min():.1f} - {Vs_grid.max():.1f} м/с")
print(f"  Плотность: {density_grid.min():.3f} - {density_grid.max():.3f} г/см³")

# Получаем параметры сетки
x_coords = coords_grid[:, 0, 0]  # Координаты X
y_coords = coords_grid[0, :, 1]  # Координаты Y (глубины)

dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 5.0
dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 5.0

print(f"\nПараметры сетки:")
print(f"  Nx = {len(x_coords)}, Ny = {len(y_coords)}")
print(f"  dx = {dx:.1f} м, dy = {dy:.1f} м")
print(f"  X: {x_coords[0]:.1f} - {x_coords[-1]:.1f} м")
print(f"  Y: {y_coords[0]:.1f} - {y_coords[-1]:.1f} м")

""" #### Функция для записи данных в формат SEG-Y #### """

def write_segy_2d(filename, data, dx=5.0, dy=5.0, x_origin=0.0, y_origin=0.0):
    """
    Записывает 2D массив данных в формат SEG-Y.
    
    Args:
        filename: имя выходного файла
        data: 2D массив данных [nx, ny]
        dx: шаг по X (м)
        dy: шаг по Y (м)  
        x_origin: начало координат по X (м)
        y_origin: начало координат по Y (м)
    """
    nx, ny = data.shape
    
    # Создаем спецификацию SEG-Y
    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = range(ny)  # индексы сэмплов
    spec.tracecount = nx  # количество трасс = количество точек по X
    
    # Интервал в микросекундах (для совместимости с сейсмикой)
    # Используем dy в метрах, умножаем на 1000 для перевода в мкс
    spec.interval = int(dy * 1000)
    
    with segyio.create(filename, spec) as f:
        # Записываем бинарный заголовок
        f.bin = {
            segyio.BinField.JobID: 1,
            segyio.BinField.Samples: ny,
            segyio.BinField.Interval: spec.interval,
            segyio.BinField.Format: 5  # IEEE float
        }
        
        # Записываем трассы
        for i in range(nx):
            # Заголовок трассы
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.INLINE_3D: i + 1,
                segyio.TraceField.CROSSLINE_3D: 1,
                segyio.TraceField.CDP_X: int(x_origin + i * dx),
                segyio.TraceField.CDP_Y: int(y_origin),
                segyio.TraceField.TRACE_SAMPLE_COUNT: ny,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: spec.interval
            }
            
            # Данные трассы (столбец по Y)
            f.trace[i] = data[i, :].astype(np.float32)
    
    print(f"  Записан файл: {filename}")
    print(f"    Размер: {nx} × {ny}")
    print(f"    Диапазон: [{data.min():.2f}, {data.max():.2f}]")

""" #### Запись данных в файлы SEG-Y #### """

print("\n" + "="*60)
print("ЗАПИСЬ ДАННЫХ В ФОРМАТ SEG-Y")
print("="*60 + "\n")

output_dir = Path('data')

# Записываем Vp
vp_output = output_dir / 'dev_2_1_Vp_model.sgy'
write_segy_2d(str(vp_output), Vp_grid, dx=dx, dy=dy)

# Записываем Vs
vs_output = output_dir / 'dev_2_1_Vs_model.sgy'
write_segy_2d(str(vs_output), Vs_grid, dx=dx, dy=dy)

# Записываем плотность
density_output = output_dir / 'dev_2_1_Density_model.sgy'
write_segy_2d(str(density_output), density_grid, dx=dx, dy=dy)

print("\n" + "="*60)
print("ПРОВЕРКА ЗАПИСАННЫХ ФАЙЛОВ")
print("="*60)

# Проверяем записанные файлы
for param_name, output_file in [('Vp', vp_output), ('Vs', vs_output), ('Density', density_output)]:
    with segyio.open(str(output_file), 'r', ignore_geometry=True) as f:
        data_check = segyio.tools.collect(f.trace[:])
        print(f"\n{param_name} (проверка):")
        print(f"  Файл: {output_file.name}")
        print(f"  Форма: {data_check.shape}")
        print(f"  Диапазон: [{data_check.min():.2f}, {data_check.max():.2f}]")
        print(f"  Количество трасс: {f.tracecount}")
        print(f"  Сэмплов на трассу: {len(f.trace[0])}")

print("\n" + "="*60)
print("ЗАВЕРШЕНО")
print("="*60)

"""
### Выводы

В Главе VIII была выполнена подготовка данных для конечно-разностного моделирования в Tesseral:

1. **Анализ формата:** Изучена структура файлов SEG-Y из примера Tesseral
2. **Конвертация данных:** Преобразованы декартовы данные материала в формат SEG-Y
3. **Выходные файлы:** Созданы три файла для Tesseral:
   - data/dev_1_8_Vp_model.sgy - модель скоростей продольных волн
   - data/dev_1_8_Vs_model.sgy - модель скоростей поперечных волн
   - data/dev_1_8_Density_model.sgy - модель плотности

Данные готовы для использования в программе Tesseral для конечно-разностного моделирования.
"""