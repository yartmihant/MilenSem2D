""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть вторая. """

""" ## Глава I: Модель для геофизиков """

"""
### Научная задача

Наше исследование заключается в построении конечноэлементной и конечноразностной моделей с последующим совместным их решением.

Мы уже построили конечноэлементную модель для программы CAE-Fidesys
Теперь нужно построить аналогичную конечно-разностную двумерную модель задачи в файле формата, совместимого с tesseral.

Дополнительно создаем отдельный вариант SEG-Y на узловой сетке без сдвига на половину ячейки и количественно сравниваем фактический узловой материал Fidesys с кусочно-постоянным растром Tesseral.
"""

import base64
import csv
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy.spatial import cKDTree

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

""" #### Задача 3: Экспорт данных в CSV формат #### """

def export_material_to_csv(filename, material_grid, coords_grid, layer_indexes_grid):
    """
    Экспортирует данные материала в CSV формат с сортировкой по X, затем по Z.

    Args:
        filename: имя выходного CSV файла
        material_grid: массив материала [nx, ny, 3] (E, nu, rho)
        coords_grid: массив координат [nx, ny, 2] (x, z)
        layer_indexes_grid: массив номеров слоев [nx, ny]
    """
    nx, ny = material_grid.shape[:2]

    # Создаем список всех ячеек
    cells_data = []

    for i in range(nx):
        for j in range(ny):
            cell_idx = i * ny + j + 1  # Порядковый индекс начиная с 1
            x_coord = coords_grid[i, j, 0]
            z_coord = coords_grid[i, j, 1]
            layer_num = layer_indexes_grid[i, j]
            E_modulus = material_grid[i, j, 0]
            poisson_ratio = material_grid[i, j, 1]
            density = material_grid[i, j, 2]

            cells_data.append([
                cell_idx, x_coord, z_coord, layer_num,
                E_modulus, poisson_ratio, density
            ])

    # Сортируем по X, затем по Z
    cells_data.sort(key=lambda x: (x[1], x[2]))

    # Записываем в CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Заголовки
        writer.writerow([
            'Порядковый индекс ячейки',
            'Координата по X',
            'Координата по Z',
            'Номер слоя',
            'Модуль Юнга',
            'Коэффициент Пуассона',
            'Плотность'
        ])

        # Данные
        for row in cells_data:
            writer.writerow(row)

    print(f"Экспортировано {len(cells_data)} ячеек в файл: {filename}")

# Экспортируем данные в CSV
csv_output = output_dir / 'dev_2_1_material_data.csv'
export_material_to_csv(str(csv_output), material_grid, coords_grid, layer_indexes_grid)

# Проверяем результат
print(f"\nПроверка экспорта:")
print(f"  Файл: {csv_output}")
print(f"  Существует: {csv_output.exists()}")

if csv_output.exists():
    # Читаем первые и последние несколько строк
    with open(csv_output, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"  Всего строк: {len(lines)}")
        print("  Первые 5 строк:")
        for i in range(min(6, len(lines))):  # header + 5 строк
            print(f"    {lines[i].strip()}")
        if len(lines) > 10:
            print("  Последние 5 строк:")
            for i in range(len(lines)-5, len(lines)):
                print(f"    {lines[i].strip()}")


""" #### Задача 4: Создание отдельного SEG-Y без сдвига на половину ячейки #### """

"""
Исходный NPZ хранит значения в центрах ячеек с координатами 2.5, 7.5, ... м.
Стандартное поле начала вертикальных отсчетов SEG-Y не позволяет точно сохранить
дробное начало 2.5 м через `segyio`: после чтения ось начинается с 0 м.

Поэтому не изменяем существующие SEG-Y, а создаем отдельный узловой вариант.
Восстанавливаем кусочно-постоянное поле ячеек на узлах 0, 5, ... м. На общей
границе двух ячеек используем значение ячейки справа и снизу; на правой и нижней
границах модели продолжаем значение последней ячейки.
"""

def convert_cell_centers_to_nodes(data):
    """
    Переносит кусочно-постоянное поле с центров ячеек на узлы сетки.

    Args:
        data: массив значений в центрах ячеек [nx, ny]

    Returns:
        Массив узловых значений [nx + 1, ny + 1]
    """
    return np.pad(data, ((0, 1), (0, 1)), mode='edge')


Vp_grid_node_aligned = convert_cell_centers_to_nodes(Vp_grid)
Vs_grid_node_aligned = convert_cell_centers_to_nodes(Vs_grid)
density_grid_node_aligned = convert_cell_centers_to_nodes(density_grid)

vp_node_aligned_output = output_dir / 'dev_2_1_Vp_model_node_aligned.sgy'
vs_node_aligned_output = output_dir / 'dev_2_1_Vs_model_node_aligned.sgy'
density_node_aligned_output = output_dir / 'dev_2_1_Density_model_node_aligned.sgy'

print("\n" + "="*60)
print("ЗАПИСЬ ОТДЕЛЬНЫХ SEG-Y НА УЗЛОВОЙ СЕТКЕ")
print("="*60 + "\n")

write_segy_2d(
    str(vp_node_aligned_output),
    Vp_grid_node_aligned,
    dx=dx,
    dy=dy,
    x_origin=0.0,
    y_origin=0.0,
)
write_segy_2d(
    str(vs_node_aligned_output),
    Vs_grid_node_aligned,
    dx=dx,
    dy=dy,
    x_origin=0.0,
    y_origin=0.0,
)
write_segy_2d(
    str(density_node_aligned_output),
    density_grid_node_aligned,
    dx=dx,
    dy=dy,
    x_origin=0.0,
    y_origin=0.0,
)

for param_name, output_file, expected_data in [
    ('Vp', vp_node_aligned_output, Vp_grid_node_aligned),
    ('Vs', vs_node_aligned_output, Vs_grid_node_aligned),
    ('Density', density_node_aligned_output, density_grid_node_aligned),
]:
    with segyio.open(str(output_file), 'r', ignore_geometry=True) as f:
        data_check = segyio.tools.collect(f.trace[:])
        x_first = f.header[0][segyio.TraceField.CDP_X]
        x_last = f.header[f.tracecount - 1][segyio.TraceField.CDP_X]

        assert data_check.shape == expected_data.shape
        assert np.allclose(data_check, expected_data.astype(np.float32))
        assert x_first == 0
        assert x_last == 11750
        assert np.isclose(f.samples[0], 0.0)
        assert np.isclose(f.samples[-1], 2750.0)

        print(f"{param_name}: проверена узловая сетка {data_check.shape}")
        print(f"  X = {x_first} - {x_last} м")
        print(f"  Y = {f.samples[0]:.1f} - {f.samples[-1]:.1f} м")


""" #### Задача 5: Сравнение материалов Fidesys и Tesseral #### """

r"""
Сравниваем не исходный массив сам с собой, а материал, фактически вычисленный
Fidesys в узлах спектрально-элементной сетки, с кусочно-постоянным растром
Tesseral. Для каждого узла Fidesys ищем ближайший центр ячейки только внутри
того же геологического слоя. Подписанное относительное различие вычисляем как

$$
\delta_q = \frac{q_{Fidesys} - q_{Tesseral}}{q_{Tesseral}}\,100\%,
$$

где $q$ — $V_p$, $V_s$ или плотность. Такое сравнение не смешивает материалы
по разные стороны границы слоя и показывает различие, создаваемое интерполяцией
табличного материала Fidesys и растрированием материала Tesseral.
"""

def read_fidesys_vtu_material(filename):
    """
    Читает координаты и узловой материал из сжатого VTU файла Fidesys.

    Поддерживает используемый Fidesys формат VTK XML с base64-кодированными
    блоками `vtkZLibDataCompressor`. Поле строковых метаданных не разбирается.

    Args:
        filename: путь к VTU файлу результата Fidesys

    Returns:
        points: координаты узлов [n, 2]
        elasticity: модуль Юнга и коэффициент Пуассона [n, 2]
        common: плотность и массовое демпфирование [n, 2]
    """
    root = ET.parse(filename).getroot()

    if root.attrib.get('compressor') != 'vtkZLibDataCompressor':
        raise ValueError(f"Неподдерживаемое сжатие VTU: {filename}")

    byte_order = '<' if root.attrib.get('byte_order') == 'LittleEndian' else '>'
    header_type_name = root.attrib.get('header_type', 'UInt32')
    header_types = {'UInt32': 'u4', 'UInt64': 'u8'}
    value_types = {
        'Float32': 'f4',
        'Float64': 'f8',
        'Int32': 'i4',
        'Int64': 'i8',
        'UInt32': 'u4',
        'UInt64': 'u8',
    }

    if header_type_name not in header_types:
        raise ValueError(f"Неподдерживаемый тип заголовка VTU: {header_type_name}")

    appended = root.find('AppendedData')
    if appended is None or appended.text is None:
        raise ValueError(f"В VTU отсутствуют appended data: {filename}")

    encoded_data = ''.join(appended.text.split())
    if not encoded_data.startswith('_'):
        raise ValueError(f"Некорректный префикс appended data: {filename}")
    encoded_data = encoded_data[1:]

    header_dtype = np.dtype(byte_order + header_types[header_type_name])

    def decode_data_array(data_array):
        """Декодирует один сжатый appended DataArray."""
        offset = int(data_array.attrib['offset'])
        encoded_array = encoded_data[offset:]

        first_header_chars = 4 * ((header_dtype.itemsize + 2) // 3)
        first_header = base64.b64decode(encoded_array[:first_header_chars])
        number_of_blocks = int(
            np.frombuffer(first_header, dtype=header_dtype, count=1)[0]
        )

        number_of_header_values = 3 + number_of_blocks
        number_of_header_bytes = number_of_header_values * header_dtype.itemsize
        number_of_header_chars = 4 * ((number_of_header_bytes + 2) // 3)
        header_bytes = base64.b64decode(encoded_array[:number_of_header_chars])
        header = np.frombuffer(
            header_bytes[:number_of_header_bytes],
            dtype=header_dtype,
            count=number_of_header_values,
        )

        compressed_bytes = base64.b64decode(encoded_array[number_of_header_chars:])
        blocks = []
        block_start = 0
        for block_size in header[3:].astype(int):
            block_end = block_start + block_size
            blocks.append(zlib.decompress(compressed_bytes[block_start:block_end]))
            block_start = block_end

        value_type_name = data_array.attrib['type']
        if value_type_name not in value_types:
            raise ValueError(f"Неподдерживаемый тип данных VTU: {value_type_name}")

        value_dtype = np.dtype(byte_order + value_types[value_type_name])
        values = np.frombuffer(b''.join(blocks), dtype=value_dtype)
        number_of_components = int(data_array.attrib.get('NumberOfComponents', 1))

        if number_of_components > 1:
            values = values.reshape((-1, number_of_components))
        return values

    points_array = root.find('.//Points/DataArray')
    point_data_arrays = {
        item.attrib.get('Name'): item
        for item in root.findall('.//PointData/DataArray')
    }

    required_names = {'Elasticity Modulus', 'Common Modulus'}
    missing_names = required_names.difference(point_data_arrays)
    if points_array is None or missing_names:
        raise ValueError(
            f"В VTU отсутствуют необходимые массивы {sorted(missing_names)}: {filename}"
        )

    points = decode_data_array(points_array)[:, :2]
    elasticity = decode_data_array(point_data_arrays['Elasticity Modulus'])
    common = decode_data_array(point_data_arrays['Common Modulus'])

    if not (len(points) == len(elasticity) == len(common)):
        raise ValueError(f"Размеры узловых массивов VTU не совпадают: {filename}")

    return points, elasticity, common


def calculate_wave_speeds(young_modulus, poisson_ratio, density):
    """
    Вычисляет скорости продольных и поперечных волн из E, nu и rho.

    Args:
        young_modulus: модуль Юнга, Па
        poisson_ratio: коэффициент Пуассона
        density: плотность, кг/м³

    Returns:
        vp: скорость продольной волны, м/с
        vs: скорость поперечной волны, м/с
    """
    vp = np.sqrt(
        young_modulus / density * (1 - poisson_ratio)
        / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    vs = np.sqrt(
        young_modulus / (2 * density * (1 + poisson_ratio))
    )
    return vp, vs


def compare_fidesys_and_tesseral_materials(
    fidesys_directory,
    coords_grid,
    layer_indexes_grid,
    vp_grid,
    vs_grid,
    rho_grid,
):
    """
    Сравнивает узловой материал Fidesys с растром Tesseral внутри каждого слоя.

    Args:
        fidesys_directory: каталог VTU файлов Fidesys
        coords_grid: координаты центров ячеек Tesseral [nx, ny, 2]
        layer_indexes_grid: номера слоев Tesseral [nx, ny]
        vp_grid: поле Vp Tesseral [nx, ny]
        vs_grid: поле Vs Tesseral [nx, ny]
        rho_grid: поле плотности Tesseral [nx, ny]

    Returns:
        coordinates: координаты узлов Fidesys [n, 2]
        layer_indexes: номера слоев Fidesys [n]
        fidesys_values: Vp, Vs, rho в Fidesys [n, 3]
        tesseral_values: ближайшие Vp, Vs, rho Tesseral [n, 3]
        relative_difference_percent: подписанное различие, % [n, 3]
    """
    number_of_layers = int(layer_indexes_grid.max()) + 1
    expected_paths = [
        fidesys_directory / f'case1_step0000_part{layer_idx}.vtu'
        for layer_idx in range(number_of_layers)
    ]
    missing_paths = [path for path in expected_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Не найдены {len(missing_paths)} VTU файлов Fidesys, "
            f"первый отсутствующий файл: {missing_paths[0]}"
        )

    coordinates_parts = []
    layer_index_parts = []
    fidesys_parts = []
    tesseral_parts = []

    for layer_idx, vtu_path in enumerate(expected_paths):
        points, elasticity, common = read_fidesys_vtu_material(vtu_path)

        fidesys_density = common[:, 0]
        fidesys_vp, fidesys_vs = calculate_wave_speeds(
            elasticity[:, 0],
            elasticity[:, 1],
            fidesys_density,
        )
        fidesys_values = np.column_stack((
            fidesys_vp,
            fidesys_vs,
            fidesys_density,
        ))

        grid_indexes = np.argwhere(layer_indexes_grid == layer_idx)
        layer_grid_coordinates = coords_grid[
            grid_indexes[:, 0], grid_indexes[:, 1]
        ]
        nearest_tree = cKDTree(layer_grid_coordinates)
        _, nearest_indexes = nearest_tree.query(points)

        tesseral_values = np.column_stack((
            vp_grid[grid_indexes[:, 0], grid_indexes[:, 1]][nearest_indexes],
            vs_grid[grid_indexes[:, 0], grid_indexes[:, 1]][nearest_indexes],
            rho_grid[grid_indexes[:, 0], grid_indexes[:, 1]][nearest_indexes],
        ))

        coordinates_parts.append(points)
        layer_index_parts.append(np.full(len(points), layer_idx, dtype=np.int16))
        fidesys_parts.append(fidesys_values)
        tesseral_parts.append(tesseral_values)

    coordinates = np.vstack(coordinates_parts)
    layer_indexes = np.concatenate(layer_index_parts)
    fidesys_values = np.vstack(fidesys_parts)
    tesseral_values = np.vstack(tesseral_parts)
    relative_difference_percent = (
        100.0 * (fidesys_values - tesseral_values) / tesseral_values
    )

    if not np.all(np.isfinite(relative_difference_percent)):
        raise ValueError("Относительные различия содержат NaN или Inf")

    return (
        coordinates,
        layer_indexes,
        fidesys_values,
        tesseral_values,
        relative_difference_percent,
    )


fidesys_result_directory = Path('data/dev_1_7_model_material_full')
(
    comparison_coordinates,
    comparison_layer_indexes,
    fidesys_material_values,
    tesseral_material_values,
    relative_difference_percent,
) = compare_fidesys_and_tesseral_materials(
    fidesys_result_directory,
    coords_grid,
    layer_indexes_grid,
    Vp_grid,
    Vs_grid,
    rho_grid,
)

comparison_output = output_dir / 'dev_2_1_material_relative_difference.npz'
np.savez_compressed(
    comparison_output,
    coordinates=comparison_coordinates,
    layer_indexes=comparison_layer_indexes,
    fidesys_material=fidesys_material_values,
    tesseral_material=tesseral_material_values,
    relative_difference_percent=relative_difference_percent,
    parameter_names=np.array(['Vp_m_per_s', 'Vs_m_per_s', 'density_kg_per_m3']),
)
print(f"\nСравнение материалов сохранено: {comparison_output}")
print(f"Количество узлов Fidesys: {len(comparison_coordinates):,}")

parameter_titles = [
    r'$V_p$',
    r'$V_s$',
    r'$\rho$',
]

for parameter_idx, parameter_title in enumerate(parameter_titles):
    absolute_difference = np.abs(relative_difference_percent[:, parameter_idx])
    print(f"\n{parameter_title}:")
    print(f"  среднее подписанное: {relative_difference_percent[:, parameter_idx].mean():.4f}%")
    print(f"  среднее абсолютное: {absolute_difference.mean():.4f}%")
    print(f"  P95 абсолютного различия: {np.percentile(absolute_difference, 95):.4f}%")
    print(f"  P99 абсолютного различия: {np.percentile(absolute_difference, 99):.4f}%")
    print(f"  максимум абсолютного различия: {absolute_difference.max():.4f}%")


""" #### Визуализация относительных различий #### """

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(18, 12),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for parameter_idx, (axis, parameter_title) in enumerate(zip(axes, parameter_titles)):
    differences = relative_difference_percent[:, parameter_idx]
    color_limit = max(np.percentile(np.abs(differences), 99), 1e-12)
    clipped_differences = np.clip(differences, -color_limit, color_limit)

    scatter = axis.scatter(
        comparison_coordinates[:, 0],
        comparison_coordinates[:, 1],
        c=clipped_differences,
        s=0.25,
        linewidths=0,
        cmap='RdBu_r',
        vmin=-color_limit,
        vmax=color_limit,
        rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.01, extend='both')
    colorbar.set_label(r'$(Fidesys - Tesseral) / Tesseral$, %')

    absolute_difference = np.abs(differences)
    axis.set_title(
        f"{parameter_title}: среднее |δ| = {absolute_difference.mean():.3f}%, "
        f"P95 = {np.percentile(absolute_difference, 95):.3f}%, "
        f"P99 = {np.percentile(absolute_difference, 99):.3f}%"
    )
    axis.set_ylabel('Глубина, м')
    axis.set_xlim(0, 11750)

axes[0].set_ylim(2750, 0)
axes[-1].set_xlabel('Расстояние, м')
fig.suptitle(
    'Относительное различие узлового материала Fidesys и растра Tesseral',
    fontsize=15,
)

relative_difference_figure = Path('img/dev_2_1_material_relative_difference.png')
fig.savefig(relative_difference_figure, dpi=200, bbox_inches='tight')
print(f"Рисунок сохранен: {relative_difference_figure}")
plt.show()

"""
### Выводы

В Главе I части второй была выполнена подготовка данных для конечно-разностного моделирования в Tesseral:

1. **Анализ формата:** Изучена структура файлов SEG-Y из примера Tesseral
2. **Конвертация данных:** Преобразованы декартовы данные материала в формат SEG-Y
3. **Экспорт в CSV:** Создана таблица со всеми данными материала для дополнительного анализа
4. **Исправление привязки:** Созданы отдельные SEG-Y на узловой сетке 5×5 м без сдвига на половину ячейки; исходные файлы сохранены без изменений
5. **Сравнение сеток:** Построены карты подписанных относительных различий Vp, Vs и плотности между узловым материалом Fidesys и растром Tesseral
6. **Выходные файлы:**
   - data/dev_2_1_Vp_model.sgy - модель скоростей продольных волн
   - data/dev_2_1_Vs_model.sgy - модель скоростей поперечных волн
   - data/dev_2_1_Density_model.sgy - модель плотности
   - data/dev_2_1_Vp_model_node_aligned.sgy - отдельная узловая модель Vp
   - data/dev_2_1_Vs_model_node_aligned.sgy - отдельная узловая модель Vs
   - data/dev_2_1_Density_model_node_aligned.sgy - отдельная узловая модель плотности
   - data/dev_2_1_material_relative_difference.npz - данные сравнения материалов
   - data/dev_2_1_material_data.csv - полная таблица данных материала
   - img/dev_2_1_material_relative_difference.png - карты относительных различий

Данные готовы для использования в программе Tesseral для конечно-разностного моделирования.
"""
