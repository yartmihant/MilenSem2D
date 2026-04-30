""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть вторая. """

""" ## Глава III: Приведение синтетических сейсмограмм к формату Tesseral """

"""
### Задача

Синтетические сейсмограммы (data/dev_2_3/sgy и сырые data/dev_2_3/src) должны совпадать по компоновке и заголовкам с образцом из Tesseral (data/dev_2_3/example/21-01-26+GathAP-X.sgy).

Первый шаг: изучить заголовки и описать формат примера.
"""

import segyio
from pathlib import Path

# Путь к примеру (Tesseral)
example_sgy = Path('data/dev_2_3/example/21-01-26+GathAP-X.sgy')
output_dir = Path('data')
output_dir.mkdir(parents=True, exist_ok=True)

""" ## 1. Текстовый заголовок файла (3200 байт) ## """

def read_text_header(path):
    """Читает текстовый заголовок SEG-Y (40 строк по 80 символов)."""
    with open(path, 'rb') as f:
        raw = f.read(3200)
    try:
        text = raw.decode('cp500')
    except Exception:
        text = raw.decode('ascii', errors='replace')
    lines = [text[i:i+80] for i in range(0, 3200, 80)]
    return lines

text_lines = read_text_header(example_sgy)
print("Текстовый заголовок (первые 40 строк по 80 символов):")
print("-" * 60)
for i, line in enumerate(text_lines):
    if line.strip():
        print(f"{i+1:2}: {line.rstrip()}")

""" ## 2. Бинарный заголовок файла ## """

with segyio.open(str(example_sgy), 'r', ignore_geometry=True) as f:
    bin_header = f.bin
    trace_count = f.tracecount
    samples_per_trace = len(f.trace[0])
    data_format = f.format

bin_field_names = [x for x in dir(segyio.BinField) if not x.startswith('_') and x not in ('enums',)]
print("\nБинарный заголовок (все поля):")
print("-" * 60)
for name in sorted(bin_field_names):
    try:
        field = getattr(segyio.BinField, name)
        if isinstance(field, int):
            val = bin_header[field]
            print(f"  {name}: {val}")
    except Exception as e:
        print(f"  {name}: (ошибка: {e})")

print(f"\n  [сводка] Трасс: {trace_count}, сэмплов на трассу: {samples_per_trace}, формат данных: {data_format}")

""" ## 3. Заголовки трасс (Trace Headers) ## """

# Имена полей трассового заголовка (стандарт SEG-Y)
trace_field_names = [x for x in dir(segyio.TraceField) if not x.startswith('_') and x not in ('enums', 'offset')]

def trace_header_dict(f, trace_idx):
    """Словарь всех полей трассового заголовка для одной трассы."""
    header = f.header[trace_idx]
    out = {}
    for name in sorted(trace_field_names):
        try:
            field = getattr(segyio.TraceField, name)
            if isinstance(field, int):
                out[name] = header[field]
        except Exception:
            pass
    return out

with segyio.open(str(example_sgy), 'r', ignore_geometry=True) as f:
    indices = [0, trace_count // 2, trace_count - 1] if trace_count > 2 else list(range(trace_count))
    print("\nЗаголовки трасс (трассы 0, середина, последняя):")
    print("-" * 60)
    for idx in indices:
        h = trace_header_dict(f, idx)
        non_zero = {k: v for k, v in h.items() if v != 0}
        print(f"\n  Трасса {idx}: всего полей {len(h)}, ненулевых: {len(non_zero)}")
        for k, v in sorted(non_zero.items()):
            print(f"    {k}: {v}")

""" ## 4. Сводная таблица полей трассового заголовка (все трассы) ## """

with segyio.open(str(example_sgy), 'r', ignore_geometry=True) as f:
    first = trace_header_dict(f, 0)
    last = trace_header_dict(f, f.tracecount - 1)
    # Поля, которые меняются по трассам
    varying = [k for k in first if first[k] != last.get(k, None)]
    constant = [k for k in first if k not in varying]

print("\nПоля трассового заголовка:")
print("  Постоянные по файлу (одинаковые для всех трасс):")
for k in sorted(constant):
    if first[k] != 0:
        print(f"    {k}: {first[k]}")
print("  Меняющиеся по трассам:")
for k in sorted(varying):
    print(f"    {k}: первая={first[k]}, последняя={last[k]}")

""" ## 5. Сохранение описания формата в файл ## """

report_path = output_dir / 'dev_2_3_1_example_headers.md'

report_lines = [
    "# Описание формата SEG-Y примера Tesseral",
    "",
    f"Файл: `{example_sgy}`",
    "",
    "## Текстовый заголовок (File Header)",
    "",
    "```",
]
report_lines.extend([ln.rstrip() for ln in text_lines if ln.strip()])
report_lines.append("```")
report_lines.extend([
    "",
    "## Бинарный заголовок",
    "",
    f"- Трасс в файле: {trace_count}",
    f"- Сэмплов на трассу: {samples_per_trace}",
    f"- Формат данных (код): {data_format} (1=4-byte IBM float, 5=4-byte IEEE float)",
    "",
])

with segyio.open(str(example_sgy), 'r', ignore_geometry=True) as f:
    bh = f.bin
    for name in sorted(bin_field_names):
        try:
            field = getattr(segyio.BinField, name)
            if isinstance(field, int):
                report_lines.append(f"- **{name}**: {bh[field]}")
        except Exception:
            pass

report_lines.extend([
    "",
    "## Трассовый заголовок (Trace Header)",
    "",
    "### Постоянные поля (одинаковые для всех трасс)",
    "",
])
for k in sorted(constant):
    if first.get(k, 0) != 0:
        report_lines.append(f"- **{k}**: {first[k]}")

report_lines.extend([
    "",
    "### Поля, меняющиеся по трассам",
    "",
])
for k in sorted(varying):
    report_lines.append(f"- **{k}**: от {first[k]} до {last[k]}")

report_lines.append("")
report_lines.append("---")
report_lines.append("Сгенерировано скриптом dev_2_3_1_example_headers.py")

report_text = "\n".join(report_lines)
report_path.write_text(report_text, encoding='utf-8')
print(f"\nОписание формата сохранено: {report_path}")

"""
### Вывод по главе

Изучены заголовки примера SEG-Y из Tesseral: текстовый (3200 байт), бинарный и трассовые.
Сводка записана в data/dev_2_3_1_example_headers.md для использования при приведении наших сейсмограмм к этому формату.
"""

""" ## Глава III.2: Сборка сводных сейсмограмм из NPZ → SEG-Y """

"""
### Задача

На основе анализа заголовков (глава III.1) разработан скрипт `dev_2_3_2_merge_shots.py`,
который конвертирует данные численного моделирования из формата NPZ в SEG-Y с корректными
заголовками, совместимыми с форматом `output_line1_Vy/Vx_gather_merged.sgy`.

### Источник данных

Каждый расчётный источник хранится в отдельной папке:
    data/dev_2_3/research_seismic_sweep/x_{pos}/data.npz

Ключевые поля NPZ:
- `sensor_x`    — координаты сейсмоприёмников (0..11750 м, шаг 10 м, 1176 точек)
- `seismo_vx`   — компонента скорости Vx: (2000, 1176), float64
- `seismo_vy`   — компонента скорости Vy: (2000, 1176), float64
- `seismo_times`— моменты записи (2000 точек, шаг ~2002 мкс)

### Принятые решения по параметрам сейсмограммы

- В SEG-Y записываются **первые 1501 отсчётов** из 2000 имеющихся.
  Если отсчётов меньше 1501 — остаток дополняется нулями.
- Шаг дискретизации в заголовке SEG-Y: **2000 мкс** (ровно).
- Формат данных: **4-byte IBM float** (Format=1), как в образце.

### Конфигурация наблюдений

Правый шаблон (right spread):
- Источники: x = 250..7450 м, шаг 50 м → 145 источников
- Приёмники: 80 точек вправо от источника через 50 м
- Диапазон приёмников: 300..11450 м

Левый шаблон (left spread):
- Источники: x = 4300..11500 м, шаг 50 м → 145 источников
- Приёмники: 80 точек влево от источника через 50 м
- Диапазон приёмников: 300..11450 м

Итого: 4 файла (Vx/Vy × right/left), по 11600 трасс каждый.

### Заполнение заголовков SEG-Y (ключевые поля)

Приведены в соответствие с образцом `output_line1_Vy_gather_merged.sgy`:
- SourceGroupScalar = ElevationScalar = -100  (координаты × 100 = сантиметры)
- SourceX, GroupX → x × 100
- offset = GroupX_m - SourceX_m  (метры, со знаком)
- CDP_X = midpoint × 100
- ReceiverGroupElevation = -8  (поверхность: -0.08 м)
- TRACE_SAMPLE_INTERVAL = 2000 мкс

Все параметры вынесены в глобальный словарь CONFIG внутри скрипта.

### Запуск скрипта

    cd /path/to/MilenSem2D
    python dev_2_3_2_merge_shots.py

Результаты записываются в:
    data/dev_2_3/output_line1_Vx_right.sgy
    data/dev_2_3/output_line1_Vx_left.sgy
    data/dev_2_3/output_line1_Vy_right.sgy
    data/dev_2_3/output_line1_Vy_left.sgy
"""