""" # Численное моделирование распространения сейсмических волн в двумерной среде MILEN SEM 2D. Часть вторая. """

""" ## Глава III: Приведение синтетических сейсмограмм к формату Tesseral """

"""
### Задача

Синтетические сейсмограммы (data/dev_2_3/sgy и сырые data/dev_2_3/src) должны совпадать по компоновке и заголовкам с образцом из Tesseral (data/dev_2_3/example/21-01-26+GathAP-X.sgy).

Первый шаг: изучить заголовки и описать формат примера.
Второй шаг. Собрать всё воедино.
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

"""
Сборка сводных сейсмограмм из NPZ-данных (research_seismic_sweep) в формат SEG-Y.

Два шаблона наблюдений (правый и левый), две компоненты (Vx, Vy) — итого 4 выходных файла.

Все параметры задаются глобальным словарём CONFIG (см. ниже).
"""

from pathlib import Path

import numpy as np
import segyio

"""# Глобальная конфигурация
"""

CONFIG = {
    # Каталог с NPZ-данными: research_seismic_sweep/x_{pos}/data.npz
    "npz_dir": Path("data/dev_2_3/research_seismic_sweep_7_1_0_full"),

    # Каталог для выходных SEG-Y
    "output_dir": Path("data/dev_2_3"),

    # Компоненты для записи
    "components": ["Vx", "Vy"],

    # -----------------------------------------------------------------------
    # Параметры сейсмограммы (время)
    # -----------------------------------------------------------------------
    # Число временных отсчётов в каждой трассе
    "sample_count": 1501,
    # Шаг дискретизации, мкс — записывается в заголовок SEG-Y
    "sample_interval_us": 2000,
    # Если в NPZ отсчётов БОЛЬШЕ sample_count — берём первые sample_count.
    # Если МЕНЬШЕ — дополняем нулями справа.

    # -----------------------------------------------------------------------
    # Правый шаблон (Right spread)
    # Источник -> приёмники вправо
    # -----------------------------------------------------------------------
    "right": {
        # Координаты источников, м: от 250 до 7500 шаг 50 -> 146 источников
        "source_x_start": 250,
        "source_x_stop":  7500,
        "source_x_step":  50,
        # Число приёмников, шаг (приёмники правее источника)
        "receiver_count": 81,
        "receiver_step_m": 50,
        # receiver_k: x_r = x_s + receiver_step_m * k, k = 1..receiver_count
        "receiver_direction": 1,
    },

    # -----------------------------------------------------------------------
    # Левый шаблон (Left spread)
    # Источник -> приёмники влево
    # -----------------------------------------------------------------------
    "left": {
        # Координаты источников, м: от 4250 до 11500 шаг 50 -> 146 источников
        "source_x_start": 4250,
        "source_x_stop":  11500,
        "source_x_step":  50,
        # Число приёмников, шаг (приёмники левее источника)
        "receiver_count": 81,
        "receiver_step_m": 50,
        # receiver_k: x_r = x_s - receiver_step_m * k, k = 1..receiver_count
        "receiver_direction": -1,
    },

    # -----------------------------------------------------------------------
    # Заголовки SEG-Y (трассовые константы)
    # -----------------------------------------------------------------------
    # Скаляр для координат: -100 означает "делить на 100" -> метры.
    # Координаты хранятся умноженными на 100 (сантиметры).
    "elevation_scalar":         -100,
    "source_group_scalar":      -100,
    # Отметка поверхности, сантиметры: -8 / 100 = -0.08 м ~ 0
    "receiver_group_elevation": -8,
    # CoordinateUnits = 1 -> метры
    "coordinate_units": 1,
    # TraceIdentificationCode = 1 -> живая трасса
    "trace_id_code": 1,
    # DataUse = 1 -> реальные данные
    "data_use": 1,
}


def load_npz(npz_dir: Path, source_x: int) -> dict:
    """Загружает NPZ-файл для источника source_x (в метрах)."""
    path = npz_dir / f"x_{source_x}" / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"NPZ не найден: {path}")
    return dict(np.load(path))


def extract_trace(
    data: dict,
    component: str,
    receiver_x: float,
    sample_count: int,
) -> np.ndarray:
    """
    Извлекает трассу для одного приёмника.

    Шаг sensor_x = 10 м, поэтому индекс = round(receiver_x / sensor_step).
    Возвращает float32-вектор длиной sample_count.
    """
    sensor_x = data["sensor_x"]   # (1176,) отсортированный 0..11750 шаг 10 м
    sensor_step = sensor_x[1] - sensor_x[0]
    ix = int(round(receiver_x / sensor_step))
    if ix < 0 or ix >= len(sensor_x):
        return np.zeros(sample_count, dtype=np.float32)

    key = f"seismo_{component.lower()}"  # seismo_vx или seismo_vy
    raw = data[key][:, ix]               # (2000,)

    n = len(raw)
    if n >= sample_count:
        return raw[:sample_count].astype(np.float32)
    else:
        out = np.zeros(sample_count, dtype=np.float32)
        out[:n] = raw.astype(np.float32)
        return out


def make_text_header(component: str, spread: str, cfg: dict) -> bytes:
    """Создаёт 3200-байтовый текстовый заголовок SEG-Y."""
    scfg = cfg[spread]
    n_src = len(range(scfg["source_x_start"], scfg["source_x_stop"] + 1, scfg["source_x_step"]))
    n_rec = scfg["receiver_count"]
    total = n_src * n_rec
    direction = "RIGHT" if scfg["receiver_direction"] == 1 else "LEFT"
    lines = {
        1:  "MILEN SEM 2D - SYNTHETIC GATHER",
        2:  "SEGY IN TIME FORMAT (NPZ -> SEG-Y CONVERSION)",
        3:  f"COMPONENT: {component}",
        4:  f"SPREAD: {spread.upper()} ({direction})",
        5:  f"SOURCES: {scfg['source_x_start']}..{scfg['source_x_stop']} M STEP {scfg['source_x_step']} M  ({n_src} SOURCES)",
        6:  f"RECEIVERS: {n_rec} PER SOURCE, STEP {scfg['receiver_step_m']} M  (TOTAL {total} TRACES)",
        7:  f"TIME: {cfg['sample_count']} SAMPLES X {cfg['sample_interval_us']} US",
        8:  "FORMAT: 4-BYTE IBM FLOAT (SEG-Y FORMAT 1)",
        9:  "COORD UNITS: METERS; STORED AS CENTIMETERS WITH SCALAR -100",
        10: "DATA SOURCE: research_seismic_sweep/(x_NNN)/data.npz",
    }
    return segyio.tools.create_text_header(lines)



def build_gather(component: str, spread: str, cfg: dict) -> None:
    """
    Собирает один gather-файл:
      компонента (Vx/Vy) x шаблон (right/left).
    """
    scfg = cfg[spread]
    npz_dir: Path = cfg["npz_dir"]
    output_dir: Path = cfg["output_dir"]
    sample_count: int = cfg["sample_count"]
    sample_interval_us: int = cfg["sample_interval_us"]

    source_xs = list(range(scfg["source_x_start"], scfg["source_x_stop"] + 1, scfg["source_x_step"]))
    n_src = len(source_xs)
    n_rec = scfg["receiver_count"]
    rec_step = scfg["receiver_step_m"]
    rec_dir = scfg["receiver_direction"]
    total_traces = n_src * n_rec

    output_path = output_dir / f"output_line_v2_{component}_{spread}.sgy"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = segyio.spec()
    spec.format = 1           # 4-byte IBM float
    spec.samples = np.arange(sample_count, dtype=np.float32)
    spec.tracecount = total_traces

    with segyio.create(str(output_path), spec) as out:
        out.text[0] = make_text_header(component, spread, cfg)

        out.bin.update({
            segyio.BinField.JobID:               1,
            segyio.BinField.LineNumber:          1,
            segyio.BinField.ReelNumber:          1,
            segyio.BinField.Traces:              total_traces,
            segyio.BinField.AuxTraces:           0,
            segyio.BinField.Interval:            sample_interval_us,
            segyio.BinField.IntervalOriginal:    sample_interval_us,
            segyio.BinField.Samples:             sample_count,
            segyio.BinField.SamplesOriginal:     sample_count,
            segyio.BinField.Format:              1,
            segyio.BinField.EnsembleFold:        n_rec,
            segyio.BinField.SortingCode:         5,   # 5 = common shot point
            segyio.BinField.MeasurementSystem:   1,   # 1 = metres
            segyio.BinField.SEGYRevision:        0,
            segyio.BinField.SEGYRevisionMinor:   0,
        })

        trace_idx = 0
        for shot_num_1b, source_x in enumerate(source_xs, start=1):
            data = load_npz(npz_dir, source_x)

            for rec_num_1b in range(0, n_rec):
                rec_x = source_x + rec_dir * rec_step * rec_num_1b
                cdp_x = 0.5 * (source_x + rec_x)
                offset_m = rec_x - source_x   # со знаком

                trace = extract_trace(data, component, rec_x, sample_count)

                out.trace[trace_idx] = trace
                out.header[trace_idx] = {
                    # Глобальная нумерация
                    segyio.TraceField.TRACE_SEQUENCE_LINE: trace_idx + 1,
                    segyio.TraceField.TRACE_SEQUENCE_FILE: trace_idx + 1,
                    segyio.TraceField.CROSSLINE_3D:        trace_idx + 1,
                    # Поля источника
                    segyio.TraceField.FieldRecord:         shot_num_1b,
                    segyio.TraceField.EnergySourcePoint:   shot_num_1b,
                    segyio.TraceField.CDP:                 shot_num_1b,
                    # Поля приёмника
                    segyio.TraceField.CDP_TRACE:           rec_num_1b,
                    segyio.TraceField.INLINE_3D:           rec_num_1b,
                    segyio.TraceField.TraceNumber:         rec_num_1b,
                    # Координаты x 100 (сантиметры) + скаляр -100
                    segyio.TraceField.SourceX: int(round(source_x * 100)),
                    segyio.TraceField.GroupX:  int(round(rec_x    * 100)),
                    segyio.TraceField.CDP_X:   int(round(cdp_x    * 100)),
                    # Вынос в метрах (signed)
                    segyio.TraceField.offset:  int(round(offset_m)),
                    # Константы
                    segyio.TraceField.SourceGroupScalar:       cfg["source_group_scalar"],
                    segyio.TraceField.ElevationScalar:         cfg["elevation_scalar"],
                    segyio.TraceField.ReceiverGroupElevation:  cfg["receiver_group_elevation"],
                    segyio.TraceField.CoordinateUnits:         cfg["coordinate_units"],
                    segyio.TraceField.DataUse:                 cfg["data_use"],
                    segyio.TraceField.TraceIdentificationCode: cfg["trace_id_code"],
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL:   sample_interval_us,
                    segyio.TraceField.TRACE_SAMPLE_COUNT:      sample_count,
                }
                trace_idx += 1

            if shot_num_1b % 10 == 0 or shot_num_1b == n_src:
                print(f"  [{spread}/{component}] shot {shot_num_1b}/{n_src} (x={source_x}m), "
                      f"трасс записано: {trace_idx}")

    print(f"Готово: {output_path}  ({total_traces} трасс)\n")



for component in CONFIG["components"]:
    for spread in ("right", "left"):
        print(f"=== {component} / {spread} ===")
        build_gather(component, spread, CONFIG)
print("Все файлы записаны.")

