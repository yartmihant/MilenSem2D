""" # Экспорт отдельных сейсмограмм источников из NPZ в SEG-Y # """

"""
Для каждого каталога `x_<координата>` прочитать `data.npz` и записать рядом две
сейсмограммы: `seismogram_vx.sgy` и `seismogram_vy.sgy`.

Структура бинарных и трассовых заголовков повторяет решения из
`scripts/2_3_merge_script.py`. В каждый файл входят все приёмники конкретного
источника. Записи короче 2001 отсчёта дополняются нулями справа.
"""

from pathlib import Path

import numpy as np
import segyio

""" ## Параметры экспорта ## """

CONFIG = {
    "npz_dir": Path("data/dev_2_3/research_seismic_sweep_7_1_0_full"),
    "components": ("vx", "vy"),
    "output_name": "seismogram_{component}.sgy",
    "sample_count": 2001,
    "sample_interval_us": 2000,
    "elevation_scalar": -100,
    "source_group_scalar": -100,
    "receiver_group_elevation": -8,
    "coordinate_units": 1,
    "trace_id_code": 1,
    "data_use": 1,
}


""" ## Поиск и проверка каталогов источников ## """

def source_coordinate(source_dir: Path) -> int:
    """Извлечь целочисленную координату источника из имени каталога `x_<X>`."""
    try:
        return int(source_dir.name.removeprefix("x_"))
    except ValueError as error:
        raise ValueError(f"Некорректное имя каталога источника: {source_dir}") from error


def find_source_dirs(npz_dir: Path) -> list[Path]:
    """Найти каталоги источников с `data.npz` и отсортировать их по координате."""
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"Каталог расчёта не найден: {npz_dir}")

    source_dirs = sorted(
        (path for path in npz_dir.glob("x_*") if path.is_dir()),
        key=source_coordinate,
    )
    if not source_dirs:
        raise FileNotFoundError(f"Каталоги источников не найдены: {npz_dir}/x_*")

    missing_npz = [path / "data.npz" for path in source_dirs if not (path / "data.npz").is_file()]
    if missing_npz:
        missing_text = "\n".join(str(path) for path in missing_npz)
        raise FileNotFoundError(f"Для источников отсутствуют NPZ-файлы:\n{missing_text}")
    return source_dirs


""" ## Подготовка данных сейсмограммы ## """

def validate_npz_arrays(
    data: np.lib.npyio.NpzFile,
    source_dir: Path,
    source_x: int,
    sample_interval_us: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Проверить координаты, размеры компонент и временной шаг одного NPZ-файла."""
    required_keys = {"x_center", "sensor_x", "seismo_times", "seismo_vx", "seismo_vy"}
    missing_keys = sorted(required_keys.difference(data.files))
    if missing_keys:
        raise KeyError(f"{source_dir / 'data.npz'}: отсутствуют поля {missing_keys}")

    sensor_x = np.asarray(data["sensor_x"], dtype=np.float64)
    seismo_times = np.asarray(data["seismo_times"], dtype=np.float64)
    input_sample_count = len(seismo_times)
    expected_shape = (input_sample_count, len(sensor_x))

    if sensor_x.ndim != 1 or seismo_times.ndim != 1:
        raise ValueError(f"{source_dir / 'data.npz'}: sensor_x и seismo_times должны быть одномерными")
    if len(sensor_x) == 0 or input_sample_count == 0:
        raise ValueError(f"{source_dir / 'data.npz'}: обнаружена пустая координатная или временная ось")
    if not np.all(np.diff(sensor_x) > 0):
        raise ValueError(f"{source_dir / 'data.npz'}: координаты приёмников не возрастают строго")

    stored_source_x = float(data["x_center"])
    if not np.isclose(stored_source_x, source_x, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"{source_dir / 'data.npz'}: x_center={stored_source_x} не совпадает с каталогом x_{source_x}"
        )

    for component in CONFIG["components"]:
        actual_shape = data[f"seismo_{component}"].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"{source_dir / 'data.npz'}: seismo_{component}{actual_shape}, "
                f"ожидается {expected_shape}"
            )

    if input_sample_count > 1:
        actual_interval_us = float(np.median(np.diff(seismo_times)) * 1_000_000)
        if not np.isclose(actual_interval_us, sample_interval_us, atol=1.0, rtol=0.0):
            raise ValueError(
                f"{source_dir / 'data.npz'}: временной шаг {actual_interval_us:.6f} мкс "
                f"не согласуется с SEG-Y шагом {sample_interval_us} мкс"
            )

    return sensor_x, seismo_times, input_sample_count


def prepare_traces(
    raw_seismogram: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Преобразовать сейсмограмму в float32 и привести её к заданной длине."""
    input_sample_count, receiver_count = raw_seismogram.shape
    traces = np.zeros((receiver_count, sample_count), dtype=np.float32)
    copied_sample_count = min(input_sample_count, sample_count)
    traces[:, :copied_sample_count] = raw_seismogram[:copied_sample_count, :].T
    return traces


""" ## Формирование заголовков SEG-Y ## """

def make_text_header(
    component: str,
    source_number: int,
    source_x: int,
    sensor_x: np.ndarray,
    input_sample_count: int,
    cfg: dict,
) -> bytes:
    """Создать текстовый заголовок индивидуальной сейсмограммы."""
    lines = {
        1: "MILEN SEM 2D - SYNTHETIC SHOT GATHER",
        2: "SEGY IN TIME FORMAT (NPZ -> SEG-Y CONVERSION)",
        3: f"COMPONENT: {component.upper()}",
        4: f"SOURCE NUMBER: {source_number}; X: {source_x} M",
        5: f"RECEIVERS: {len(sensor_x)}; X: {sensor_x[0]:.0f}..{sensor_x[-1]:.0f} M",
        6: f"INPUT SAMPLES: {input_sample_count}; OUTPUT SAMPLES: {cfg['sample_count']}",
        7: f"TIME: {cfg['sample_count']} SAMPLES X {cfg['sample_interval_us']} US",
        8: "FORMAT: 4-BYTE IBM FLOAT (SEG-Y FORMAT 1)",
        9: "COORD UNITS: METERS; STORED AS CENTIMETERS WITH SCALAR -100",
        10: f"DATA SOURCE: x_{source_x}/data.npz",
    }
    return segyio.tools.create_text_header(lines)


def make_binary_header(trace_count: int, cfg: dict) -> dict:
    """Сформировать бинарный заголовок по образцу merge-скрипта."""
    return {
        segyio.BinField.JobID: 1,
        segyio.BinField.LineNumber: 1,
        segyio.BinField.ReelNumber: 1,
        segyio.BinField.Traces: trace_count,
        segyio.BinField.AuxTraces: 0,
        segyio.BinField.Interval: cfg["sample_interval_us"],
        segyio.BinField.IntervalOriginal: cfg["sample_interval_us"],
        segyio.BinField.Samples: cfg["sample_count"],
        segyio.BinField.SamplesOriginal: cfg["sample_count"],
        segyio.BinField.Format: 1,
        segyio.BinField.EnsembleFold: trace_count,
        segyio.BinField.SortingCode: 5,
        segyio.BinField.MeasurementSystem: 1,
        segyio.BinField.SEGYRevision: 0,
        segyio.BinField.SEGYRevisionMinor: 0,
    }


def make_trace_header(
    trace_number: int,
    source_number: int,
    source_x: int,
    receiver_x: float,
    cfg: dict,
) -> dict:
    """Сформировать заголовок одной трассы по образцу merge-скрипта."""
    cdp_x = 0.5 * (source_x + receiver_x)
    offset_m = receiver_x - source_x
    return {
        segyio.TraceField.TRACE_SEQUENCE_LINE: trace_number,
        segyio.TraceField.TRACE_SEQUENCE_FILE: trace_number,
        segyio.TraceField.CROSSLINE_3D: trace_number,
        segyio.TraceField.FieldRecord: source_number,
        segyio.TraceField.EnergySourcePoint: source_number,
        segyio.TraceField.CDP: source_number,
        segyio.TraceField.CDP_TRACE: trace_number,
        segyio.TraceField.INLINE_3D: trace_number,
        segyio.TraceField.TraceNumber: trace_number,
        segyio.TraceField.SourceX: int(round(source_x * 100)),
        segyio.TraceField.GroupX: int(round(receiver_x * 100)),
        segyio.TraceField.CDP_X: int(round(cdp_x * 100)),
        segyio.TraceField.offset: int(round(offset_m)),
        segyio.TraceField.SourceGroupScalar: cfg["source_group_scalar"],
        segyio.TraceField.ElevationScalar: cfg["elevation_scalar"],
        segyio.TraceField.ReceiverGroupElevation: cfg["receiver_group_elevation"],
        segyio.TraceField.CoordinateUnits: cfg["coordinate_units"],
        segyio.TraceField.DataUse: cfg["data_use"],
        segyio.TraceField.TraceIdentificationCode: cfg["trace_id_code"],
        segyio.TraceField.TRACE_SAMPLE_INTERVAL: cfg["sample_interval_us"],
        segyio.TraceField.TRACE_SAMPLE_COUNT: cfg["sample_count"],
    }


""" ## Запись и проверка файлов ## """

def validate_segy(
    output_path: Path,
    source_number: int,
    source_x: int,
    sensor_x: np.ndarray,
    cfg: dict,
) -> None:
    """Проверить размеры и ключевые поля записанного SEG-Y-файла."""
    with segyio.open(str(output_path), "r", ignore_geometry=True) as segy_file:
        if segy_file.tracecount != len(sensor_x):
            raise ValueError(f"{output_path}: записано {segy_file.tracecount} трасс вместо {len(sensor_x)}")
        if len(segy_file.samples) != cfg["sample_count"]:
            raise ValueError(
                f"{output_path}: записано {len(segy_file.samples)} отсчётов вместо {cfg['sample_count']}"
            )
        if segy_file.bin[segyio.BinField.Interval] != cfg["sample_interval_us"]:
            raise ValueError(f"{output_path}: неверный шаг дискретизации в бинарном заголовке")

        first_header = segy_file.header[0]
        last_header = segy_file.header[len(sensor_x) - 1]
        expected_source_x = int(round(source_x * 100))
        expected_last_group_x = int(round(sensor_x[-1] * 100))
        if first_header[segyio.TraceField.FieldRecord] != source_number:
            raise ValueError(f"{output_path}: неверный сквозной номер источника")
        if first_header[segyio.TraceField.SourceX] != expected_source_x:
            raise ValueError(f"{output_path}: неверная координата источника")
        if last_header[segyio.TraceField.GroupX] != expected_last_group_x:
            raise ValueError(f"{output_path}: неверная координата последнего приёмника")


def write_component_segy(
    source_dir: Path,
    source_number: int,
    source_x: int,
    component: str,
    sensor_x: np.ndarray,
    raw_seismogram: np.ndarray,
    input_sample_count: int,
    cfg: dict,
) -> Path:
    """Записать одну компоненту сейсмограммы в отдельный SEG-Y-файл."""
    trace_count = len(sensor_x)
    output_path = source_dir / cfg["output_name"].format(component=component)
    temporary_path = Path(f".dev_2_3_x_{source_x}_{component}.sgy.tmp")
    traces = prepare_traces(raw_seismogram, cfg["sample_count"])

    specification = segyio.spec()
    specification.format = 1
    specification.samples = np.arange(cfg["sample_count"], dtype=np.float32)
    specification.tracecount = trace_count

    try:
        with segyio.create(str(temporary_path), specification) as segy_file:
            segy_file.text[0] = make_text_header(
                component,
                source_number,
                source_x,
                sensor_x,
                input_sample_count,
                cfg,
            )
            segy_file.bin.update(make_binary_header(trace_count, cfg))
            segy_file.trace[:] = traces

            for trace_number, receiver_x in enumerate(sensor_x, start=1):
                segy_file.header[trace_number - 1] = make_trace_header(
                    trace_number,
                    source_number,
                    source_x,
                    float(receiver_x),
                    cfg,
                )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    validate_segy(output_path, source_number, source_x, sensor_x, cfg)
    return output_path


def export_source(source_dir: Path, source_number: int, cfg: dict) -> tuple[list[Path], int]:
    """Экспортировать обе компоненты одного источника и вернуть созданные пути."""
    source_x = source_coordinate(source_dir)
    output_paths = []

    with np.load(source_dir / "data.npz") as data:
        sensor_x, _, input_sample_count = validate_npz_arrays(
            data,
            source_dir,
            source_x,
            cfg["sample_interval_us"],
        )
        for component in cfg["components"]:
            output_paths.append(
                write_component_segy(
                    source_dir,
                    source_number,
                    source_x,
                    component,
                    sensor_x,
                    data[f"seismo_{component}"],
                    input_sample_count,
                    cfg,
                )
            )
    return output_paths, input_sample_count


""" ## Экспорт всех источников ## """

source_dirs = find_source_dirs(CONFIG["npz_dir"])
created_files = []
short_record_count = 0

for source_index, current_source_dir in enumerate(source_dirs, start=1):
    current_files, current_input_sample_count = export_source(
        current_source_dir,
        source_index,
        CONFIG,
    )
    created_files.extend(current_files)
    if current_input_sample_count < CONFIG["sample_count"]:
        short_record_count += 1
    print(
        f"[{source_index:03d}/{len(source_dirs)}] {current_source_dir.name}: "
        f"{', '.join(path.name for path in current_files)} "
        f"({current_input_sample_count} -> {CONFIG['sample_count']} отсчётов)"
    )

print(
    f"Готово: {len(created_files)} SEG-Y-файла для {len(source_dirs)} источников; "
    f"нулевое дополнение применено к {short_record_count} источникам."
)


""" ## Выводы ## """

"""
Для каждого расчётного источника сформированы отдельные SEG-Y-сейсмограммы
компонент Vx и Vy. Координаты источника, приёмников, средних точек и выносы
записаны в трассовые заголовки; временные параметры и формат IBM float — в
бинарный заголовок. Каждый созданный файл повторно открыт и проверен.
"""
