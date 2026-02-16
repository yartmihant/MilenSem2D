#!/usr/bin/env python3
"""Сборка shot-файлов SEG-Y в один общий gather-файл с заполнением заголовков."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import segyio


SHOT_RE = re.compile(r"_x(\d+)p00_d8\.sgy$")


def parse_shot_x(path: Path) -> int:
    m = SHOT_RE.search(path.name)
    if not m:
        raise ValueError(f"Не удалось извлечь X источника из имени: {path.name}")
    return int(m.group(1))


def receiver_xs_for_shot(source_x_m: int, shot_index_1b: int, receiver_count: int) -> list[int]:
    """Возвращает X приемников в метрах для одного источника.

    Правило пользователя:
    - источники 1..113: справа от источника, шаг 50 м (source+50 ... source+4000)
    - источники 114..226: слева от источника, 80 приемников, затем источник.
      По порядку трасс используем возрастающий X (farthest-left -> nearest-left),
      как в исходных файлах Fidesys.
    """
    step_m = 50

    if shot_index_1b <= 113:
        return [source_x_m + step_m * i for i in range(1, receiver_count + 1)]

    # Слева от источника: source-4000 ... source-50 (возрастающий X)
    return [source_x_m - step_m * i for i in range(receiver_count, 0, -1)]


def build_text_header(component: str) -> str:
    lines = {
        1: "MILEN SEM 2D - SYNTHETIC GATHER",
        2: "SEGY IN TIME FORMAT (MERGED SHOTS)",
        3: "START TIME:      0.00 MSEC",
        4: "STOP TIME:    3000.00 MSEC",
        5: "SAMPLING STEP:   2.00 MSEC",
        6: "FORMAT: 4-BYTE IBM FLOAT (SEG-Y FORMAT 1)",
        7: f"COMPONENT: {component}",
        8: "GEOMETRY: 226 SOURCES X 80 RECEIVERS = 18080 TRACES",
        9: "SRC 1..113: RECEIVERS RIGHT, SRC 114..226: RECEIVERS LEFT",
        10: "COORD UNITS: METERS; STORED AS CENTIMETERS WITH SCALAR -100",
    }
    return segyio.tools.create_text_header(lines)


def merge_component(
    input_dir: Path,
    output_path: Path,
    component: str,
    expected_shots: int,
    expected_receivers: int,
) -> None:
    pattern = f"output_line1_{component}_x*p00_d8.sgy"
    shot_files = sorted(input_dir.glob(pattern), key=parse_shot_x)

    if len(shot_files) != expected_shots:
        raise RuntimeError(
            f"Для {component} найдено {len(shot_files)} shot-файлов, ожидалось {expected_shots}"
        )

    first_file = shot_files[0]
    with segyio.open(str(first_file), "r", ignore_geometry=True) as f0:
        sample_count = len(f0.trace[0])
        sample_interval_us = f0.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        input_format = f0.format

    if sample_count != 1501:
        raise RuntimeError(f"Ожидалось 1501 сэмпл, получено {sample_count}")
    if sample_interval_us != 2000:
        raise RuntimeError(f"Ожидался интервал 2000 us, получено {sample_interval_us}")

    total_traces = expected_shots * expected_receivers

    spec = segyio.spec()
    spec.format = int(input_format)  # 1 = IBM float
    spec.samples = range(sample_count)
    spec.tracecount = total_traces

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with segyio.create(str(output_path), spec) as out:
        out.text[0] = build_text_header(component)

        out.bin.update(
            {
                segyio.BinField.JobID: 1,
                segyio.BinField.LineNumber: 1,
                segyio.BinField.ReelNumber: 1,
                segyio.BinField.Traces: total_traces,
                segyio.BinField.AuxTraces: 0,
                segyio.BinField.Interval: sample_interval_us,
                segyio.BinField.IntervalOriginal: sample_interval_us,
                segyio.BinField.Samples: sample_count,
                segyio.BinField.SamplesOriginal: sample_count,
                segyio.BinField.Format: int(input_format),
                segyio.BinField.EnsembleFold: 32,
                segyio.BinField.SortingCode: 5,
                segyio.BinField.MeasurementSystem: 1,
                segyio.BinField.SEGYRevision: 0,
                segyio.BinField.SEGYRevisionMinor: 0,
            }
        )

        trace_idx = 0
        for shot_idx_1b, shot_path in enumerate(shot_files, start=1):
            source_x_m = parse_shot_x(shot_path)
            receiver_xs_m = receiver_xs_for_shot(source_x_m, shot_idx_1b, expected_receivers)

            with segyio.open(str(shot_path), "r", ignore_geometry=True) as shot:
                if shot.tracecount != expected_receivers:
                    raise RuntimeError(
                        f"{shot_path.name}: трасс {shot.tracecount}, ожидалось {expected_receivers}"
                    )

                for rec_idx_1b in range(1, expected_receivers + 1):
                    rec_x_m = receiver_xs_m[rec_idx_1b - 1]
                    cdp_x_m = 0.5 * (source_x_m + rec_x_m)
                    offset_m_signed = rec_x_m - source_x_m

                    out.trace[trace_idx] = np.asarray(shot.trace[rec_idx_1b - 1], dtype=np.float32)

                    out.header[trace_idx] = {
                        # Глобальная нумерация трасс
                        segyio.TraceField.TRACE_SEQUENCE_FILE: trace_idx + 1,
                        segyio.TraceField.TRACE_SEQUENCE_LINE: trace_idx + 1,
                        segyio.TraceField.CROSSLINE_3D: trace_idx + 1,
                        # Поля источника
                        segyio.TraceField.FieldRecord: shot_idx_1b,
                        segyio.TraceField.EnergySourcePoint: shot_idx_1b,
                        segyio.TraceField.CDP: shot_idx_1b,
                        # Поля приемника
                        segyio.TraceField.CDP_TRACE: rec_idx_1b,
                        segyio.TraceField.INLINE_3D: rec_idx_1b,
                        segyio.TraceField.TraceNumber: rec_idx_1b,
                        # Координаты (см) + скаляр -100 => метры
                        segyio.TraceField.SourceX: int(round(source_x_m * 100)),
                        segyio.TraceField.GroupX: int(round(rec_x_m * 100)),
                        segyio.TraceField.CDP_X: int(round(cdp_x_m * 100)),
                        # Смещение источник-приемник (метры, signed)
                        segyio.TraceField.offset: int(round(offset_m_signed)),
                        # Трассовые константы
                        segyio.TraceField.SourceGroupScalar: -100,
                        segyio.TraceField.ElevationScalar: -100,
                        segyio.TraceField.ReceiverGroupElevation: -8,
                        segyio.TraceField.CoordinateUnits: 1,
                        segyio.TraceField.DataUse: 1,
                        segyio.TraceField.TraceIdentificationCode: 1,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval_us,
                        segyio.TraceField.TRACE_SAMPLE_COUNT: sample_count,
                    }
                    trace_idx += 1

    print(f"Готово: {output_path} | component={component} | traces={total_traces}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MILEN shot SEG-Y files into a single SEG-Y")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dev_2_3/sgy"),
        help="Папка с shot-файлами SEG-Y",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dev_2_3/merged"),
        help="Папка для объединенных SEG-Y",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["Vx", "Vy"],
        choices=["Vx", "Vy"],
        help="Компоненты для сборки",
    )
    parser.add_argument("--shots", type=int, default=226, help="Ожидаемое число источников")
    parser.add_argument("--receivers", type=int, default=80, help="Ожидаемое число приемников на источник")

    args = parser.parse_args()

    for component in args.components:
        output_path = args.output_dir / f"output_line1_{component}_gather_merged.sgy"
        merge_component(
            input_dir=args.input_dir,
            output_path=output_path,
            component=component,
            expected_shots=args.shots,
            expected_receivers=args.receivers,
        )


if __name__ == "__main__":
    main()
