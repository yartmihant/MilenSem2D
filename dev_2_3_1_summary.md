# Глава III, задача 1: Заголовки примера SEG-Y (Tesseral)

## Цель

Изучить заголовки и структуру файла-образца сейсмограммы Tesseral для последующего приведения синтетических сейсмограмм к тому же формату.

## Входные файлы

- `data/dev_2_3/example/21-01-26+GathAP-X.sgy` — образец SEG-Y из Tesseral.

## Выполненная работа

1. Разбор **текстового заголовка** (3200 байт): источник (Tesseral), время 0–4000 мс, шаг 4 мс, формат данных.
2. Разбор **бинарного заголовка**: число трасс (12635), сэмплов на трассу (1001), интервал 4000 µs (4 мс), формат данных IBM float (код 1), SortingCode 5 (CMP), EnsembleFold 32.
3. Разбор **трассовых заголовков**: перечислены постоянные поля (CoordinateUnits, DataUse, скаляры, TRACE_SAMPLE_COUNT/INTERVAL, TraceIdentificationCode) и поля, меняющиеся по трассам (CDP, CDP_TRACE, CDP_X, CROSSLINE_3D, EnergySourcePoint, FieldRecord, GroupX, INLINE_3D, SourceX, TRACE_SEQUENCE_*, TraceNumber).

## Выходные файлы

- `data/dev_2_3_1_example_headers.md` — текстовое описание формата примера (текстовый, бинарный и трассовые заголовки).

## Скрипт

- `dev_2_3_1_example_headers.py` — скрипт анализа и генерации отчёта.
