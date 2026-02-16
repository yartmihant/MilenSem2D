# Diff заголовков SEG-Y: эталон Tesseral vs синтетика MILEN

Сравнение:
- Эталон: `data/dev_2_3/example/21-01-26+GathAP-X.sgy`
- Ваш файл (пример): `data/dev_2_3/sgy/output_line1_Vx_x10000p00_d8.sgy`

## 1) Текстовый заголовок

- Эталон: Tesseral, time-format, окно `0..4000 ms`, шаг `4 ms`, IBM float.
- Ваш: служебные строки CAE Fidesys, другое описание эксперимента.

Вывод: текстовый заголовок не в формате Tesseral (для совместимости лучше заменить/шаблонизировать под эталон).

## 2) Бинарный заголовок (ключевые расхождения)

- `Traces`: `12635` vs `80`
- `Samples`: `1001` vs `1501`
- `Interval`: `4000` us vs `2000` us
- `SortingCode`: `5` vs `0`
- `EnsembleFold`: `32` vs `0`
- `MeasurementSystem`: `1` vs `0`
- `LineNumber`: `1` vs `0`
- `ReelNumber`: `1` vs `0`
- `Format`: `1` vs `1` (совпадает)

## 3) Трассовые заголовки (first/last trace)

- `TRACE_SEQUENCE_FILE`: `1..12635` vs `0..0`
- `TRACE_SEQUENCE_LINE`: `1..12635` vs `0..0`
- `FieldRecord`: `1..156` vs `0..0`
- `TraceNumber`: `1..80` vs `0..0`
- `EnergySourcePoint`: `1..156` vs `0..0`
- `CDP`: `1..156` vs `0..0`
- `CDP_TRACE`: `1..80` vs `0..0`
- `INLINE_3D`: `1..80` vs `955383..955383`
- `CROSSLINE_3D`: `1..12635` vs `1..1`
- `SourceGroupScalar`: `-100` vs `0`
- `ElevationScalar`: `-100` vs `0`
- `ReceiverGroupElevation`: `-8` vs `0`
- `TraceIdentificationCode`: `1` vs `0`
- `TRACE_SAMPLE_INTERVAL`: `4000` vs `2000`
- `TRACE_SAMPLE_COUNT`: `1001` vs `1501`

## 4) Поля, о которых вы уточнили (offset/cdp/cdp_x/sou_x/rec_x)

Нотация:
- `sou_x` -> `SourceX`
- `rec_x` -> `GroupX` (координата приемника/группы)

Статистика по файлу:

- Эталон:
  - `offset`: min `0`, max `4000`, unique `81`
  - `CDP`: min `1`, max `156`, unique `156`
  - `CDP_X`: min `0`, max `11700`, unique `235`
  - `SourceX`: min `0`, max `775000`, unique `156`
  - `GroupX`: min `8`, max `1170000`, unique `235`

- Ваш файл:
  - `offset`: min `0`, max `0`, unique `1`
  - `CDP`: min `0`, max `0`, unique `1`
  - `CDP_X`: min `0`, max `0`, unique `1`
  - `SourceX`: min `0`, max `0`, unique `1`
  - `GroupX`: min `6000`, max `9950`, unique `80`

Вывод: в эталоне эти поля заполнены и информативны; в вашем файле требуется заполнить `offset`, `CDP`, `CDP_X`, `SourceX` и связанные индексные поля.

## 5) Семантика ключевых заголовков

- `offset`: расстояние между источником и приемником вдоль профиля (обычно `GroupX - SourceX` с учетом скалярей и знака/системы координат проекта).
- `CDP`: номер общей средней точки (Common Depth Point/CMP gather index).
- `CDP_X`: X-координата средней точки пары источник-приемник.
- `SourceX`: X-координата источника.
- `GroupX` (`rec_x`): X-координата приемника (group).

---
Этот diff фиксирует, что нужно заполнять в первую очередь, чтобы синтетика выглядела как Tesseral по заголовкам.
