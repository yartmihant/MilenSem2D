# Описание формата SEG-Y примера Tesseral

Файл: `data/dev_2_3/example/21-01-26+GathAP-X.sgy`

## Текстовый заголовок (File Header)

```
C 1 __ Tesseral Technologies Inc. __
C 2 __ SEGY in Time Format __
C 3                 __ Start Time:      0.00msec
C 4                  __ Stop Time:   4000.00msec
C 5              __ Sampling Step:      4.00msec
C 6 __ Format:                    16 bit IBM floating point
C 7
C 8
C 9
C10
C11
C12
C13
C14
C15
C16
C17
C18
C19
C20
C21
C22
C23
C24
C25
C26
C27
C28
C29
C30
C31
C32
C33
C34
C35
C36
C37
C38
C39
C40
```

## Бинарный заголовок

- **Трасс в файле**: 12635
- **Сэмплов на трассу**: 1001
- **Формат данных (код)**: 1 = 4-byte IBM float (в текстовом заголовке Tesseral указано «16 bit IBM» — имеется в виду разрядность мантиссы; в файле используется стандартный SEG-Y Format 1)
- **Interval**: 4000 (интервал дискретизации в микросекундах = 4 мс)
- **SortingCode**: 5 (Common Midpoint)
- **EnsembleFold**: 32

- **AmplitudeRecovery**: 0
- **AuxTraces**: 0
- **BinaryGainRecovery**: 0
- **CorrelatedTraces**: 0
- **EnsembleFold**: 32
- **ExtAuxTraces**: 0
- **ExtEnsembleFold**: 0
- **ExtSamples**: 0
- **ExtSamplesOriginal**: 0
- **ExtTraces**: 0
- **ExtendedHeaders**: 0
- **Format**: 1
- **ImpulseSignalPolarity**: 0
- **Interval**: 4000
- **IntervalOriginal**: 4000
- **JobID**: 0
- **LineNumber**: 1
- **MeasurementSystem**: 1
- **ReelNumber**: 1
- **SEGYRevision**: 0
- **SEGYRevisionMinor**: 0
- **Samples**: 1001
- **SamplesOriginal**: 1001
- **SortingCode**: 5
- **Sweep**: 0
- **SweepChannel**: 0
- **SweepFrequencyEnd**: 0
- **SweepFrequencyStart**: 0
- **SweepLength**: 0
- **SweepTaperEnd**: 0
- **SweepTaperStart**: 0
- **Taper**: 0
- **TraceFlag**: 0
- **Traces**: 12635
- **Unassigned1**: 0
- **VerticalSum**: 0
- **VibratoryPolarity**: 0

## Трассовый заголовок (Trace Header)

### Постоянные поля (одинаковые для всех трасс)

- **CoordinateUnits**: 1
- **DataUse**: 1
- **ElevationScalar**: -100
- **ReceiverGroupElevation**: -8
- **SourceGroupScalar**: -100
- **TRACE_SAMPLE_COUNT**: 1001
- **TRACE_SAMPLE_INTERVAL**: 4000
- **TraceIdentificationCode**: 1

### Поля, меняющиеся по трассам

- **CDP**: от 1 до 156
- **CDP_TRACE**: от 1 до 80
- **CDP_X**: от 0 до 11700
- **CROSSLINE_3D**: от 1 до 12635
- **EnergySourcePoint**: от 1 до 156
- **FieldRecord**: от 1 до 156
- **GroupX**: от 8 до 1170000
- **INLINE_3D**: от 1 до 80
- **SourceX**: от 0 до 775000
- **TRACE_SEQUENCE_FILE**: от 1 до 12635
- **TRACE_SEQUENCE_LINE**: от 1 до 12635
- **TraceNumber**: от 1 до 80

---
Сгенерировано скриптом dev_2_3_1_example_headers.py