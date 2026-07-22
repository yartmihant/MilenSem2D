# Fidesys table/discovery tests for fc_calc v2

Этот каталог содержит калибровочные `.fc`-файлы для завершения `fc_calc v2`.
Это не физические regression tests в обычном смысле, а discovery-набор:
он нужен, чтобы восстановить фактическое кодирование Fidesys `.fc` для
материалов, нагрузок, ограничений, начальных условий, constraints и receivers.

Основной план: `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_07.md`.
Суб-планы:

- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_06.md`;
- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_04.md`;
- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_02.md`;
- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_03.md`;
- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_05.md`.

`F01_material_dependency_forms.fc` был создан пользователем вручную в
Fidesys. Основные файлы F02-F10 и диагностический split-набор F02 создаются
`fc_calc/gen_table_tests.py`, кроме пользовательского
`F02_split/F02_09_point_dead_force_node_fix.fc`. Текущий
`F11_receivers.fc` пересоздан пользователем в Fidesys и может отличаться от
варианта, который генерирует `gen_table_tests.py`.

## Общие соглашения

- Простые 3D-файлы используют куб `HEX8`, обычно 2x2x2 элемента, с
  линейно-упругим материалом `calibration_linear_elastic`.
- Простые 2D-файлы используют квадрат `QUAD4`.
- В generated files force-нагрузки записываются шестью компонентами FC format:
  первые три компоненты являются поступательной силой/traction, последние три
  оставлены нулями как rotational/moment-like slots.
- Для spatial interpolation в generated settings задан
  `settings.table_interpolation = near/gauss/4/1.0/1.0`. Это поле
  экспериментальное; UI Fidesys пока его не задаёт.
- `PointDeadForce`, `PointTrackingForce`, `PointHydrodynamicForce`, sloshing и
  constraints не считаются подтверждёнными поддерживаемыми case-ами. Если они
  встречаются, текущая политика `fc_calc v2`: warning/report + ignore, пока
  нет физического SEMGPU-контракта или реального UI-образца.
- Пользователь проверял generated files импортом вида:
  `import fidesyscase '.../Fxx_....fc' mesh_geometry`.

## F01_material_dependency_forms.fc

Назначение: самый важный материалный образец. Он создан пользователем вручную
в Fidesys, потому что именно UI/экспорт Fidesys является источником истины для
реального кодирования material property tables.

Фактическое содержимое текущего файла:

- геометрия: один `HEX8`-элемент, 8 узлов;
- material `mat1`;
- `elasticity/HOOK/YOUNG_MODULE`: `TABLE` по `TABULAR_NODE_ID`, значения
  `2001..2008` на узлах `1..8`;
- `elasticity/HOOK/POISSON_RATIO`: `TABLE` по `TABULAR_ELEMENT_ID`, значение
  `0.25` на элементе `1`;
- `common/USUAL/DENSITY`: `CONSTANT`, значение `1000`;
- `common/USUAL/STRUCTURAL_DAMPING_RATIO`: `FORMULA`, выражение `x^2+y^2`;
- `common/USUAL/STIFFNESS_DAMPING_RATIO`: `TABLE` по `TABULAR_X`, точки
  `10 -> 0`, `-10 -> 1`;
- `plasticity/MISES/YIELD_STRENGTH`: `CONSTANT`, значение `1`, нужен как
  незнакомое/нецелевое material property для проверки ignore policy.

Контекст реакции пользователя: пользователь создал этот файл первым и отметил,
что здесь описана самая сложная часть - таблицы. Этот файл должен направлять
реализацию `FCData evaluator` для material properties. Он также показывает, что
часть желаемых форм из исходного плана может отсутствовать в текущем F01
например `MASS_DAMPING_RATIO` time table. Реализующий агент должен смотреть
фактический файл, а не только исходные пожелания плана.

Ожидаемые выводы для `fc_calc v2`:

- material mapping должен работать через общий `FCData evaluator`;
- `TABULAR_NODE_ID`, `TABULAR_ELEMENT_ID`, `TABULAR_X`, `FORMULA` и
  `CONSTANT` должны поддерживаться;
- неизвестные или нецелевые material properties не должны валить pipeline.

## F02_loads_static_supported_3d.fc

Назначение: основной 3D-набор механических static loads с constant data.

Геометрия: куб `HEX8`, 2x2x2 элемента, 27 узлов. В файле 10 loads:

1. `F02_face_dead_stress_left`, `FaceDeadStress`, давление `1000`;
2. `F02_face_tracking_stress_right`, `FaceTrackingStress`, давление `800`;
3. `F02_face_absorbing_front`, `FaceAbsorbingBC`, без data-компонент;
4. `F02_face_distributed_force_back`, `FaceDistributedForce`,
   `(10, 20, 30)`;
5. `F02_face_equivalent_force_front`, `FaceEquivalentForce`,
   `(11, 21, 31)`;
6. `F02_face_tracking_distributed_force_top`,
   `FaceTrackingDistributedForce`, `(12, 22, 32)`;
7. `F02_face_tracking_equivalent_force_left`,
   `FaceTrackingEquivalentForce`, `(13, 23, 33)`;
8. `F02_node_force_top_corners`, `NodeForce`, `(1, 2, 3)`, на два верхних
   угловых узла в generated file;
9. `F02_gravity_mass_force_all_nodes`, `GravityMassForce`,
   `(0, 0, -9.81)`, на все узлы;
10. `F02_volume_gravity_mass_force_all_elements`, `VolumeGravityMassForce`,
    `(0, 0, -9.81)`, на все элементы.

Что пользователь увидел в Fidesys:

- пункты 1 и 2 отображаются как давления `1000` и `800`, приложенные к
  противоположным граням куба; визуально они не различаются по типу, различие
  остаётся только в `load.type`;
- пункт 3 отображается как неотражающее граничное условие `ABC` на грани;
- пункты 4-7 отображаются как распределённая сила, результирующая сила,
  следящая распределённая сила и следящая результирующая сила с указанными
  компонентами;
- пункты 8 и 9 в пользовательском описании выглядят как точечные силы:
  `(1,2,3)` на одну вершину и `(0,0,-9.81)` на все узлы грани кроме одной
  вершины; после дальнейшей проверки было уточнено, что UI Fidesys для
  "точечной силы" экспортирует `NodeForce`, а не `PointDeadForce`;
- пункт 10 отображается как гравитация `(0,0,-9.81)`, приложенная ко всему
  телу.

Важные решения:

- `FaceDeadStress` и `FaceTrackingStress` можно сначала маппить как pressure,
  но `load.type` нужно сохранять для диагностики;
- `GravityMassForce` оказался не body gravity, а нагрузкой для точечной массы;
  пока в SEMGPU нет point mass contract, его не поддерживаем;
- обычная гравитация тела для `fc_calc v2` - это `VolumeGravityMassForce`;
- `PointDeadForce`/`PointTrackingForce` исключены из основного F02, чтобы файл
  импортировался в Fidesys.

## F02_split/

Назначение: диагностический набор "один load на файл" для F02. Он нужен, чтобы
локализовать ошибки импорта Fidesys по конкретному `load.type` или `apply_to`.

Файлы:

- `F02_00_baseline_no_loads.fc` - базовый куб без нагрузок;
- `F02_01_face_dead_stress_left.fc`;
- `F02_02_face_tracking_stress_right.fc`;
- `F02_03_face_absorbing_front.fc`;
- `F02_04_face_distributed_force_back.fc`;
- `F02_05_face_equivalent_force_front.fc`;
- `F02_06_face_tracking_distributed_force_top.fc`;
- `F02_07_face_tracking_equivalent_force_left.fc`;
- `F02_08_node_force_top_corners.fc`;
- `F02_09_point_dead_force_node.fc` - synthetic `PointDeadForce` probe;
- `F02_09_point_dead_force_node_fix.fc` - пользовательский файл, созданный в
  Fidesys; он содержит все способы задать точечную силу через UI и показал,
  что UI экспортирует их как `NodeForce`;
- `F02_10_point_tracking_force_node.fc` - synthetic `PointTrackingForce` probe;
- `F02_11_gravity_mass_force_all_nodes.fc`;
- `F02_12_volume_gravity_mass_force_all_elements.fc`.

История ошибок:

- сначала combined F02 падал на
  `Bad base64_encode('loads[8].apply_to')`;
- затем были ошибки `Wrong element type for local surface id`, связанные с
  local surface id `0`;
- split-набор помог отделить поддерживаемые face/node/volume cases от
  неподтверждённых `Point*Force`.

Итог: `PointDeadForce` и `PointTrackingForce` пока не поддерживаются. Реальная
"точечная сила" из UI Fidesys для текущего workflow - это `NodeForce`.

## F03_loads_static_supported_2d_segments.fc

Назначение: 2D-аналог F02 для segment/edge loads.

Геометрия: квадрат `QUAD4`, 2x2 элемента. В файле 7 segment loads:

1. `SegmentDeadStress`;
2. `SegmentTrackingStress`;
3. `SegmentDistributedForce`;
4. `SegmentEquivalentForce`;
5. `SegmentTrackingDistributedForce`;
6. `SegmentTrackingEquivalentForce`;
7. `SegmentAbsorbingBC`.

Что пользователь увидел в Fidesys:

- файл открывается как плоский квадрат;
- первые две нагрузки отображаются как давления;
- затем идут четыре разновидности распределённых/результирующих сил,
  включая следящие варианты;
- последняя нагрузка отображается как неотражающее условие.

Вывод для `fc_calc v2`: segment loads должны идти через тот же mapping, что
и face loads, но на 2D boundary edge domain. `SegmentAbsorbingBC` поддерживать
только если есть подтверждённый SEMGPU condition contract; иначе warning.

## F04_load_dependency_forms.fc

Назначение: покрыть основные формы задания данных для loads.

Геометрия: 3D cube `HEX8`, 2x2x2 элемента. В файле 7 loads:

1. `FaceDeadStress` с `TABULAR_TIME`: `0 -> 100 -> 0` на `0..0.02`;
2. `FaceDistributedForce`, компоненты заданы отдельными таблицами
   `TABULAR_X`, `TABULAR_Y`, `TABULAR_Z`;
3. `NodeForce` с `TABULAR_NODE_ID`, все узлы явно относятся к грани задания;
4. `VolumeGravityMassForce` с `TABULAR_ELEMENT_ID`, разные значения `z` для
   8 элементов;
5. `NodeForce` с coordinate point cloud из трёх точек;
6. `NodeForce` с формулой `10*x + 2*y + t`;
7. `FaceEquivalentForce` с `TABULAR_FREQUENCY`, unsupported case.

Что пользователь увидел в Fidesys:

- pressure time table отображается корректно как `0-100-0` от `0` до
  `0.02` секунды;
- distributed force показывает разные таблицы по `x`, `y`, `z`;
- node force с node table отображается как точечная сила по узлам грани;
- gravity с element table отображается как разная гравитация в разных
  8 элементах;
- coordinate point cloud и formula отображаются как точечные силы, потому что
  они теперь заданы через `NodeForce`, а не через неподтверждённые `Point*`;
- frequency dependency отображается как зависимость от собственной частоты и
  не должен поддерживаться в первом SEMGPU mapping.

Вывод для `FCData evaluator`: нужны `TABULAR_TIME` с линейной интерполяцией,
spatial tables, point cloud, formula и structured warning для frequency.

## F05_restraints_static_supported.fc

Назначение: статические restraints, включая direction/normal restrictions.

Геометрия: 3D cube. В файле 8 restraints:

1. `Displacement` по всем осям;
2. `Velocity`;
3. `Acceleration`;
4. `PorePressure`;
5. `DirectionDisplacement`;
6. `DirectionVelocity`;
7. `DirectionAcceleration`;
8. `VolumeAngularVelocity`.

Что пользователь увидел в Fidesys:

- displacement отображается как перемещение, закреплённое по всем осям;
- velocity и acceleration отображаются как скорость и ускорение;
- `PorePressure` классифицируется Fidesys как restraint;
- `DirectionDisplacement`, `DirectionVelocity`, `DirectionAcceleration`
  отображаются как ограничение по нормали;
- `VolumeAngularVelocity` отображается как угловая скорость.

Выводы:

- direction restraints маппим как normal constraints;
- `PorePressure` не является обычным nodal Dirichlet. Если в материале задан
  `geomechanic.BIOT_ALPHA`, оно должно переходить в SEMGPU
  `EXT_VOLUME_PRESSURE`; без `BIOT_ALPHA` - warning/ignore;
- `VolumeAngularVelocity` пока warning/ignore, пока нет SEMGPU contract.

## F06_restraint_dependency_forms.fc

Назначение: формы задания данных для restraints.

Геометрия: 3D cube. В файле 6 restraints:

1. `Displacement` с `TABULAR_TIME`;
2. `Velocity` с `TABULAR_NODE_ID`;
3. `DirectionDisplacement` с `TABULAR_X`;
4. `DirectionVelocity` с coordinate point cloud;
5. `PorePressure` с формулой `1000 + 10*x + t`;
6. `Acceleration` с `TABULAR_FREQUENCY`, unsupported case.

Что пользователь увидел в Fidesys:

- все первые пять случаев отображаются согласно назначению;
- `Acceleration` с frequency table отображается как частотная зависимость и
  должен оставаться unsupported/warning.

Выводы:

- restraints должны использовать тот же `FCData evaluator`, что и loads;
- time-dependent и spatially non-uniform Dirichlet data нужны для
  `Displacement`, `Velocity` и direction restraints;
- formula pore pressure должен проходить через Biot path только при наличии
  `BIOT_ALPHA`.

## F07_initial_sets_dependency_forms.fc

Назначение: initial sets и их spatial dependency forms.

Геометрия: 3D cube. В файле 9 initial sets:

1. initial `Displacement`, constant zero;
2. initial `Velocity`, constant `(0, 0, 0.01)`;
3. initial `Displacement`, `TABULAR_NODE_ID`;
4. initial `Velocity`, `TABULAR_ELEMENT_ID`;
5. initial `Displacement`, coordinate point cloud;
6. initial `Velocity`, formula `0.001*x + 0.002*y + 0.003*z`;
7. initial `AngularVelocity`, constant `(0, 0, 1)`;
8. initial `Temperature`, constant `293.15`;
9. initial `PorePressure`, constant `1000`.

Что пользователь увидел в Fidesys:

- initial displacement constant отображается как всё равно `0`;
- initial displacement by nodes отображается;
- initial displacement point cloud отображается;
- initial velocity constant отображается как `0 0 0.01`;
- initial velocity by element фактически выглядит как "не задана ничем";
- initial velocity formula отображается;
- angular velocity, temperature и pore pressure отображаются как initial sets.

Выводы:

- на первом проходе поддерживать initial `Displacement` и `Velocity`;
- initial velocity `TABULAR_ELEMENT_ID` не считать подтверждённым рабочим case;
- `AngularVelocity`, `Temperature`, `PorePressure` initial sets пока
  warning/ignore.

## F08_unsupported_loads_and_constraints.fc

Назначение: собрать неподдержанные loads и constraints, чтобы `fc_calc v2`
не падал на незнакомых FC-сущностях, а выдавал понятный warning/report.

Текущий generated файл содержит:

- thermal face loads: `FaceHeatFlux`, `FaceConvection`, `FaceRadiation`;
- shell heat/convection variants;
- fluid face/segment/node/volume loads;
- segment thermal loads;
- node thermal/fluid loads;
- volume heat/fluid sources;
- contact constraints: 4 типа;
- coupling constraints: 7 типов;
- periodic constraints: 6 типов.

История ошибок и решений:

- первоначально F08 падал на `Bad base64_encode('loads[10].apply_to')`;
- sloshing (`FaceSloshingBC`, `SegmentSloshingBC`) и
  `PointHydrodynamicForce` были исключены из основного F08 до появления
  реального UI/parser-образца;
- затем Fidesys сообщил
  `'coupling_constraints[0].dofs' field is empty`;
- были добавлены минимальные type-specific поля для coupling/periodic probes;
- пользователь решил, что полноценную импортопригодность constraints через UI
  сейчас добивать не нужно: любой `FCConstraint` - только warning + ignore.

Вывод для `fc_calc v2`: unsupported thermal/fluid/shell/hydrodynamic loads и
все constraints не должны валить pipeline. Их нужно перечислять в warning/report
с типом и id/name.

## F09_finite_deformations_2d.fc

Назначение: проверить `settings.finite_deformations=true` для 2D и выбор
`fid_elast_2d_b_newmark_kim_nb`.

Геометрия: 2D square `QUAD4`, finite deformations включены.

Содержимое:

- `NodeForce` на верхние узлы, малая нагрузка;
- `Displacement` на left nodes;
- `DirectionDisplacement` на right edge.

Что пользователь увидел в Fidesys:

- пластина;
- `Displacement`;
- ограничение по нормали.

Вывод: `prepare.py` больше не должен запрещать `finite_deformations=true`;
`solver_select.py` должен выбирать fid 2D solver. Direction displacement должен
использовать тот же normal-constraint mapping, что F05/F06.

## F10_finite_deformations_3d.fc

Назначение: проверить `settings.finite_deformations=true` для 3D и выбор
`fid_elast_3d_b_newmark_kim_nb`.

Геометрия: 3D cube `HEX8`, finite deformations включены.

Содержимое:

- `NodeForce` на верхние узлы;
- `Displacement` на left nodes;
- `DirectionDisplacement` на right face.

Что пользователь увидел в Fidesys:

- `NodeForce`;
- `Displacement`;
- ограничение по нормали;
- у `NodeForce` и `Displacement` в UI может совпадать локальный индекс `1`.

Вывод: идентифицировать сущности надо по `load.type` / `restraint.flag`, а не
по UI index. Solver selection должен выбрать fid 3D solver.

## F11_receivers.fc

Назначение: покрыть `fc_model.receivers` и подготовить отдельный этап
receivers/SEGY.

Геометрия: 3D cube. Текущий файл пересоздан пользователем в Fidesys и содержит
нагрузку, restraint и receiver-набор без pressure/stress receivers.

Что пользователь увидел в Fidesys:

- есть 3 приёмника;
- приёмник на грань: все перемещения по всем осям;
- приёмник на ребро: скорость по оси X;
- приёмник на вершину: ускорение по осям X и Y;
- pressure и stress receivers намеренно исключены из текущего F11, потому что
  их решили пока не поддерживать.

Что сейчас видит `FCModel.load` для пересозданного файла:

- 3 receiver-записи;
- типы парсятся как `DISPLACEMENT`, `VELOCITY`, `ACCELERATION`;
- `dofs`: `[1,1,1]`, `[1,0,0]`, `[1,1,0]`;
- `output_step` отсутствует (`None`);
- `apply` содержит node-id списки на 9, 3 и 1 узел соответственно.

`dofs` трактуются как флаги компонент, а node-id `apply` переносится в
sampling-оператор на `calc_mesh`.

Решение по SEGY:

- формат восстановлен по пользовательским Fidesys-эталонам
  `F11_receivers_*.sgy`;
- поддерживать кинематические receivers (`DISPLACEMENT`, `VELOCITY`,
  `ACCELERATION`) с валидной привязкой;
- писать по одному SEG-Y файлу на компоненту;
- pressure/stress receivers пока warning/ignore;
- receiver записи без понятной привязки давать как warning/report.

## Текущая роль набора

Этот каталог является living discovery fixture. Если generated files меняются,
нужно обновить:

- `fc_calc/gen_table_tests.py`;
- этот `README.md`;
- `work/old/0_9_2026-05-23_T040_SOLVE_FC_V2_plan_07.md` или соответствующий
  суб-план;
- `CHANGELOG.md`.

Тяжёлые `fc_calc v2` прогоны и визуальную acceptance-проверку в Fidesys/ParaView
выполняет пользователь.
