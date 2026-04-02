# Результаты перехода к реальным данным

## Текущий статус

На этом шаге реализованы и реально посчитаны первые части плана из [plan.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/plan.md):

1. `A0` — подготовка корпусов данных;
2. `A1` — первичный выбор основного наблюдаемого ряда для `IPP`;
3. `A2/A3` — перенос маршрутизатора на реальные окна и фильтр интерпретируемости.

Пока основным эмпирическим корпусом выступают месячные индексы промышленного производства. Линия выручки уже подготовлена в tidy-формате и для неё запущен первый самостоятельный блок `B1`.

Отдельно линия `B1` по групповой медианной выручке уже запущена и описана в [b1_results.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/b1_results.md). Отраслевой подкорпус выручки и результаты `B2` вынесены в [b2_results.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/b2_results.md). Первый bridge-case между `IPP` и выручкой описан в [c1_results.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/c1_results.md).

---

## A0. Подготовка корпусов

### Полученные файлы

- [ipp_long.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/processed/ipp_long.csv)
- [ipp_hierarchy.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/processed/ipp_hierarchy.csv)
- [revenue_wide.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/processed/revenue_wide.csv)
- [revenue_long.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/real_data/processed/revenue_long.csv)

### Сводка по корпусам

| Корпус | Наблюдения | Комментарий |
|---|---:|---|
| `IPP long` | 21,996 | 47 серий × 3 варианта × 156 месяцев |
| `IPP hierarchy` | 47 | Иерархия рядов для отраслевых сопоставлений |
| `Revenue wide` | 23,943 | Стабильные компании, 2003–2022 |
| `Revenue long` | 478,860 | 23,943 компании × 20 лет |

### Ключевые технические выводы

1. Промышленный файл успешно раскрыт как корпус из `47` рядов, каждый в трёх версиях: `adj_smoothed`, `adj_unsmoothed`, `raw`.
2. Горизонт `IPP` подтверждён: `2013-01` — `2025-12`.
3. По выручке подтверждена большая доля пропусков по ОКВЭД: около `22.2%`.
4. Это ещё раз поддерживает решение начинать эмпирический анализ именно с промышленного массива.

---

## A1. Выбор основного наблюдаемого ряда для `IPP`

### Постановка

На первом шаге сравнивались три наблюдения одного и того же отраслевого процесса:

- `adj_smoothed`
- `adj_unsmoothed`
- `raw`

Базовая конфигурация окна:

- `W = 24`
- `L = 3`

Результаты baseline-расчёта сохранены в:

- [variant_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_variant_qc_baseline/variant_summary.csv)
- [series_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_variant_qc_baseline/series_summary.csv)

### Интегральная сводка по вариантам ряда

| Вариант | Серий | mean Var(omega) | mean ACF1 | mean ACF2 | no-model share | mean cond | mean R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adj_smoothed` | 47 | 0.000579 | 0.986223 | 0.951738 | 0.000000 | 69.862329 | 0.948516 |
| `adj_unsmoothed` | 47 | 0.006601 | 0.887660 | 0.856962 | 0.000000 | 3.452471 | 0.405905 |
| `raw` | 47 | 0.023603 | 0.730063 | 0.618777 | 0.000000 | 3.389284 | 0.502995 |

### Промежуточная интерпретация

1. `adj_smoothed` даёт почти идеальную автокорреляцию и очень высокий `R2`, но платой за это становится сильное “размазывание” динамики и завышенная инерционность.
2. `raw` существенно повышает дисперсию темпа роста и разрушает автокорреляционную структуру, то есть вносит слишком много календарного и нерегулярного шума.
3. `adj_unsmoothed` занимает промежуточное положение и на этом этапе выглядит лучшим кандидатом на основной слой для структурного чтения.

### Рабочий вывод по `A1`

На текущем этапе в качестве **основного рабочего слоя** для дальнейшего анализа принимается `adj_unsmoothed`.

Смысл выбора:

- `adj_smoothed` использовать как вспомогательный слой для визуализации и very local probe;
- `adj_unsmoothed` использовать как основной слой для routing и структурного чтения;
- `raw` использовать как stress-test на устойчивость результата.

### Expanded sensitivity: `W = 24, 36, 48` и `L = 3, 6`

Результаты расширенного прогона сохранены в:

- [variant_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_variant_qc_expanded/variant_summary.csv)

Главные наблюдения по расширенной сетке:

1. При всех окнах и обоих лаговых пространствах `adj_smoothed` остаётся самым инерционным и самым “слишком хорошим” вариантом: у него максимальные `ACF1/ACF2`, высокий `R2`, но и резко возрастающая обусловленность при `L = 6`.
2. `raw` стабильно даёт наибольшую `Var(omega)` и сохраняет шумовой характер даже при длинных окнах.
3. `adj_unsmoothed` сохраняет статус лучшего компромисса и в expanded-сетке.

Особенно важно:

- для `adj_unsmoothed` рост окна с `24` до `48` уменьшает `mean R2` при `L = 3` (`0.406 -> 0.278`), то есть длинное окно не даёт “чуда fit`;
- при `L = 6` `adj_unsmoothed` становится несколько плотнее (`mean R2` выше), но всё ещё не уходит в искусственную инерционность `adj_smoothed`;
- у `adj_smoothed` при `L = 6` `mean_cond` становится экстремально большим: от `620.6` до `923.9`.

Это усиливает исходный вывод: **`adj_smoothed` не должен быть основным структурным слоем**.

---

## A2/A3. Перенос маршрутизатора и фильтр интерпретируемости

### Артефакты

- [window_features.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_all_variants/window_features.csv)
- [regime_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_all_variants/regime_summary.csv)
- [interpretability_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_all_variants/interpretability_summary.csv)
- [routing_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_all_variants/routing_summary.csv)

Использовалась та же базовая конфигурация:

- `W = 24`
- `L = 3`

Всего в baseline было обработано `18,753` окон по `47` сериям и `3` вариантам наблюдения.

### Распределение режимов по вариантам ряда

| Вариант | growth no memory | memory-like growth | plateau | turbulent informative |
|---|---:|---:|---:|---:|
| `adj_smoothed` | 0.068629 | 0.348744 | 0.578787 | 0.003839 |
| `adj_unsmoothed` | 0.002240 | 0.004799 | 0.172772 | 0.820189 |
| `raw` | 0.007359 | 0.155815 | 0.000960 | 0.835866 |

### Распределение классов интерпретируемости

| Вариант | interpretable | collinearity heavy | low dispersion | plateau |
|---|---:|---:|---:|---:|
| `adj_smoothed` | 0.353863 | 0.055831 | 0.011518 | 0.578787 |
| `adj_unsmoothed` | 0.649016 | 0.104303 | 0.073908 | 0.172772 |
| `raw` | 0.779235 | 0.115182 | 0.104623 | 0.000960 |

### Что здесь важно

1. `adj_smoothed` действительно слишком часто уходит в `plateau_degenerate`.
Это сильный аргумент против использования сглаженного ряда как основного слоя структурной интерпретации.

2. `raw` даёт наибольшую долю формально `interpretable` окон, но почти целиком маршрутизируется в `turbulent_informative`.
Это не преимущество, а указание на то, что шум делает динамику слишком турбулентной для спокойного регрессионного чтения.

3. `adj_unsmoothed` выглядит наиболее сбалансированным вариантом.
Он ещё сохраняет высокий уровень интерпретируемости (`~64.9%` окон), но не проваливается в тотальную plateau-деградацию, как `adj_smoothed`.

### Практическое routing-следствие

Для `adj_unsmoothed` распределение preferred tool получилось таким:

| Инструмент | Доля окон |
|---|---:|
| `phase_short_forecast` | 0.746761 |
| `do_not_read_regression` | 0.246681 |
| `structural_regression` | 0.004319 |
| `direct_probe + enter` | 0.002240 |

Это очень жёсткий, но содержательный результат: на реальных окнах короткая структурная регрессия допустима далеко не всегда. Уже сейчас видно, что реальная промышленная динамика чаще отправляет окно либо в осторожный фазовый контур, либо вообще в режим “не читать обычную регрессию”.

### Expanded sensitivity для `adj_unsmoothed`: `W = 24, 36, 48`, `L = 6`

Расширенный прогон сохранён в:

- [interpretability_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_adj_unsmoothed_expanded/interpretability_summary.csv)
- [regime_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_routing_adj_unsmoothed_expanded/regime_summary.csv)

Ключевой результат здесь очень сильный:

| Окно | interpretable | collinearity heavy | low dispersion | plateau |
|---|---:|---:|---:|---:|
| `24` | 0.643257 | 0.110062 | 0.073908 | 0.172772 |
| `36` | 0.686302 | 0.113417 | 0.086513 | 0.113768 |
| `48` | 0.724185 | 0.104626 | 0.083740 | 0.087449 |

То есть при увеличении окна:

1. доля `interpretable` окон растёт;
2. доля `plateau_degenerate` заметно падает;
3. `collinearity_heavy` остаётся почти на том же уровне.

Это уже почти прямое подтверждение твоей синтетической логики:

> увеличение окна действительно помогает против исчезающего сигнала, но не устраняет проблему содержательной коллинеарности.

По режимам картина тоже устойчива:

| Окно | turbulent informative | plateau |
|---|---:|---:|
| `24` | 0.820189 | 0.172772 |
| `36` | 0.881836 | 0.113768 |
| `48` | 0.911185 | 0.087449 |

Это значит, что длинное окно не делает реальные ряды “простыми”; оно лишь уменьшает долю деградировавших окон и переводит больше случаев в турбулентный, но ещё информативный режим.

---

## Первые отраслевые наблюдения для `adj_unsmoothed`

### Серии с наибольшей долей интерпретируемых окон

| Код | Серия | interpretable share | dominant regime |
|---|---|---:|---|
| `21` | Производство лекарственных средств | 0.842105 | turbulent informative |
| `39` | Предоставление услуг в области ликвидации загрязнений | 0.827068 | turbulent informative |
| `18` | Деятельность полиграфическая | 0.827068 | turbulent informative |
| `31` | Производство мебели | 0.819549 | turbulent informative |
| `16` | Обработка древесины | 0.819549 | turbulent informative |
| `06.2` | Добыча природного газа и газового конденсата | 0.804511 | turbulent informative |

### Серии с наиболее выраженной plateau-деградацией

| Код | Серия | plateau share | dominant regime |
|---|---|---:|---|
| `07` | Добыча металлических руд | 0.857143 | plateau degenerate |
| `36` | Забор, очистка и распределение воды | 0.857143 | plateau degenerate |
| ПРОМ | Промышленность - всего | 0.849624 | plateau degenerate |
| `B` | Добыча полезных ископаемых | 0.759398 | plateau degenerate |
| `06` | Добыча сырой нефти и природного газа | 0.751880 | plateau degenerate |
| `06.1` | Добыча сырой нефти | 0.751880 | plateau degenerate |

### Комментарий

Это хорошо согласуется с экономическим смыслом:

- многие высокоагрегированные ряды действительно выглядят “инерционными” и часто уходят в plateau-тип окна;
- более узкие и волатильные отрасли чаще оказываются информативными, но уже не в простом growth-режиме, а в турбулентном.

---

## Графики

### Карта режимов по вариантам

![Regime Shares](outputs/real_data/ipp_routing_all_variants/regime_shares.png)

### Карта интерпретируемости по вариантам

![Interpretability Shares](outputs/real_data/ipp_routing_all_variants/interpretability_shares.png)

### Scatter `Var(omega)` vs `cond(X_scaled)`

![Var Cond Scatter](outputs/real_data/ipp_routing_all_variants/var_cond_scatter.png)

---

## Текущий промежуточный вывод

На данный момент уже можно зафиксировать три результата.

1. Переход к реальным данным технически состоялся: оба корпуса приведены к рабочему long-format.
2. Для промышленного корпуса baseline-расчёты поддерживают выбор `adj_unsmoothed` как основного наблюдаемого ряда.
3. Реальные короткие окна действительно требуют того самого маршрута, который был выведен на синтетике:
   сначала режим, затем интерпретируемость, затем выбор контура, и только потом чтение структуры.

Самый важный ранний вывод здесь такой: **на живых рядах короткая регрессия не является default-операцией**. Значительная часть окон либо уходит в plateau/low-dispersion, либо требует осторожного фазового чтения вместо прямой интерпретации списка лагов.

---

## A4. Структурное чтение окон для `adj_unsmoothed`

### Артефакты

- [window_structures.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed/window_structures.csv)
- [series_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed/series_summary.csv)
- [mode_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed/mode_summary.csv)
- [mode_shares.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed/mode_shares.png)
- [top_interpretable_series.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed/top_interpretable_series.png)

На этом шаге структурное чтение строилось только для основного рабочего слоя `adj_unsmoothed` и только в baseline-конфигурации:

- `W = 24`
- `L = 3`

### Распределение режимов структурного чтения

| Режим чтения | Доля окон |
|---|---:|
| `phase_caution` | 0.644697 |
| `do_not_read` | 0.246681 |
| `beta_bsum_caution` | 0.102064 |
| `memory_structure` | 0.004319 |
| `current_state` | 0.002240 |

### Интерпретация

Это подтверждает предыдущий routing-результат в более содержательной форме.

1. Основная масса реальных окон не допускает прямого “чтения лагов как структуры памяти”.
2. Наиболее частый рабочий режим — `phase_caution`, то есть сначала фазовый и качественный анализ, а не буквальная регрессионная интерпретация.
3. Чисто memory-like окна на реальных `IPP`-рядах редки.
4. Окна, в которых допустимо осторожное чтение через `beta + B_sum`, есть, но это меньшинство.

### Поведение доминирующего фактора

В `adj_unsmoothed`-окнах лидером по `|beta|` чаще всего остаётся `X_n`:

| reading mode | top beta predictor | окон |
|---|---|---:|
| `phase_caution` | `X_n` | 3262 |
| `do_not_read` | `X_n` | 1190 |
| `beta_bsum_caution` | `X_n` | 374 |
| `phase_caution` | `Lag_1` | 321 |
| `phase_caution` | `Lag_2` | 265 |
| `phase_caution` | `Lag_3` | 182 |

Промежуточный смысл этого результата такой:

- даже в турбулентных окнах текущий уровень ряда чаще остаётся главным фактором локальной динамики;
- память в реальных `IPP`-рядах присутствует, но редко доминирует так же чисто, как в synthetic delay/mixed-кейсах.

### Поведение `B_sum`

Распределение суммарного лагового эффекта по baseline-окнам:

| Метрика | Значение |
|---|---:|
| mean | 0.000820 |
| std | 0.004188 |
| min | -0.028559 |
| 25% | -0.001537 |
| median | 0.000784 |
| 75% | 0.003182 |
| max | 0.039744 |

На этом этапе это указывает на важную вещь: в среднем суммарный лаговый эффект по реальным окнам невелик и концентрируется близко к нулю. Значит, на реальных промышленных рядах гораздо важнее отличать:

- сильный сигнал текущего состояния;
- слабую/вторичную память;
- и ложную структурную активность, возникающую на турбулентных окнах.

### Серии с наибольшей долей допускаемых интерпретируемых чтений

По итогам `series_summary.csv` наиболее “читаемыми” сериями оказались:

| Код | Серия | interpretable share | dominant mode | dominant top beta |
|---|---|---:|---|---|
| `21` | Производство лекарственных средств | 0.842105 | `phase_caution` | `X_n` |
| `18` | Деятельность полиграфическая | 0.827068 | `phase_caution` | `X_n` |
| `39` | Услуги по ликвидации загрязнений | 0.827068 | `phase_caution` | `X_n` |
| `16` | Обработка древесины | 0.819549 | `phase_caution` | `X_n` |
| `31` | Производство мебели | 0.819549 | `phase_caution` | `X_n` |
| `15` | Производство кожи | 0.812030 | `phase_caution` | `X_n` |

Это тоже характерно: даже у наиболее интерпретируемых серий доминирующий режим чтения остаётся осторожным, а не “чистой структурной регрессией”.

### Графики

#### Карта режимов структурного чтения

![Mode Shares](outputs/real_data/ipp_structural_adj_unsmoothed/mode_shares.png)

#### Серии с наибольшей долей интерпретируемых окон

![Top Interpretable Series](outputs/real_data/ipp_structural_adj_unsmoothed/top_interpretable_series.png)

### Вывод по `A4`

Промежуточный методологический вывод после `A4` такой:

> даже на выбранном основном рабочем слое `adj_unsmoothed` короткое окно чаще требует осторожного качественно-структурного чтения, чем прямой параметрической интерпретации.

Именно это и делает связку из `A2`, `A3` и `A4` полезной:

- `A2` отвечает за режим;
- `A3` — за допуск к интерпретации;
- `A4` — за форму допустимого чтения окна.

### Expanded sensitivity для `A4`: `W = 24, 36, 48`, `L = 6`

Расширенные артефакты:

- [mode_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed_expanded/mode_summary.csv)
- [series_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed_expanded/series_summary.csv)
- [mode_shares.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed_expanded/mode_shares.png)
- [top_interpretable_series.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_structural_adj_unsmoothed_expanded/top_interpretable_series.png)

Интегральная картина по expanded-прогону:

| reading mode | window share |
|---|---:|
| `phase_caution` | 0.678975 |
| `do_not_read` | 0.208546 |
| `beta_bsum_caution` | 0.108376 |
| `memory_structure` | 0.002462 |
| `current_state` | 0.001641 |

По сравнению с baseline это означает:

1. доля `phase_caution` ещё увеличивается;
2. доля `do_not_read` уменьшается;
3. чисто memory-like окна остаются редкими даже при более длинных окнах и `L = 6`.

Иначе говоря, увеличение окна помогает извлечь больше **допустимых** окон, но не переводит реальный промышленный ряд в простую memory-regression картину.

Expanded-top серии тоже немного сдвигаются. В expanded-сетке наиболее читаемыми оказываются:

| Код | Серия | interpretable share | dominant mode |
|---|---|---:|---|
| `13` | Производство текстильных изделий | 0.818182 | `phase_caution` |
| `16` | Обработка древесины | 0.818182 | `phase_caution` |
| `39` | Услуги по ликвидации загрязнений | 0.818182 | `phase_caution` |
| `30.2` | Производство железнодорожных средств | 0.815427 | `phase_caution` |
| `31` | Производство мебели | 0.812672 | `phase_caution` |

То есть состав “лучших” отраслей остаётся похожим, а dominant mode не меняется: это всё ещё осторожное фазово-структурное чтение, а не прямое чтение памяти.

---

## A5. Baseline consistency для `adj_unsmoothed`

### Артефакты

- [overall_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed/overall_summary.csv)
- [series_consistency.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed/series_consistency.csv)
- [variant_consistency.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed/variant_consistency.csv)
- [variant_consistency.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed/variant_consistency.png)

Пока consistency-слой рассчитан для baseline-конфигурации:

- `variant = adj_unsmoothed`
- `W = 24`
- `L = 3`

То есть это ещё не полный `A5` по всем окнам и вариантам, а первая опорная версия.

### Интегральная сводка

| Метрика | Среднее значение |
|---|---:|
| regime consistency | 0.907695 |
| interpretability consistency | 0.740522 |
| tool consistency | 0.834267 |
| top beta consistency | 0.772516 |
| top B consistency | 0.762918 |
| reading mode consistency | 0.737002 |

### Что это означает

1. На уровне baseline-анализа режим окна оказывается довольно устойчивым.
2. Интерпретируемость и режим чтения уже менее стабильны, но всё ещё остаются заметно выше случайного уровня.
3. Доминирующий фактор по `|beta|` и по `|B|` в среднем совпадает не идеально, но достаточно часто, чтобы использовать это как опорный сигнал для содержательной интерпретации.

### Примеры серий с высокой согласованностью

| Код | regime consistency | interpretability consistency | top beta consistency | reading mode consistency |
|---|---:|---:|---:|---:|
| `21` | 1.000000 | 0.842105 | 0.902256 | 0.842105 |
| `18` | 1.000000 | 0.827068 | 0.947368 | 0.827068 |
| `31` | 1.000000 | 0.819549 | 0.879699 | 0.819549 |
| `16` | 1.000000 | 0.819549 | 0.909774 | 0.819549 |
| `15` | 1.000000 | 0.812030 | 0.984962 | 0.812030 |

### Комментарий

Этот baseline-consistency уже поддерживает важный эмпирический тезис:

> даже если конкретное окно редко допускает прямую регрессионную интерпретацию, режим и тип допустимого чтения на реальных промышленных рядах всё же оказываются достаточно устойчивыми, чтобы строить на них дальнейшие отраслевые кейсы.

В то же время это ещё не финальный `A5`, потому что пока не проверена устойчивость по:

- `W = 36` и `W = 48`;
- `L = 6`;
- сравнительному анализу между `adj_smoothed`, `adj_unsmoothed` и `raw`.

### Expanded consistency: `W = 24, 36, 48`, `L = 6`

Расширенные артефакты:

- [overall_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed_expanded/overall_summary.csv)
- [series_consistency.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed_expanded/series_consistency.csv)
- [variant_consistency.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed_expanded/variant_consistency.csv)
- [variant_consistency.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_consistency_adj_unsmoothed_expanded/variant_consistency.png)

Интегральная expanded-сводка:

| Метрика | Среднее значение |
|---|---:|
| regime consistency | 0.929731 |
| interpretability consistency | 0.747145 |
| tool consistency | 0.848679 |
| top beta consistency | 0.718697 |
| top B consistency | 0.709998 |
| reading mode consistency | 0.745210 |

Динамика по окнам:

| Окно | regime consistency | interpretability consistency | tool consistency | top beta consistency | top B consistency |
|---|---:|---:|---:|---:|---:|
| `24` | 0.820189 | 0.643257 | 0.746761 | 0.691409 | 0.675252 |
| `36` | 0.881836 | 0.686302 | 0.795850 | 0.716195 | 0.711095 |
| `48` | 0.911185 | 0.724185 | 0.827445 | 0.736678 | 0.730822 |

Это один из самых важных реальных результатов на текущий момент:

> увеличение окна повышает не только долю интерпретируемых окон, но и устойчивость самого маршрута анализа.

При этом рост consistency идёт не за счёт “упрощения” динамики до базовой модели, а за счёт уменьшения доли вырожденных окон и повышения согласованности между режимом, интерпретируемостью и допустимой формой чтения.

Таким образом, expanded `A5` уже даёт хорошее эмпирическое основание для следующего шага — отраслевых карточек и deep-case анализа.

---

## A6. Кейсовые отраслевые карточки

### Артефакты

- [case_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_case_cards/case_summary.csv)
- [shock_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_case_cards/shock_summary.csv)
- [hierarchy_summary.csv](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_case_cards/hierarchy_summary.csv)
- [case_cards.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_case_cards/case_cards.md)
- [case_overview.png](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/outputs/real_data/ipp_case_cards/case_overview.png)

Для кейсовых карточек был взят следующий стартовый набор отраслей:

- ПРОМ — промышленность в целом;
- `06` и `06.1` — нефть и газ;
- `10` — пищевые продукты;
- `35.1+35.3` — производство, передача и тепло;
- `21` — лекарства;
- `26` — компьютеры и электроника;
- `38` — отходы.

### Интегральная сводка по кейсам

| Код | Интерпрет. | Plateau | Дом. режим | Дом. контур | mean R2 | mean B_sum |
|---|---:|---:|---|---|---:|---:|
| ПРОМ | 0.055 | 0.915 | `plateau_degenerate` | `do_not_read` | 0.415 | 0.0011 |
| `06` | 0.366 | 0.614 | `plateau_degenerate` | `do_not_read` | 0.284 | 0.0016 |
| `06.1` | 0.366 | 0.614 | `plateau_degenerate` | `do_not_read` | 0.284 | 0.0016 |
| `10` | 0.708 | 0.116 | `turbulent_informative` | `phase_caution` | 0.580 | 0.0049 |
| `21` | 0.799 | 0.000 | `turbulent_informative` | `phase_caution` | 0.434 | 0.0008 |
| `26` | 0.774 | 0.000 | `turbulent_informative` | `phase_caution` | 0.493 | 0.0016 |
| `35.1+35.3` | 0.711 | 0.036 | `turbulent_informative` | `phase_caution` | 0.443 | -0.0030 |
| `38` | 0.766 | 0.000 | `turbulent_informative` | `phase_caution` | 0.323 | -0.0003 |

### Что здесь видно сразу

1. Агрегированный ряд ПРОМ почти целиком вырождается в `plateau_degenerate`. Это подтверждает предыдущие разделы: наиболее агрегированные ряды плохо подходят для буквального чтения короткой лаговой структуры.
2. Нефтяной контур `06 / 06.1` ведёт себя почти так же, как промышленность в целом: высокая plateau-доля и доминирующий режим `do_not_read`.
3. Отрасли `10`, `21`, `26`, `35.1+35.3` и `38` оказываются содержательно более информативными, но и они не переходят в “чистый memory-reading”. Их нормальный контур чтения — это `phase_caution`.
4. Во всех кейсах доминирующий фактор по `|beta|` остаётся `X_n`. Это сильный эмпирический сигнал: на реальных месячных `IPP`-рядах текущий уровень обычно важнее, чем устойчивая и явно доминирующая память первого лага.

### Шоковые точки 2020 и 2022

Наиболее показательные наблюдения по шокам таковы:

| Код | 2020 режим | 2020 контур | 2022 режим | 2022 контур |
|---|---|---|---|---|
| ПРОМ | `turbulent_informative` | `phase_caution` | `plateau_degenerate` | `do_not_read` |
| `06` | `turbulent_informative` | `phase_caution` | `plateau_degenerate` | `do_not_read` |
| `06.1` | `turbulent_informative` | `phase_caution` | `plateau_degenerate` | `do_not_read` |
| `10` | `turbulent_informative` | `phase_caution` | `turbulent_informative` | `phase_caution` |
| `21` | `turbulent_informative` | `phase_caution` | `turbulent_informative` | `phase_caution` |
| `26` | `turbulent_informative` | `phase_caution` | `turbulent_informative` | `phase_caution` |
| `35.1+35.3` | `turbulent_informative` | `beta_bsum_caution` | `turbulent_informative` | `do_not_read` |
| `38` | `turbulent_informative` | `phase_caution` | `turbulent_informative` | `phase_caution` |

Это даёт уже содержательную отраслевую картину:

1. Для агрегатов и нефтяного контура `2022` выглядит как более “закрывающий” шок: окно чаще перестаёт быть читаемым и уходит в `do_not_read`.
2. Для пищевой, фармацевтической, электронной и отходной отраслей и `2020`, и `2022` остаются турбулентными, но ещё информативными эпизодами.
3. Энергетический кейс `35.1+35.3` выделяется отдельно: в `2020` он ещё допускает осторожное чтение через `beta_bsum_caution`, а к `2022` уходит в более жёсткий режим `do_not_read`.

### Representative cards

#### Общая карта кейсов

![Case Overview](outputs/real_data/ipp_case_cards/case_overview.png)

#### Промышленность в целом

![PROM Card](outputs/real_data/ipp_case_cards/card_ПРОМ.png)

#### Лекарственные средства

![21 Card](outputs/real_data/ipp_case_cards/card_21.png)

#### Энергетический кейс `35.1+35.3`

![35 Card](outputs/real_data/ipp_case_cards/card_35.1+35.3_Производство_передача_и.png)

### Методологический смысл `A6`

Отраслевые карточки подтверждают центральный результат всей real-data линии:

> даже там, где окно достаточно часто оказывается интерпретируемым, нормальным режимом чтения реального промышленного ряда остаётся не “прямая память”, а осторожное фазово-структурное чтение.

Именно поэтому маршрут главы 2 сохраняется на эмпирике почти без изменений:

1. сначала определить режим окна;
2. затем проверить, не вырождено ли оно;
3. потом выбрать допустимый контур анализа;
4. и только после этого читать `beta`, `B_sum`, `ENTER` и `Stepwise`.

### Что делать дальше

Первый круг по промышленным индексам теперь закрыт на уровне `A0–A6`. Следующие логичные шаги:

1. переходить к линии выручки (`B1` и `B2`) на уже подготовленном годовом корпусе;
2. после этого строить bridge-case между `IPP` и выручкой только для ограниченного набора отраслей;
3. в текст диплома переносить real-data линию уже как отдельную главу, опираясь на `results.md` и сохранённые карточки.
