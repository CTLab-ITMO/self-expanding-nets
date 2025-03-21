# Описание метрик для анализа нелинейности

### 1. GradientMeanEdgeMetric
- **Описание**: Вычисляет среднее абсолютное значение градиентов весов последнего слоя.
- **Цель**: Оценить, насколько сильно каждый вес изменяется во время обучения, показывая важность каждого ребра.

### 2. ActivationStdEdgeMetric
- **Описание**: Определяет стандартное отклонение активаций перед последним слоем, взвешенных по каждому весу.
- **Цель**: Оценить вариативность активаций для каждого ребра, что может помочь выявить наиболее "активные" веса в модели.

### 3. PerturbationSensitivityEdgeMetric
- **Описание**: Измеряет чувствительность модели к небольшому возмущению каждого веса последнего слоя.
- **Цель**: Определить, насколько сильно каждое изменение веса влияет на выход модели, что может указывать на важность каждого веса для предсказания.

### 4. CosineGradientSimilarityEdgeMetric
- **Описание**: Рассчитывает косинусное сходство между соседними градиентами весов последнего слоя.
- **Цель**: Оценить степень направленной схожести градиентов для каждого ребра, что может указывать на взаимосвязь соседних весов в обучении.
