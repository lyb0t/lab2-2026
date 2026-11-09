"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

# Импорты из твоей папки контрактов
from grader_contracts.correlation_tasks import BrainDataInput, BrainCorrelationSummary


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _pearson_map(df: pd.DataFrame) -> dict:
    """Вспомогательная функция для подсчета корреляции признаков с MRI_Count."""
    return {
        feat: float(df[feat].corr(df[TARGET], method="pearson"))
        for feat in FEATURES
    }


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Загружает brainsize.txt и считает корреляции по группам."""
    
    # 1. Загружаем данные, учитывая табуляцию и значения NA
    df = pd.read_csv(
        data.txt_path,
        sep="\t",
        na_values=["NA"],
        keep_default_na=True,
    )

    # 2. Разделяем на мужчин и женщин (приводим к нижнему регистру на всякий случай)
    gender = df["Gender"].astype(str).str.strip().str.lower()
    males = df[gender.isin(["male", "m"])]
    females = df[gender.isin(["female", "f"])]

    # 3. Считаем корреляции для каждой группы
    male_corr = _pearson_map(males)
    female_corr = _pearson_map(females)

    # 4. Ищем признак с максимальным модулем корреляции среди обеих групп
    best_feat = None
    best_abs = -1.0
    for feat in FEATURES:
        for r in (male_corr[feat], female_corr[feat]):
            if abs(r) > best_abs:
                best_abs = abs(r)
                best_feat = feat

    # Возвращаем результат
    return BrainCorrelationSummary(
        male_correlations=male_corr,
        female_correlations=female_corr,
        strongest_mri_feature=best_feat,
    )
