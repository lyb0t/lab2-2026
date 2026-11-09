"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

# Импорт из папки grader_contracts
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Загружает CSV и выполняет анализ датасета Titanic."""
    
    df = pd.read_csv(data.csv_path)

    # 1) пропуски по столбцам
    missing = {col: int(df[col].isna().sum()) for col in df.columns}

    # 2) пассажиры строго старше 30
    older_than_30 = int((df["Age"] > 30).sum())

    # 3) средний возраст по Pclass
    mean_age_by_class = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Age"].mean().items()
    }

    # 4) доля выживших по Pclass
    survival_rate_by_class = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Survived"].mean().items()
    }

    # 5) топ-5 Fare по убыванию
    top_fares = (
        df["Fare"]
        .dropna()
        .sort_values(ascending=False)
        .head(5)
        .astype(float)
        .tolist()
    )

    return TitanicSummary(
        missing_counts=missing,
        older_than_30=older_than_30,
        mean_age_by_pclass=mean_age_by_class,
        survival_rate_by_pclass=survival_rate_by_class,
        top_fares=top_fares,
    )
