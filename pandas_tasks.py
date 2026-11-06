"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """

    df = pd.read_csv(data.csv_path)

    missing_by_column = {
        str(column): int(count) for column, count in df.isna().sum().items()
    }

    adults_over_30_count = int((df["Age"] > 30).sum())

    mean_age = df.groupby("Pclass")["Age"].mean()
    mean_age_by_pclass = {int(pclass): float(age) for pclass, age in mean_age.items()}

    survival = df.groupby("Pclass")["Survived"].mean()
    survival_rate_by_pclass = {
        int(pclass): float(rate) for pclass, rate in survival.items()
    }

    highest_fares = [float(fare) for fare in df["Fare"].nlargest(5)]

    return TitanicSummary(
        row_count=int(len(df)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )
