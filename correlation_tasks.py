"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"

def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    corr = group[FEATURES].corrwith(group[TARGET], method="pearson")
    return {str(feature): float(value) for feature, value in corr.items()}

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """

    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    df["Gender"] = df["Gender"].str.strip()

    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    women_corr = _mri_correlations(women)
    men_corr = _mri_correlations(men)

    strongest_feature = ""
    strongest_abs = -1.0
    for corr in (women_corr, men_corr):
        for feature, value in corr.items():
            if pd.notna(value) and abs(value) > strongest_abs:
                strongest_abs = abs(value)
                strongest_feature = feature

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature,
    )
