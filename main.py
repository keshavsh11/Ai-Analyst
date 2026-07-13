from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import Any, Dict

import pandas as pd
import numpy as np

import re

from difflib import get_close_matches


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# REQUEST MODEL
# ==========================================

class AnalyzeRequest(BaseModel):

    filePath: str
    instructions: Dict[str, Any]


# ==========================================
# NORMALIZE COLUMN
# ==========================================

def normalize_column(column):

    return re.sub(
        r'[^a-z0-9]',
        '',
        str(column).lower()
    )


# ==========================================
# RESOLVE COLUMN NAME
# ==========================================

def resolve_column_name(

    requested_column,
    actual_columns

):

    if not requested_column:
        return None

    # EXACT MATCH
    if requested_column in actual_columns:
        return requested_column

    normalized_actual = {

        normalize_column(col): col
        for col in actual_columns
    }

    normalized_requested = normalize_column(
        requested_column
    )

    # NORMALIZED MATCH
    if normalized_requested in normalized_actual:

        return normalized_actual[
            normalized_requested
        ]

    # FUZZY MATCH
    closest = get_close_matches(

        normalized_requested,

        normalized_actual.keys(),

        n=1,

        cutoff=0.55
    )

    if closest:

        return normalized_actual[
            closest[0]
        ]

    return None


# ==========================================
# FIND BEST NUMERIC COLUMN
# ==========================================

def find_best_numeric_column(df):

    numeric_columns = []

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid_count = converted.notna().sum()

        if valid_count > 0:

            numeric_columns.append(
                (
                    column,
                    valid_count
                )
            )

    if not numeric_columns:
        return None

    numeric_columns.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return numeric_columns[0][0]


# ==========================================
# FIND BEST GROUP COLUMN
# ==========================================

def find_best_group_column(df):

    preferred = [

        "year",
        "date",
        "month",
        "name",
        "category",
        "repository",
        "repo"
    ]

    normalized_map = {

        normalize_column(col): col
        for col in df.columns
    }

    for pref in preferred:

        if pref in normalized_map:
            return normalized_map[pref]

    # fallback = first non numeric column
    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.isna().sum() > 0:
            return col

    return df.columns[0]


# ==========================================
# CLEAN JSON VALUES
# ==========================================

def clean_json_data(chart_data):

    cleaned_data = []

    for row in chart_data:

        cleaned_row = {}

        for key, value in row.items():

            # NUMPY INTEGER
            if isinstance(
                value,
                (
                    np.integer,
                    np.int64
                )
            ):

                cleaned_row[key] = int(value)

            # FLOAT
            elif isinstance(
                value,
                (
                    np.floating,
                    np.float64,
                    float
                )
            ):

                if (
                    np.isnan(value)
                    or
                    np.isinf(value)
                ):

                    cleaned_row[key] = None

                else:

                    cleaned_row[key] = round(
                        float(value),
                        2
                    )

            else:

                cleaned_row[key] = value

        cleaned_data.append(
            cleaned_row
        )

    return cleaned_data


# ==========================================
# ANALYZE ROUTE
# ==========================================

@app.post("/analyze")
def analyze_data(request: AnalyzeRequest):

    # ==========================================
    # LOAD CSV
    # ==========================================

    try:

        df = pd.read_csv(
            request.filePath
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load CSV: {str(e)}"
        )

    # EMPTY FILE
    if df.empty:

        raise HTTPException(
            status_code=400,
            detail="CSV file is empty"
        )

    # ==========================================
    # EXTRACT INSTRUCTIONS
    # ==========================================

    instructions = request.instructions

    operation = instructions.get(
        "operation"
    )

    metric = instructions.get(
        "metric"
    )

    group_by = instructions.get(
        "group_by"
    )

    limit = int(
        instructions.get(
            "limit",
            10
        )
    )

    filters = instructions.get(
        "filters",
        {}
    )

    sort_order = instructions.get(
        "sort_order",
        "desc"
    )

    if not operation:

        raise HTTPException(
            status_code=400,
            detail="Operation is missing"
        )

    # ==========================================
    # AUTO DETECT COLUMNS
    # ==========================================

    if not metric or metric == "*":

        metric = find_best_numeric_column(df)

    else:

        metric = resolve_column_name(
            metric,
            df.columns
        )

    if group_by:

        group_by = resolve_column_name(
            group_by,
            df.columns
        )

    # AUTO GROUP FOR TREND
    if (
        operation == "trend_analysis"
        and not group_by
    ):

        group_by = find_best_group_column(df)

    # ==========================================
    # VALIDATE METRIC
    # ==========================================

    if (
        operation != "count"
        and metric
    ):

        if metric not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=f"Metric column '{metric}' not found"
            )

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        df = df.dropna(
            subset=[metric]
        )

    # ==========================================
    # APPLY FILTERS
    # ==========================================

    for column, conditions in filters.items():

        resolved_column = resolve_column_name(
            column,
            df.columns
        )

        if not resolved_column:
            continue

        try:

            df[resolved_column] = pd.to_numeric(
                df[resolved_column],
                errors="ignore"
            )

        except:
            pass

        # BETWEEN
        if "between" in conditions:

            start, end = conditions["between"]

            df = df[
                (
                    df[resolved_column] >= start
                )
                &
                (
                    df[resolved_column] <= end
                )
            ]

        # GREATER THAN
        if "greater_than" in conditions:

            df = df[
                df[resolved_column]
                >
                conditions["greater_than"]
            ]

        # LESS THAN
        if "less_than" in conditions:

            df = df[
                df[resolved_column]
                <
                conditions["less_than"]
            ]

        # EQUALS
        if "equals" in conditions:

            df = df[
                df[resolved_column]
                ==
                conditions["equals"]
            ]

    # ==========================================
    # EMPTY DATA
    # ==========================================

    if df.empty:

        return {
            "chartData": [],
            "warning": "No matching data found"
        }

    chart_data = []

    # ==========================================
    # EXECUTE OPERATION
    # ==========================================

    try:

        # TREND ANALYSIS
        if operation == "trend_analysis":

            grouped = (
                df.groupby(group_by)[metric]
                .mean()
                .reset_index()
            )

            grouped = grouped.sort_values(
                by=group_by
            )

            chart_data = grouped.to_dict(
                orient="records"
            )

        # MAX VALUE
        elif operation == "max_value":

            max_row = df.loc[
                df[metric].idxmax()
            ]

            chart_data = [
                max_row.to_dict()
            ]

        # MIN VALUE
        elif operation == "min_value":

            min_row = df.loc[
                df[metric].idxmin()
            ]

            chart_data = [
                min_row.to_dict()
            ]

        # AVERAGE
        elif operation == "average":

            chart_data = [
                {
                    "average":
                    float(df[metric].mean())
                }
            ]

        # SUM
        elif operation == "sum":

            chart_data = [
                {
                    "sum":
                    float(df[metric].sum())
                }
            ]

        # COUNT
        elif operation == "count":

            if group_by:

                grouped = (
                    df.groupby(group_by)
                    .size()
                    .reset_index(name="count")
                )

                chart_data = grouped.to_dict(
                    orient="records"
                )

            else:

                chart_data = [
                    {
                        "count":
                        int(len(df))
                    }
                ]

        # TOP N
        elif operation == "top_n":

            ascending = (
                sort_order == "asc"
            )

            top_rows = df.sort_values(

                by=metric,

                ascending=ascending
            ).head(limit)

            chart_data = top_rows.to_dict(
                orient="records"
            )

        # GROUPED SUMMARY
        elif operation == "grouped_summary":

            grouped = (
                df.groupby(group_by)[metric]
                .sum()
                .reset_index()
            )

            chart_data = grouped.to_dict(
                orient="records"
            )

        else:

            raise HTTPException(
                status_code=400,
                detail=f"Unsupported operation '{operation}'"
            )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # ==========================================
    # CLEAN DATA
    # ==========================================

    cleaned_data = clean_json_data(
        chart_data
    )

    # ==========================================
    # RESPONSE
    # ==========================================

    return {

        "chartData": cleaned_data,

        "meta": {

            "rows": len(df),

            "metric": metric,

            "group_by": group_by,

            "operation": operation
        }
    }