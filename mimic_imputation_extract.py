#!/usr/bin/env python3

"""Extract a dataset from MIMIC-III for imputation experiments

This is based on `mimic_direct_extract.py`, slimmed down to our needs and
modified to extract the sort of dataset we require for our experiments.
"""

import os
from typing import Union, Optional
import logging

import numpy as np
import pandas as pd  # type: ignore
from tap import Tap  # typed-argument-parser package
import tableschema  # type: ignore

from datapackage_io_util import (
    load_datapackage_schema,
    save_sanitized_df_to_csv,
    sanitize_df,
)
from mimic_querier import (
    get_values_by_name_from_df_column_or_index,
    MIMIC_Querier,
)


QueryArgsType = dict[str, Union[str, int]]
FilenamesType = dict[str, str]

ENV_CODE_DIR = os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MIMIC_CODE_DIR = ENV_CODE_DIR if os.path.exists(ENV_CODE_DIR) else SCRIPT_DIR

OUTPUT_FILENAMES = {
    "population": "population_data.csv",  # matches the population schema
    "clinical": "clinical_data.csv",  # matches the clinical schema
    "summary": "summary_data.csv",  # full summary spreadsheet with multiindex
    "condensed": "condensed_summary_data.csv",  # simplified summary
}

ID_COLS = ["subject_id", "hadm_id", "icustay_id"]
ITEM_COLS = ["itemid", "LEVEL1", "LEVEL2"]


class ImputationArgs(Tap):
    outdir: str
    """The directory for output data"""
    pop_size: int = 0
    """Size of population to extract. 0 means extract all patients"""
    exclude_elective: bool = True
    """Exclude elective surgery patients from data"""

    min_observations: int = 5
    """Patients with fewer than this number of observations in any of the
    observation types will be removed."""
    min_age: int = 15
    """Minimum age of patients to be included"""
    min_duration: int = 36
    """Minimum duration (in hours) of (first) ICU stay to be included"""
    max_data_duration: int = 240
    """Maximum number of hours of data to be included"""
    survival_window: int = 30
    """Number of days after entering ICU to consider for outcome variable"""

    psql_host: Optional[str]
    """PostgreSQL host; use Unix socket if not specified;
    you can specify "/var/run/postgresql/" explicitly if wished"""
    psql_port: int = 5432
    """PostgreSQL port"""
    psql_user: Optional[str]
    """PostgreSQL username"""
    psql_password: Optional[str]
    """PostgreSQL password"""
    psql_dbname: str = "mimic"
    """PostgreSQL database name"""
    psql_schema_name: str = "mimiciii"
    """PostgreSQL schema name"""

    resource_path: str = os.path.join(MIMIC_CODE_DIR, "resources")
    """The directory containing the code resources"""
    queries_path: str = os.path.join(
        MIMIC_CODE_DIR, "SQL_Queries", "imputation"
    )
    """The directory containing the internal SQL queries"""

    reload_population: bool = True
    """Whether to use existing extracted population data if present;
    if this option is set to false, then the population data is always
    extracted from the MIMIC database"""
    reload_clinical: bool = True
    """Whether to use existing clinical data if present;
    if this option is set to false, then the clinical data is always
    extracted from the MIMIC database"""

    loglevel: str = "INFO"
    """The logging level (DEBUG, INFO, WARN)"""

    def process_args(self):
        if not os.path.isdir(self.resource_path):
            raise ValueError(
                f"Invalid resource_path: {self.resource_path}; "
                + "you may have to set the MIMIC_EXTRACT_CODE_DIR environment "
                + "variable"
            )

        if not os.path.isdir(self.outdir):
            raise ValueError(f"outdir: {self.outdir} does not exist")

        if self.survival_window > 90:
            raise ValueError(
                "survival_window must be <= 90, as deaths beyond "
                "90 days may not be recorded"
            )

        if self.psql_host == "SOCKET":
            self.psql_host = None


def initialise_logging(loglevel: str):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=numeric_level
    )


def insert_pop_size(filename: str, pop_size: int) -> str:
    """Modify the output filename to possibly include the population size"""
    if pop_size > 0:
        fn_parts = os.path.splitext(filename)
        return f"{fn_parts[0]}_{pop_size}{fn_parts[1]}"
    return filename


def get_filenames(args: ImputationArgs) -> FilenamesType:
    filenames = {"outdir": args.outdir}
    for filetype, filename in OUTPUT_FILENAMES.items():
        # "columns" are independent of the population size
        if "columns" in filetype:
            filenames[filetype] = os.path.join(args.outdir, filename)
        else:
            filenames[filetype] = os.path.join(
                args.outdir, insert_pop_size(filename, args.pop_size)
            )

    filenames["mimic_mapping"] = os.path.join(
        args.resource_path, "itemid_to_variable_map.csv"
    )
    filenames["range"] = os.path.join(
        args.resource_path, "variable_ranges.csv"
    )
    filenames["keep_variables"] = os.path.join(
        args.resource_path, "imputation", "keep_variables.csv"
    )

    return filenames


def get_querier(args: ImputationArgs) -> MIMIC_Querier:
    schema_name = "public," + args.psql_schema_name
    query_args = get_query_args(args)
    return MIMIC_Querier(query_args=query_args, schema_name=schema_name)


def get_query_args(args: ImputationArgs) -> QueryArgsType:
    query_args: QueryArgsType = {"dbname": args.psql_dbname}
    for conn_param in ["host", "port", "user", "password"]:
        if (value := getattr(args, f"psql_{conn_param}")) is not None:
            query_args[conn_param] = value

    return query_args


def to_hours(delta: pd.Timedelta) -> int:
    return delta.days * 24 + delta.seconds // 3600


def to_days(delta: pd.Timedelta) -> int:
    return delta.days


def extraction_required(reload: bool, path: str) -> bool:
    if reload and os.path.isfile(path):
        return False
    return True


def load_population_data(
    args: ImputationArgs,
    filenames: FilenamesType,
    querier: MIMIC_Querier,
) -> pd.DataFrame:
    # Load specs for output tables
    population_data_schema = load_datapackage_schema(
        os.path.join(
            args.resource_path, "imputation", "population_data_spec.json"
        )
    )

    population_path = filenames["population"]

    if extraction_required(args.reload_population, population_path):
        population = extract_population_data(
            args, querier, population_data_schema
        )
        logging.info("Storing data to %s", population_path)
        save_sanitized_df_to_csv(
            population_path, population, population_data_schema
        )
    else:
        population = reload_population_data(
            population_path, population_data_schema
        )

    # So all subsequent queries will limit to just that already
    # extracted in population.
    querier.add_exclusion_criteria_from_df(
        population, columns=["hadm_id", "subject_id"]
    )
    logging.info("Loaded population data")

    return population


def extract_population_data(
    args: ImputationArgs,
    querier: MIMIC_Querier,
    population_data_schema: tableschema.Schema,
) -> pd.DataFrame:
    logging.info("Extracting population data")
    pop_size_string = ""
    if args.pop_size > 0:
        pop_size_string = f"LIMIT {args.pop_size}"

    min_age_string = str(args.min_age)
    min_dur_string = str(args.min_duration)
    min_day_string = str(args.min_duration / 24)
    admission_types = "'EMERGENCY', 'URGENT'"
    if not args.exclude_elective:
        admission_types += ", 'ELECTIVE'"

    template_vars = dict(
        limit=pop_size_string,
        min_age=min_age_string,
        min_dur=min_dur_string,
        min_day=min_day_string,
        admission_types=admission_types,
    )

    query_file = os.path.join(args.queries_path, "population.sql")
    population = querier.query(
        query_file=query_file, extra_template_vars=template_vars
    )

    append_population_outcome_to_population(args, population)
    population = sanitize_df(population, population_data_schema)

    return population


def reload_population_data(
    population_path: str,
    population_data_schema: tableschema.Schema,
) -> pd.DataFrame:
    logging.info("Reloading population data from %s", population_path)
    population = pd.read_csv(population_path)
    population = sanitize_df(population, population_data_schema)

    return population


def append_population_outcome_to_population(
    args: ImputationArgs,
    population: pd.DataFrame,
) -> None:
    population["survival_time"] = population["dod"] - population["intime"]
    population["survival_days"] = population["survival_time"].apply(to_days)
    # 1 = died within survival_window, 0 = survived that period
    population["outcome"] = (
        (population["expire_flag"] == 1)
        & (  # we know date of death
            population["survival_days"] <= args.survival_window
        )
    ).astype(int)
    population.drop(columns=["survival_time"], inplace=True)


def append_population_outcome_to_df(
    df: pd.DataFrame,
    population: pd.DataFrame,
) -> None:
    if df.index.names != ID_COLS:
        raise ValueError(f"df index names do not match the expected {ID_COLS}")

    df_idx = df.index
    outcomes = population.loc[df_idx, "outcome"]
    if isinstance(df.columns, pd.MultiIndex):
        df[("outcome",) * df.columns.nlevels] = outcomes


def load_clinical_data(
    args: ImputationArgs,
    filenames: FilenamesType,
    querier: MIMIC_Querier,
    population: pd.DataFrame,
) -> pd.DataFrame:
    # Load specs for output tables
    clinical_data_schema = load_datapackage_schema(
        os.path.join(
            args.resource_path, "imputation", "clinical_data_spec.json"
        )
    )

    clinical_path = filenames["clinical"]

    if extraction_required(args.reload_clinical, clinical_path):
        clinical = extract_clinical_data(
            args,
            filenames,
            querier,
            population,
            clinical_data_schema,
        )
        logging.info("Storing data to %s", clinical_path)
        save_sanitized_df_to_csv(clinical_path, clinical, clinical_data_schema)
    else:
        clinical = reload_clinical_data(clinical_path, clinical_data_schema)

    logging.info("Loaded raw clinical data")

    return clinical


def extract_clinical_data(
    args: ImputationArgs,
    filenames: FilenamesType,
    querier: MIMIC_Querier,
    population: pd.DataFrame,
    clinical_data_schema: tableschema.Schema,
) -> pd.DataFrame:
    logging.info("Extracting raw clinical data")

    # We only extract clinical data from the patients in our population
    icuids_to_keep = get_values_by_name_from_df_column_or_index(
        population, "icustay_id"
    )
    icuids_to_keep = {str(s) for s in icuids_to_keep}

    # We only want chart events noted as "chartevents" in the mapping file
    var_map = get_variable_mapping(filenames["mimic_mapping"])
    var_map_chartevents = var_map.loc[var_map["LINKSTO"] == "chartevents"]
    chartids_to_keep = var_map_chartevents.ITEMID
    chartitems_to_keep = {str(i) for i in chartids_to_keep}

    template_vars = {
        "icuids": ",".join(icuids_to_keep),
        "chitem": ",".join(chartitems_to_keep),
    }

    # Select out SID, TIME, ITEMID, VALUE from each of the sources!
    query_file = os.path.join(args.queries_path, "clinical.sql")
    clinical = querier.query(
        query_file=query_file, extra_template_vars=template_vars
    )
    clinical = clinical.rename(columns={"valuenum": "value"})

    clinical = sanitize_df(clinical, clinical_data_schema)

    return clinical


def reload_clinical_data(
    clinical_path: str,
    clinical_data_schema: tableschema.Schema,
) -> pd.DataFrame:
    logging.info("Reloading raw clinical data from %s", clinical_path)
    clinical = pd.read_csv(clinical_path)
    clinical = sanitize_df(clinical, clinical_data_schema)

    return clinical


def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = pd.read_csv(mimic_mapping_filename, index_col=None)
    var_map = var_map[
        (var_map["LEVEL2"] != "")
        & (var_map["COUNT"] > 0)
        & (var_map["STATUS"] == "ready")
    ]
    var_map["ITEMID"] = var_map["ITEMID"].astype(int)

    return var_map


def process_clinical_data(
    args: ImputationArgs,
    filenames: FilenamesType,
    population: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    labelled = label_clinical_data(filenames, population, clinical)
    standardize_units(labelled, name_col="LEVEL1", inplace=True)
    var_ranges = get_variable_ranges(filenames["range"])
    cleaned = apply_variable_limits(labelled, var_ranges, "LEVEL2")
    cleaned = cleaned[~pd.isna(cleaned["value"])]
    filtered_clinical_data = filter_clinical_data(args, filenames, cleaned)

    return filtered_clinical_data


def label_clinical_data(
    filenames: FilenamesType,
    population: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    """Add labels and hours_in to specify the time into the ICU stay"""
    labelled = clinical.copy()
    labelled = labelled.join(population[["intime"]])
    labelled["hours_in"] = (labelled["charttime"] - labelled["intime"]).apply(
        to_hours
    )

    labelled.set_index("itemid", append=True, inplace=True)

    var_map = get_variable_mapping(filenames["mimic_mapping"])

    # Most of the var_map DataFrame is irrelevant for our purposes;
    # we only want to append the LEVEL1 and LEVEL2 fields to our clinical
    # data.
    var_map = (
        var_map[["LEVEL2", "ITEMID", "LEVEL1"]]
        .rename(columns={"ITEMID": "itemid"})
        .set_index("itemid")
    )

    labelled = labelled.join(var_map).set_index(
        ["LEVEL1", "LEVEL2"], append=True
    )

    return labelled


# name, unit, range_check_function, convert_function
UNIT_CONVERSIONS = [
    ("weight", "oz", None, lambda x: x / 16.0 * 0.45359237),
    ("weight", "lbs", None, lambda x: x * 0.45359237),
    ("fraction inspired oxygen", None, lambda x: x > 1, lambda x: x / 100.0),
    ("oxygen saturation", None, lambda x: x <= 1, lambda x: x * 100.0),
    ("temperature", "f", lambda x: x > 79, lambda x: (x - 32) * 5.0 / 9),
    ("height", "in", None, lambda x: x * 2.54),
]


def standardize_units(
    clinical: pd.DataFrame,
    name_col: str = "itemid",
    unit_col: str = "valueuom",
    value_col: str = "value",
    inplace: bool = True,
) -> pd.DataFrame:
    if not inplace:
        clinical = clinical.copy()

    # Need to use this more complicated function as the column of interest
    # might be part of the index
    name_col_vals = get_values_by_name_from_df_column_or_index(
        clinical, name_col
    )
    unit_col_vals = get_values_by_name_from_df_column_or_index(
        clinical, unit_col
    )

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except Exception as e:
        logging.warning("Can't call *.str")
        logging.warning("name_col_vals=%s", name_col_vals)
        logging.warning("unit_col_vals=%s", unit_col_vals)
        raise e

    def name_filter(n):
        return name_col_vals.contains(n, case=False, na=False)

    def unit_filter(n):
        return unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None:
            needs_conversion_filter_idx |= name_filter(unit) | unit_filter(
                unit
            )
        if rng_check_fn is not None:
            needs_conversion_filter_idx |= rng_check_fn(clinical[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        clinical.loc[idx, value_col] = convert_fn(clinical[value_col][idx])

    return clinical


def get_variable_ranges(range_filename):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [
        "LEVEL2",
        "OUTLIER LOW",
        "VALID LOW",
        "IMPUTE",
        "VALID HIGH",
        "OUTLIER HIGH",
    ]
    to_rename = dict(zip(columns, [c.replace(" ", "_") for c in columns]))
    to_rename["LEVEL2"] = "VARIABLE"
    var_ranges = pd.read_csv(range_filename, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(columns=to_rename, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset="VARIABLE", keep="first")
    var_ranges["VARIABLE"] = var_ranges["VARIABLE"].str.lower()
    var_ranges.set_index("VARIABLE", inplace=True)
    var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]

    return var_ranges


def apply_variable_limits(
    clinical: pd.DataFrame,
    var_ranges: pd.DataFrame,
    var_names_index_col: str = "LEVEL2",
) -> pd.DataFrame:
    idx_vals = clinical.index.get_level_values(var_names_index_col)
    non_null_idx = ~clinical.value.isnull()
    var_names = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    logging.info("Variable outlier information:")
    for var_name in var_names:
        var_name_lower = var_name.lower()
        if var_name_lower not in var_range_names:
            logging.debug("No known ranges for %s, skipping", var_name)
            continue

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x]
            for x in ("OUTLIER_LOW", "OUTLIER_HIGH", "VALID_LOW", "VALID_HIGH")
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx = clinical.value < outlier_low_val
        outlier_high_idx = clinical.value > outlier_high_val
        valid_low_idx = ~outlier_low_idx & (clinical.value < valid_low_val)
        valid_high_idx = ~outlier_high_idx & (clinical.value > valid_high_val)

        var_outlier_idx = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        clinical.loc[var_outlier_idx, "value"] = np.nan
        clinical.loc[var_valid_low_idx, "value"] = valid_low_val
        clinical.loc[var_valid_high_idx, "value"] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0:
            logging.info(
                "%s had %d / %d rows cleaned:",
                var_name,
                n_outlier + n_valid_low + n_valid_high,
                sum(running_idx),
            )
            logging.info(
                "%d rows were strict outliers, set to np.nan", n_outlier
            )
            logging.info(
                "%d rows were low valid outliers, set to %.2f",
                n_valid_low,
                valid_low_val,
            )
            logging.info(
                "%d rows were high valid outliers, set to %.2f",
                n_valid_high,
                valid_high_val,
            )

    return clinical


def filter_clinical_data(
    args: ImputationArgs,
    filenames: FilenamesType,
    processed_clinical_data: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows with data of interest and within specified time limit"""
    variables_keep = pd.read_csv(filenames["keep_variables"])
    level2_vals = get_values_by_name_from_df_column_or_index(
        processed_clinical_data, "LEVEL2"
    )
    keep_idx = level2_vals.isin(variables_keep["LEVEL2"])

    keep_idx &= (0 <= processed_clinical_data["hours_in"]) & (
        processed_clinical_data["hours_in"] <= args.max_data_duration
    )

    filtered_clinical_data = processed_clinical_data[keep_idx]

    return filtered_clinical_data


def summarise_clinical_data(clinical_data: pd.DataFrame) -> pd.DataFrame:
    df = clinical_data.reset_index()
    df = df[ID_COLS + ["LEVEL2", "value"]]
    df.rename(columns={"LEVEL2": "item"}, inplace=True)
    df_summary = df.groupby(ID_COLS + ["item"]).agg(["count", "mean", "std"])
    df_summary.columns = df_summary.columns.droplevel(0)
    df_summary.columns.names = ["aggregation"]
    df_summary.loc[:, "count"] = df_summary.loc[:, "count"].fillna(0)
    # For some reason, the count column contains floats, not ints
    df_summary = df_summary.astype({"count": int})

    # Pivot table so that we have one row per ICU stay
    df_summary = df_summary.reset_index().pivot(index=ID_COLS, columns="item")
    df_summary.columns = df_summary.columns.reorder_levels(
        order=["item", "aggregation"]
    )
    df_summary = df_summary[sorted(df_summary.columns)]

    return df_summary


def remove_rare(
    args: ImputationArgs, filenames: FilenamesType, summary: pd.DataFrame
) -> pd.DataFrame:
    summ = summary.copy()
    idx = pd.IndexSlice
    summ.loc[:, idx[:, "count"]] = summ.loc[:, idx[:, "count"]].fillna(0)

    keep_variables = pd.read_csv(filenames["keep_variables"])["LEVEL2"]
    keep_idx = pd.Series([True] * len(summ), index=summ.index)
    for variable in keep_variables:
        keep_idx &= summ.loc[:, (variable, "count")] >= args.min_observations

    return summ.loc[keep_idx]


def count_col_idx(col_idx: pd.MultiIndex) -> pd.MultiIndex:
    count_idx = col_idx.get_level_values(1) == "count"
    return col_idx[count_idx]


def condense_summary_df(summary: pd.DataFrame) -> pd.DataFrame:
    """Remove the index, column MultiIndex, and count columns"""
    condensed = summary.copy()
    condensed.reset_index(drop=True, inplace=True)
    count_cols = count_col_idx(condensed.columns)
    condensed = condensed.drop(columns=count_cols)
    colnames = [f"{obs} {agg}" for (obs, agg) in condensed.columns]
    if "outcome outcome" in colnames:
        outcome_idx = colnames.index("outcome outcome")
        colnames[outcome_idx] = "outcome"
    condensed.columns = colnames

    return condensed


def main():
    args = ImputationArgs(explicit_bool=True).parse_args()
    initialise_logging(args.loglevel)
    filenames = get_filenames(args)
    querier = get_querier(args)

    population = load_population_data(args, filenames, querier)
    clinical_data = load_clinical_data(args, filenames, querier, population)
    processed_clinical_data = process_clinical_data(
        args, filenames, population, clinical_data
    )

    summarised_data = summarise_clinical_data(processed_clinical_data)
    summarised_data = remove_rare(args, filenames, summarised_data)
    append_population_outcome_to_df(summarised_data, population)
    logging.info("Saving summarised data to %s", filenames["summary"])
    save_sanitized_df_to_csv(filenames["summary"], summarised_data)

    condensed_data = condense_summary_df(summarised_data)
    logging.info("Saving condensed summary data to %s", filenames["condensed"])
    save_sanitized_df_to_csv(filenames["condensed"], condensed_data)


if __name__ == "__main__":
    main()
