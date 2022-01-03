"""Extract a dataset for imputation experiments

This is based on mimic_direct_extract.py, cleaned up, slimmed down to
our needs and modified to extract the sort of dataset we require for our
experiments.

The MIMIC-III data is stored in a PostgreSQL database.
"""

import os
import re
import sys
import time
from os.path import isfile, isdir, splitext
import argparse
import enum

import psycopg2
import spacy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from datapackage_io_util import (
    load_datapackage_schema,
    save_sanitized_df_to_csv,
    sanitize_df,
)
from heuristic_sentence_splitter import sent_tokenize_rules
from mimic_querier import get_values_by_name_from_df_column_or_index, MIMIC_Querier


matplotlib.use("Agg")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.path.join(CURRENT_DIR, "SQL_Queries")
STATICS_QUERY_PATH = os.path.join(SQL_DIR, "statics.sql")
CODES_QUERY_PATH = os.path.join(SQL_DIR, "codes.sql")
NOTES_QUERY_PATH = os.path.join(SQL_DIR, "notes.sql")

# Output filenames
STATIC_FILENAME = "static_data.csv"

DYNAMIC_FILENAME = "vitals_hourly_data.csv"
COLUMNS_FILENAME = "vitals_colnames.txt"
SUBJECTS_FILENAME = "subjects.npy"
TIMES_FILENAME = "fenceposts.npy"
DYNAMIC_HD5_FILENAME = "vitals_hourly_data.h5"
DYNAMIC_HD5_FILT_FILENAME = "all_hourly_data.h5"

CODES_HD5_FILENAME = "C.h5"
NOTES_HD5_FILENAME = "notes.hdf"  # N.h5

OUTCOME_FILENAME = "outcomes_hourly_data.csv"
OUTCOME_HD5_FILENAME = "outcomes_hourly_data.h5"
OUTCOME_COLUMNS_FILENAME = "outcomes_colnames.txt"

# SQL command params

ID_COLS = ["subject_id", "hadm_id", "icustay_id"]
ITEM_COLS = ["itemid", "label", "LEVEL1", "LEVEL2"]


def add_outcome_indicators(out_gb):
    subject_id = out_gb["subject_id"].unique()[0]
    hadm_id = out_gb["hadm_id"].unique()[0]
    max_hrs = out_gb["max_hours"].unique()[0]
    on_hrs = set()

    for _, row in out_gb.iterrows():
        on_hrs.update(range(row["starttime"], row["endtime"] + 1))

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0] * len(off_hrs) + [1] * len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame(
        {"subject_id": subject_id, "hadm_id": hadm_id, "hours_in": hours, "on": on_vals}
    )  # icustay_id': icustay_id})


def add_blank_indicators(out_gb):
    subject_id = out_gb["subject_id"].unique()[0]
    hadm_id = out_gb["hadm_id"].unique()[0]
    # icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb["max_hours"].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0] * len(hrs))
    return pd.DataFrame(
        {"subject_id": subject_id, "hadm_id": hadm_id, "hours_in": hrs, "on": vals}
    )  #'icustay_id': icustay_id,


def continuous_outcome_processing(out_data, data, icustay_timediff):
    """

    Args
    ----
    out_data : pd.DataFrame
        index=None
        Contains subset of icustay_id corresp to specific sessions where outcome observed.
    data : pd.DataFrame
        index=icustay_id
        Contains full population of static demographic data

    Returns
    -------
    out_data : pd.DataFrame
    """
    out_data["intime"] = out_data["icustay_id"].map(data["intime"].to_dict())
    out_data["outtime"] = out_data["icustay_id"].map(data["outtime"].to_dict())
    out_data["max_hours"] = out_data["icustay_id"].map(icustay_timediff)
    out_data["starttime"] = out_data["starttime"] - out_data["intime"]
    out_data["starttime"] = out_data.starttime.apply(
        lambda x: x.days * 24 + x.seconds // 3600
    )
    out_data["endtime"] = out_data["endtime"] - out_data["intime"]
    out_data["endtime"] = out_data.endtime.apply(
        lambda x: x.days * 24 + x.seconds // 3600
    )
    out_data = out_data.groupby(["icustay_id"])

    return out_data


def save_pop(data_df, static_filepath, static_data_schema):
    # Serialize to disk
    save_sanitized_df_to_csv(static_filepath, data_df, static_data_schema)

    return data_df


def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = pd.read_csv(mimic_mapping_filename, index_col=None)
    var_map = var_map[(var_map["LEVEL2"] != "") & (var_map["COUNT"] > 0)]
    var_map = var_map[(var_map["STATUS"] == "ready")]
    var_map["ITEMID"] = var_map["ITEMID"].astype(int)

    return var_map


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


UNIT_CONVERSIONS = [
    ("weight", "oz", None, lambda x: x / 16.0 * 0.45359237),
    ("weight", "lbs", None, lambda x: x * 0.45359237),
    ("fraction inspired oxygen", None, lambda x: x > 1, lambda x: x / 100.0),
    ("oxygen saturation", None, lambda x: x <= 1, lambda x: x * 100.0),
    ("temperature", "f", lambda x: x > 79, lambda x: (x - 32) * 5.0 / 9),
    ("height", "in", None, lambda x: x * 2.54),
]


def standardize_units(
    X, name_col="itemid", unit_col="valueuom", value_col="value", inplace=True
):
    if not inplace:
        X = X.copy()
    name_col_vals = get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except:
        print("Can't call *.str", file=sys.stderr)
        print(name_col_vals, file=sys.stderr)
        print(unit_col_vals, file=sys.stderr)
        raise

    name_filter = lambda n: name_col_vals.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None:
            needs_conversion_filter_idx |= name_filter(unit) | unit_filter(unit)
        if rng_check_fn is not None:
            needs_conversion_filter_idx |= rng_check_fn(X[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        X.loc[idx, value_col] = convert_fn(X[value_col][idx])

    return X


def range_unnest(df, col, out_col_name=None, reset_index=False):
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None:
        out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y + 1)],
        columns=[df.index.names[0], out_col_name],
    )

    if not reset_index:
        col_flat = col_flat.set_index(df.index.names[0])
    return col_flat


def process_and_save_numerics(
    data,
    X,
    I,
    var_map,
    var_ranges,
    out_path,
    filenames,
    group_by_level2,
    apply_var_limit,
    min_percent,
    verbose=True,
):
    assert len(data) > 0 and len(X) > 0, "Must provide some input data to process."

    var_map = (
        var_map[["LEVEL2", "ITEMID", "LEVEL1"]]
        .rename(columns={"LEVEL2": "LEVEL2", "LEVEL1": "LEVEL1", "ITEMID": "itemid"})
        .set_index("itemid")
    )

    X["value"] = pd.to_numeric(X["value"], "coerce")
    X = X.astype({k: int for k in ID_COLS})

    to_hours = lambda x: max(0, x.days * 24 + x.seconds // 3600)

    X = X.set_index("icustay_id").join(data[["intime"]])
    X["hours_in"] = (X["charttime"] - X["intime"]).apply(to_hours)

    X.drop(columns=["charttime", "intime"], inplace=True)
    X.set_index("itemid", append=True, inplace=True)

    # Pandas has a bug with the below for small X
    # X = X.join([var_map, I]).set_index(['label', 'LEVEL1', 'LEVEL2'], append=True)
    X = X.join(var_map).join(I).set_index(["label", "LEVEL1", "LEVEL2"], append=True)
    standardize_units(X, name_col="LEVEL1", inplace=True)

    if apply_var_limit > 0:
        X = apply_variable_limits(X, var_ranges, "LEVEL2")

    group_item_cols = ["LEVEL2"] if group_by_level2 else ITEM_COLS
    X = X.groupby(ID_COLS + group_item_cols + ["hours_in"]).agg(
        ["mean", "std", "count"]
    )
    X.columns = X.columns.droplevel(0)
    X.columns.names = ["Aggregation Function"]

    data["max_hours"] = (data["outtime"] - data["intime"]).apply(to_hours)

    missing_hours_fill = range_unnest(
        data, "max_hours", out_col_name="hours_in", reset_index=True
    )
    missing_hours_fill["tmp"] = np.NaN

    fill_df = data.reset_index()[ID_COLS].join(
        missing_hours_fill.set_index("icustay_id"), on="icustay_id"
    )
    fill_df.set_index(ID_COLS + ["hours_in"], inplace=True)

    # Pivot table droups NaN columns so you lose any uniformly NaN.
    X = X.unstack(level=group_item_cols)
    X.columns = X.columns.reorder_levels(
        order=group_item_cols + ["Aggregation Function"]
    )

    X = X.reindex(fill_df.index)

    X = X.sort_index(axis=0).sort_index(axis=1)

    if verbose:
        print("Shape of X : ", X.shape)

    # Turn back into columns
    if filenames["columns"] is not None:
        col_names = [str(x) for x in X.columns.values]
        columns_outpath = os.path.join(out_path, filenames["columns"])
        with open(columns_outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(col_names))

    # Get the max time for each of the subjects so we can reconstruct!
    if filenames["subjects"] is not None:
        np.save(
            os.path.join(out_path, filenames["subjects"]), data["subject_id"].to_numpy()
        )
    if filenames["times"] is not None:
        np.save(
            os.path.join(out_path, filenames["times"]), data["max_hours"].to_numpy()
        )

    # fix nan in count to be zero
    idx = pd.IndexSlice
    if group_by_level2:
        X.loc[:, idx[:, "count"]] = X.loc[:, idx[:, "count"]].fillna(0)
    else:
        X.loc[:, idx[:, :, :, :, "count"]] = X.loc[:, idx[:, :, :, :, "count"]].fillna(
            0
        )

    # Drop columns that have very few recordings
    n = round((1 - min_percent / 100.0) * X.shape[0])
    drop_col = []
    for k in X.columns:
        if k[-1] == "mean":
            if X[k].isnull().sum() > n:
                drop_col.append(k[:-1])
    X = X.drop(columns=drop_col)

    if filenames["dynamic"] is not None:
        np.save(os.path.join(out_path, filenames["dynamic"]), X.to_numpy())
    if filenames["dynamic_hd5"] is not None:
        X.to_hdf(os.path.join(out_path, filenames["dynamic_hd5"]), "X")

    return X


def save_notes(notes, notes_h5_filepath=None):
    notes_id_cols = list(set(ID_COLS).intersection(notes.columns))
    notes_metadata_cols = ["chartdate", "charttime", "category", "description"]

    notes.set_index(notes_id_cols + notes_metadata_cols, inplace=True)

    def sbd_component(doc):
        for i, token in enumerate(doc[:-2]):
            # define sentence start if period + titlecase token
            if token.text == "." and doc[i + 1].is_title:
                doc[i + 1].sent_start = True
            if token.text == "-" and doc[i + 1].text != "-":
                doc[i + 1].sent_start = True
        return doc

    # convert de-identification text into one token
    def fix_deid_tokens(text, processed_text):
        deid_regex = r"\[\*\*.{0,15}.*?\*\*\]"
        indexes = [m.span() for m in re.finditer(deid_regex, text, flags=re.IGNORECASE)]
        for start, end in indexes:
            processed_text.merge(start_idx=start, end_idx=end)
        return processed_text

    nlp = spacy.load("en_core_web_sm")  # Maybe try lg model?
    nlp.add_pipe(sbd_component, before="parser")  # insert before the parser

    def process_sections_helper(section, processed_sections):
        processed_section = nlp(section["sections"])
        processed_section = fix_deid_tokens(section["sections"], processed_section)
        processed_sections.append(processed_section)

    def process_note_willie_spacy(note):
        note_sections = sent_tokenize_rules(note)
        processed_sections = []
        section_frame = pd.DataFrame({"sections": note_sections})
        section_frame.apply(
            process_sections_helper,
            args=(processed_sections,),
            axis=1,
        )
        return processed_sections

    def text_process(sent, note):
        sent_text = sent["sents"].text
        if len(sent_text) > 0 and sent_text.strip() != "\n":
            if "\n" in sent_text:
                sent_text = sent_text.replace("\n", " ")
            note["text"] += sent_text + "\n"

    def get_sentences(processed_section, note):
        sent_frame = pd.DataFrame({"sents": list(processed_section["sections"].sents)})
        sent_frame.apply(text_process, args=(note,), axis=1)

    def process_frame_text(note):
        try:
            note_text = str(note["text"])
            note["text"] = ""
            processed_sections = process_note_willie_spacy(note_text)
            ps = {"sections": processed_sections}
            ps = pd.DataFrame(ps)

            ps.apply(get_sentences, args=(note,), axis=1)

            return note
        except Exception as e:
            print("error", e, file=sys.stderr)
            return note

    notes = notes.apply(process_frame_text, axis=1)

    if notes_h5_filepath is not None:
        notes.to_hdf(notes_h5_filepath, "notes")
    return notes


def save_icd9_codes(codes, codes_h5_filepath):
    codes.set_index(ID_COLS, inplace=True)
    codes.to_hdf(codes_h5_filepath, "C")
    return codes


def save_outcome(
    data,
    querier,
    out_path,
    filenames,
    outcome_schema,
    verbose=True,
):
    """Retrieve outcomes from DB and save to disk

    Vent and vaso are both there already - so pull the start and stop times from there! :)

    Returns
    -------
    Y : Pandas dataframe
        Obeys the outcomes data spec
    """
    icuids_to_keep = get_values_by_name_from_df_column_or_index(data, "icustay_id")
    icuids_to_keep = {str(s) for s in icuids_to_keep}

    # Add a new column called intime so that we can easily subtract it off
    data = data.reset_index()
    data = data.set_index("icustay_id")
    data["intime"] = pd.to_datetime(data["intime"])  # , format="%m/%d/%Y"))
    data["outtime"] = pd.to_datetime(data["outtime"])
    icustay_timediff_tmp = data["outtime"] - data["intime"]
    icustay_timediff = pd.Series(
        [
            timediff.days * 24 + timediff.seconds // 3600
            for timediff in icustay_timediff_tmp
        ],
        index=data.index.values,
    )
    query = """
    select i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
    FROM icustay_detail i
    INNER JOIN ventilation_durations v ON i.icustay_id = v.icustay_id
    where v.icustay_id in ({icuids})
    and v.starttime between intime and outtime
    and v.endtime between intime and outtime;
    """

    old_template_vars = querier.exclusion_criteria_template_vars
    querier.exclusion_criteria_template_vars = dict(icuids=",".join(icuids_to_keep))

    vent_data = querier.query(query_string=query)
    vent_data = continuous_outcome_processing(vent_data, data, icustay_timediff)
    vent_data = vent_data.apply(add_outcome_indicators)
    vent_data.rename(columns={"on": "vent"}, inplace=True)
    vent_data = vent_data.reset_index()

    # Get the patients without the intervention in there too so that we...
    ids_with = vent_data["icustay_id"]
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = ids_all - ids_with

    # Create a new fake dataframe with blanks on all vent entries
    out_data = data.copy(deep=True)
    out_data = out_data.reset_index()
    out_data = out_data.set_index("icustay_id")
    out_data = out_data.iloc[out_data.index.isin(ids_without)]
    out_data = out_data.reset_index()
    out_data = out_data[["subject_id", "hadm_id", "icustay_id"]]
    out_data["max_hours"] = out_data["icustay_id"].map(icustay_timediff)

    # Create all 0 column for vent
    out_data = out_data.groupby("icustay_id")
    out_data = out_data.apply(add_blank_indicators)
    out_data.rename(columns={"on": "vent"}, inplace=True)
    out_data = out_data.reset_index()

    # Concatenate all the data vertically
    Y = pd.concat(
        [
            vent_data[["subject_id", "hadm_id", "icustay_id", "hours_in", "vent"]],
            out_data[["subject_id", "hadm_id", "icustay_id", "hours_in", "vent"]],
        ],
        axis=0,
    )

    # Start merging all other interventions
    table_names = [
        "vasopressor_durations",
        "adenosine_durations",
        "dobutamine_durations",
        "dopamine_durations",
        "epinephrine_durations",
        "isuprel_durations",
        "milrinone_durations",
        "norepinephrine_durations",
        "phenylephrine_durations",
        "vasopressin_durations",
    ]
    column_names = [
        "vaso",
        "adenosine",
        "dobutamine",
        "dopamine",
        "epinephrine",
        "isuprel",
        "milrinone",
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
    ]

    # TODO(mmd): This section doesn't work. What is its purpose?
    for t, c in zip(table_names, column_names):
        # TOTAL VASOPRESSOR DATA
        query = """
        select i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
        FROM icustay_detail i
        INNER JOIN {table} v ON i.icustay_id = v.icustay_id
        where v.icustay_id in ({icuids})
        and v.starttime between intime and outtime
        and v.endtime between intime and outtime;
        """
        new_data = querier.query(query_string=query, extra_template_vars=dict(table=t))
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns={"on": c}, inplace=True)
        new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population,
        # in which c was never performed.
        if not c in new_data:
            print("Column ", c, " not in data.", file=sys.stderr)
            continue

        Y = Y.merge(
            new_data[["subject_id", "hadm_id", "icustay_id", "hours_in", c]],
            on=["subject_id", "hadm_id", "icustay_id", "hours_in"],
            how="left",
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[c] = Y[c].astype(int)
        # Y = Y.sort_values(['subject_id', 'icustay_id', 'hours_in']) #.merge(df3,on='name')
        Y = Y.reset_index(drop=True)
        if verbose:
            print("Extracted " + c + " from " + t)

    tasks = ["colloid_bolus", "crystalloid_bolus", "nivdurations"]

    for task in tasks:
        if task == "nivdurations":
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.starttime, v.endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.starttime between intime and outtime
            and v.endtime between intime and outtime;
            """
        else:
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.charttime AS starttime,
                   v.charttime AS endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.charttime between intime and outtime
            """

        new_data = querier.query(
            query_string=query, extra_template_vars=dict(table=task)
        )
        if new_data.shape[0] == 0:
            continue
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns={"on": task}, inplace=True)
        new_data = new_data.reset_index()
        Y = Y.merge(
            new_data[["subject_id", "hadm_id", "icustay_id", "hours_in", task]],
            on=["subject_id", "hadm_id", "icustay_id", "hours_in"],
            how="left",
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[task] = Y[task].astype(int)
        Y = Y.reset_index(drop=True)
        if verbose:
            print("Extracted " + task)

    querier.exclusion_criteria_template_vars = old_template_vars

    Y = Y.filter(
        items=["subject_id", "hadm_id", "icustay_id", "hours_in", "vent"]
        + column_names
        + tasks
    )
    Y.subject_id = Y.subject_id.astype(int)
    Y.icustay_id = Y.icustay_id.astype(int)
    Y.hours_in = Y.hours_in.astype(int)
    Y.vent = Y.vent.astype(int)
    Y.vaso = Y.vaso.astype(int)
    y_id_cols = ID_COLS + ["hours_in"]
    Y = Y.sort_values(y_id_cols)
    Y.set_index(y_id_cols, inplace=True)

    if verbose:
        print("Shape of Y : ", Y.shape)

    # Turn back into columns
    df = Y.reset_index()
    df = sanitize_df(df, outcome_schema)
    csv_fpath = os.path.join(out_path, filenames["outcome"])
    save_sanitized_df_to_csv(csv_fpath, df, outcome_schema)

    col_names = list(df.columns.values)
    col_names = col_names[3:]
    with open(
        os.path.join(out_path, filenames["outcome_columns"]), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(col_names))

    Y.to_hdf(os.path.join(out_path, filenames["outcome_hd5"]), "Y")
    return df


def apply_variable_limits(df, var_ranges, var_names_index_col="LEVEL2", verbose=True):
    idx_vals = df.index.get_level_values(var_names_index_col)
    non_null_idx = ~df.value.isnull()
    var_names = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    for var_name in var_names:
        var_name_lower = var_name.lower()
        if var_name_lower not in var_range_names:
            if verbose:
                print(f"No known ranges for {var_name}")
            continue

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x]
            for x in ("OUTLIER_LOW", "OUTLIER_HIGH", "VALID_LOW", "VALID_HIGH")
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx = df.value < outlier_low_val
        outlier_high_idx = df.value > outlier_high_val
        valid_low_idx = ~outlier_low_idx & (df.value < valid_low_val)
        valid_high_idx = ~outlier_high_idx & (df.value > valid_high_val)

        var_outlier_idx = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        df.loc[var_outlier_idx, "value"] = np.nan
        df.loc[var_valid_low_idx, "value"] = valid_low_val
        df.loc[var_valid_high_idx, "value"] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0 and verbose:
            print(
                f"{var_name} had {n_outlier + n_valid_low + n_valid_high} / "
                f"{sum(running_idx)} rows cleaned:\n"
                f"  {n_outlier} rows were strict outliers, set to np.nan\n"
                f"  {n_valid_low} rows were low valid outliers, set to "
                f"{valid_low_val:.2f}\n"
                f"  {n_valid_high} rows were high valid outliers, set to "
                f"{valid_high_val:.2f}\n"
            )

    return df


def plot_variable_histograms(df, out_path):
    # Plot some of the data, just to make sure it looks ok
    for c, vals in df.iteritems():
        n = vals.dropna().count()
        if n < 2:
            continue

        # get median, variance, skewness
        med = vals.dropna().median()
        var = vals.dropna().var()
        skew = vals.dropna().skew()

        # plot
        fig = plt.figure(figsize=(13, 6))
        plt.subplots(figsize=(13, 6))
        vals.dropna().plot.hist(bins=100, label=f"HIST (n={n})")

        # fake plots for KS test, median, etc
        plt.plot([], label=" ", color="lightgray")
        plt.plot([], label=f"Median: {med:.2f}", color="lightgray")
        plt.plot([], label=f"Variance: {var:.2f}", color="lightgray")
        plt.plot([], label=f"Skew: {skew:.2f}", color="light:gray")

        # add title, labels etc.
        plt.title(f"{c} measurements in ICU")
        plt.xlabel(str(c))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        plt.xlim(0, vals.quantile(0.99))
        fig.savefig(
            os.path.join(out_path, (str(c) + "_HIST_.png")), bbox_inches="tight"
        )


def get_args():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_path",
        type=str,
        default=f"/scratch/{os.environ['USER']}/phys_acuity_modelling/data",
        help="Enter the path you want the output",
    )
    ap.add_argument(
        "--resource_path",
        type=str,
        default=os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR/resources/"),
    )
    ap.add_argument(
        "--queries_path",
        type=str,
        default=os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR/SQL_Queries/"),
    )
    ap.add_argument(
        "--extract_pop",
        type=int,
        default=1,
        help="Whether or not to extract population data: 0 - no extraction, "
        + "1 - extract if not present in the data directory, "
        + "2 - extract even if there is data",
    )

    ap.add_argument(
        "--extract_numerics",
        type=int,
        default=1,
        help="Whether or not to extract numerics data: 0 - no extraction, "
        + "1 - extract if not present in the data directory, "
        + "2 - extract even if there is data",
    )
    ap.add_argument(
        "--extract_outcomes",
        type=int,
        default=1,
        help="Whether or not to extract outcome data: 0 - no extraction, "
        + "1 - extract if not present in the data directory, "
        + "2 - extract even if there is data",
    )
    ap.add_argument(
        "--extract_codes",
        type=int,
        default=1,
        help="Whether or not to extract ICD9 codes: 0 - no extraction, "
        + "1 - extract if not present in the data directory, "
        + "2 - extract even if there is data",
    )
    ap.add_argument(
        "--extract_notes",
        type=int,
        default=1,
        help="Whether or not to extract notes: 0 - no extraction, "
        + "1 - extract if not present in the data directory, "
        + "2 - extract even if there is data",
    )
    ap.add_argument(
        "--pop_size", type=int, default=0, help="Size of population to extract"
    )
    ap.add_argument("--exit_after_loading", type=int, default=0)
    ap.add_argument(
        "--var_limits",
        type=int,
        default=1,
        help="Whether to create a version of the data with variable limits included. "
        + "1 - apply variable limits, 0 - do not apply variable limits",
    )
    ap.add_argument(
        "--plot_hist",
        type=int,
        default=1,
        help="Whether to plot the histograms of the data",
    )

    ap.add_argument(
        "--psql_host",
        type=str,
        default=None,
        help="Postgres host; use Unix socket if not specified. Try "
        + '"/var/run/postgresql/" for Unix domain socket errors.',
    )
    ap.add_argument(
        "--psql_port",
        type=int,
        default=None,
        help="Postgres port. Defaults to 5432 if not provided.",
    )
    ap.add_argument(
        "--psql_dbname", type=str, default="mimic", help="Postgres database name."
    )
    ap.add_argument(
        "--psql_schema_name",
        type=str,
        default="mimiciii",
        help="Postgres database name.",
    )
    ap.add_argument("--psql_user", type=str, default=None, help="Postgres user.")
    ap.add_argument(
        "--psql_password", type=str, default=None, help="Postgres password."
    )
    ap.add_argument(
        "--no_group_by_level2",
        action="store_false",
        dest="group_by_level2",
        default=True,
        help="Don't group by level2.",
    )

    ap.add_argument(
        "--min_percent",
        type=float,
        default=0.0,
        help="Minimum percentage of row numbers need to be observations for "
        + "each numeric column. "
        + "min_percent = 1 means columns with more than 99 percent of nan "
        + "will be removed. "
        + "Note that as our code does not split the data into train/test sets, "
        + "removing columns in this way prior to train/test splitting yields "
        + "in a (very minor) "
        + "form of leakage across the train/test set, as the overall "
        + "missingness measures are used "
        + "that are based on both the train and test sets, rather than just "
        + "the train set.",
    )
    ap.add_argument(
        "--min_age", type=int, default=15, help="Minimum age of patients to be included"
    )
    ap.add_argument(
        "--min_duration",
        type=int,
        default=12,
        help="Minimum hours of stay to be included",
    )
    ap.add_argument(
        "--max_duration",
        type=int,
        default=240,
        help="Maximum hours of stay to be included",
    )

    args = vars(ap.parse_args())

    if not isdir(args["resource_path"]):
        raise ValueError(
            f"Invalid resource_path: {args['resource_path']}; "
            + "you may have to set MIMIC_EXTRACT_CODE_DIR"
        )

    if not isdir(args["out_path"]):
        raise ValueError(f"out_path: {args['out_path']} does not exist")

    return args


class Extraction(enum.Enum):
    RELOAD = enum.auto()
    EXTRACT = enum.auto()
    NOTHING = enum.auto()


def extraction_required(arg, path):
    if arg in [0, 1] and isfile(path):
        return Extraction.RELOAD
    if (arg == 1 and not isfile(path)) or arg == 2:
        return Extraction.EXTRACT
    return Extraction.NOTHING


def extract_population(args, filenames, querier, verbose):
    # Load specs for output tables
    static_data_schema = load_datapackage_schema(
        os.path.join(args["resource_path"], "static_data_spec.json")
    )

    data = None
    static_path = os.path.join(args["out_path"], filenames["static"])
    action = extraction_required(args["extract_pop"], static_path)

    if action == Extraction.RELOAD:
        if verbose:
            print(f"Reloading data from {static_path}")
        data = pd.read_csv(static_path)
        data = sanitize_df(data, static_data_schema)
    elif action == Extraction.EXTRACT:
        if verbose:
            print("Building data from scratch.")
        pop_size_string = ""
        if args["pop_size"] > 0:
            pop_size_string = f"LIMIT {args['pop_size']}"

        min_age_string = str(args["min_age"])
        min_dur_string = str(args["min_duration"])
        max_dur_string = str(args["max_duration"])
        min_day_string = str(args["min_duration"] / 24)

        template_vars = dict(
            limit=pop_size_string,
            min_age=min_age_string,
            min_dur=min_dur_string,
            max_dur=max_dur_string,
            min_day=min_day_string,
        )

        data_df = querier.query(
            query_file=STATICS_QUERY_PATH, extra_template_vars=template_vars
        )
        data_df = sanitize_df(data_df, static_data_schema)

        if verbose:
            print(f"Storing data @ {static_path}")
        data = save_pop(data_df, static_path, static_data_schema)

    if data is None:
        if verbose:
            print("SKIPPED static_data")
    else:
        # So all subsequent queries will limit to just that already extracted in data_df.
        querier.add_exclusion_criteria_from_df(data, columns=["hadm_id", "subject_id"])
        if verbose:
            print("loaded static_data")

    return data


def extract_numerics(args, filenames, data, query_args, schema_name, verbose):
    X = None
    dynamic_hd5_path = os.path.join(args["out_path"], filenames["dynamic_hd5"])
    action = extraction_required(args["extract_numerics"], dynamic_hd5_path)

    if action == Extraction.RELOAD:
        if verbose:
            print(f"Reloading X from {dynamic_hd5_path}")
        X = pd.read_hdf(dynamic_hd5_path)
    elif action == Extraction.EXTRACT:
        if verbose:
            print("Extracting vitals data...")
        start_time = time.time()

        ########
        # Step 1) Get the set of variables we want for the patients we've identified!
        icuids_to_keep = get_values_by_name_from_df_column_or_index(data, "icustay_id")
        icuids_to_keep = {str(s) for s in icuids_to_keep}
        data = data.reset_index().set_index("icustay_id")

        # Select out SID, TIME, ITEMID, VALUE form each of the sources!
        var_map = get_variable_mapping(filenames["mimic_mapping"])
        var_ranges = get_variable_ranges(filenames["range"])

        chartitems_to_keep = var_map.loc[var_map["LINKSTO"] == "chartevents"].ITEMID
        chartitems_to_keep = {str(i) for i in chartitems_to_keep}

        labitems_to_keep = var_map.loc[var_map["LINKSTO"] == "labevents"].ITEMID
        labitems_to_keep = {str(i) for i in labitems_to_keep}

        con = psycopg2.connect(**query_args)
        cur = con.cursor()

        if verbose:
            print(f"  starting db query with {len(icuids_to_keep)} subjects...")
        cur.execute("SET search_path to " + schema_name)
        query = """
        select c.subject_id, i.hadm_id, c.icustay_id, c.charttime, c.itemid,
          c.value, valueuom
        FROM icustay_detail i
        INNER JOIN chartevents c ON i.icustay_id = c.icustay_id
        where c.icustay_id in ({icuids})
          and c.itemid in ({chitem})
          and c.charttime between intime and outtime
          and c.error is distinct from 1
          and c.valuenum is not null

        UNION ALL

        select distinct i.subject_id, i.hadm_id, i.icustay_id, l.charttime,
          l.itemid, l.value, valueuom
        FROM icustay_detail i
        INNER JOIN labevents l ON i.hadm_id = l.hadm_id
        where i.icustay_id in ({icuids})
          and l.itemid in ({lbitem})
          and l.charttime between (intime - interval '6' hour) and outtime
          and l.valuenum > 0 -- lab values cannot be 0 and cannot be negative
        ;
        """.format(
            icuids=",".join(icuids_to_keep),
            chitem=",".join(chartitems_to_keep),
            lbitem=",".join(labitems_to_keep),
        )
        X = pd.read_sql_query(query, con)

        itemids = set(X.itemid.astype(str))  # pylint: disable=no-member

        query_d_items = """
        SELECT itemid, label, dbsource, linksto, category, unitname
        FROM d_items
        WHERE itemid in ({itemid_list})
        ;
        """.format(
            itemid_list=",".join(itemids)
        )
        I = pd.read_sql_query(query_d_items, con).set_index("itemid")

        cur.close()
        con.close()
        if verbose:
            print(f"  db query finished after {time.time() - start_time:.3f} sec")
        X = process_and_save_numerics(
            data,
            X,
            I,
            var_map,
            var_ranges,
            args["out_path"],
            filenames,
            group_by_level2=args["group_by_level2"],
            apply_var_limit=args["var_limits"],
            min_percent=args["min_percent"],
        )

    if verbose:
        if X is None:
            print("SKIPPED vitals_hourly_data")
        else:
            print("LOADED vitals_hourly_data")

    return X, data


def extract_codes(args, filenames, querier, verbose):
    C = None
    codes_hd5_path = os.path.join(args["out_path"], filenames["codes_hd5"])
    action = extraction_required(args["extract_codes"], codes_hd5_path)

    if action == Extraction.RELOAD:
        if verbose:
            print(f"Reloading codes from {codes_hd5_path}")
        C = pd.read_hdf(codes_hd5_path)
    elif action == Extraction.EXTRACT:
        if verbose:
            print("Saving codes...")
        codes = querier.query(query_file=CODES_QUERY_PATH)
        C = save_icd9_codes(codes, codes_hd5_path)

    if verbose:
        if C is None:
            print("SKIPPED codes_data")
        else:
            print("LOADED codes_data")

    return C


def extract_notes(args, filenames, querier, verbose):
    N = None
    notes_hd5_path = os.path.join(args["out_path"], filenames["notes_hd5"])
    action = extraction_required(args["extract_notes"], notes_hd5_path)

    if action == Extraction.RELOAD:
        if verbose:
            print("Reloading notes from {notes_hd5_path}.")
        N = pd.read_hdf(notes_hd5_path)
    elif action == Extraction.EXTRACT:
        if verbose:
            print("Saving notes...")
        notes = querier.query(query_file=NOTES_QUERY_PATH)
        N = save_notes(notes, notes_hd5_path)

    if verbose:
        if N is None:
            print("SKIPPED notes_data")
        else:
            print("LOADED notes_data")

    return N


def extract_outcomes(args, filenames, data, querier, verbose):
    Y = None
    outcome_hd5_path = os.path.join(args["out_path"], filenames["outcome_hd5"])
    action = extraction_required(args["extract_outcomes"], outcome_hd5_path)

    if action == Extraction.RELOAD:
        if verbose:
            print("Reloading outcomes from {outcome_hd5_path}")
        Y = pd.read_hdf(outcome_hd5_path)

    elif action == Extraction.EXTRACT:
        if verbose:
            print("Saving Outcomes...")
        outcome_data_schema = load_datapackage_schema(
            os.path.join(args["resource_path"], "outcome_data_spec.json")
        )
        Y = save_outcome(
            data,
            querier,
            args["out_path"],
            filenames,
            outcome_data_schema,
        )

    return Y


def main(verbose=True):
    if verbose:
        print("Running!")

    args = get_args()
    if verbose:
        for key in sorted(args.keys()):
            print(key, args[key])
    if args["psql_host"] == "SOCKET":
        args["psql_host"] = None

    # Modify the filenames
    def insert_pop_size(filename):
        if args["pop_size"] > 0:
            fn_parts = splitext(filename)
            return fn_parts[0] + f"_{args['pop_size']}" + fn_parts[1]
        else:
            return filename

    filenames = {}
    filenames["static"] = insert_pop_size(STATIC_FILENAME)
    filenames["dynamic"] = insert_pop_size(DYNAMIC_FILENAME)
    filenames["subjects"] = insert_pop_size(SUBJECTS_FILENAME)
    filenames["times"] = insert_pop_size(TIMES_FILENAME)
    filenames["dynamic_hd5"] = insert_pop_size(DYNAMIC_HD5_FILENAME)
    filenames["outcome"] = insert_pop_size(OUTCOME_FILENAME)
    filenames["dynamic_hd5_filt"] = insert_pop_size(DYNAMIC_HD5_FILT_FILENAME)
    filenames["outcome_hd5"] = insert_pop_size(OUTCOME_HD5_FILENAME)
    filenames["codes_hd5"] = insert_pop_size(CODES_HD5_FILENAME)
    filenames["notes_hd5"] = insert_pop_size(NOTES_HD5_FILENAME)
    filenames["columns"] = COLUMNS_FILENAME
    filenames["outcome_columns"] = OUTCOME_COLUMNS_FILENAME

    filenames["mimic_mapping"] = os.path.join(
        args["resource_path"], "itemid_to_variable_map.csv"
    )
    filenames["range"] = os.path.join(args["resource_path"], "variable_ranges.csv")

    schema_name = "public," + args["psql_schema_name"]

    query_args = {"dbname": args["psql_dbname"]}
    for conn_param in ["host", "port", "user", "password"]:
        if args[f"psql_{conn_param}"] is not None:
            query_args[conn_param] = args[f"psql_{conn_param}"]

    querier = MIMIC_Querier(query_args=query_args, schema_name=schema_name)

    data = extract_population(args, filenames, querier, verbose)
    X, data = extract_numerics(args, filenames, data, query_args, schema_name, verbose)
    C = extract_codes(args, filenames, querier, verbose)
    N = extract_notes(args, filenames, querier, verbose)
    Y = extract_outcomes(args, filenames, data, querier, verbose)

    if verbose:
        if X is not None:
            print("Numerics", X.shape, X.index.names, X.columns.names, X.columns)
        if Y is not None:
            print("Outcomes", Y.shape, Y.index.names, Y.columns.names, Y.columns)
        if C is not None:
            print("Codes", C.shape, C.index.names, C.columns.names, C.columns)
        if N is not None:
            print("Notes", N.shape, N.index.names, N.columns.names, N.columns)

        print("Data", data.shape, data.index.names, data.columns.names, data.columns)

    if args["exit_after_loading"]:
        return

    shared_idx = X.index
    shared_sub = list(X.index.get_level_values("icustay_id").unique())
    # TODO(mmd): Why does this work?
    Y = Y.loc[shared_idx]
    # Problems start here.
    if C is not None:
        C = C.loc[shared_idx]
    data = data[  # pylint: disable=unsubscriptable-object
        data.index.get_level_values("icustay_id").isin(set(shared_sub))
    ]
    data = data.reset_index().set_index(ID_COLS)

    # Map the lowering function to all column names
    X.columns = pd.MultiIndex.from_tuples(
        [tuple(str(l).lower() for l in cols) for cols in X.columns],
        names=X.columns.names,
    )
    if args["group_by_level2"]:
        var_names = list(X.columns.get_level_values("LEVEL2"))
    else:
        var_names = list(X.columns.get_level_values("itemid"))

    Y.columns = Y.columns.str.lower()
    out_names = list(Y.columns.values[3:])
    if C is not None:
        C.columns = C.columns.str.lower()
        icd_names = list(C.columns.values[1:])
    data.columns = data.columns.str.lower()
    static_names = list(data.columns.values[3:])

    if verbose:
        print("Shape of X : ", X.shape)
        print("Shape of Y : ", Y.shape)
        if C is not None:
            print("Shape of C : ", C.shape)
        print("Shape of static : ", data.shape)
        print("Variable names : ", ",".join(var_names))
        print("Output names : ", ",".join(out_names))
        if C is not None:
            print("Ic_dfD9 names : ", ",".join(icd_names))
        print("Static data : ", ",".join(static_names))

    dynamic_hd5_filt_path = os.path.join(
        args["out_path"], filenames["dynamic_hd5_filt"]
    )
    X.to_hdf(dynamic_hd5_filt_path, "vitals_labs")
    Y.to_hdf(dynamic_hd5_filt_path, "interventions")
    if C is not None:
        C.to_hdf(dynamic_hd5_filt_path, "codes")
    data.to_hdf(dynamic_hd5_filt_path, "patients", format="table")

    X_mean = X.iloc[:, X.columns.get_level_values(-1) == "mean"]
    X_mean.to_hdf(dynamic_hd5_filt_path, "vitals_labs_mean")

    if args["plot_hist"] == 1:
        plot_variable_histograms(X, args["out_path"])

    if verbose:
        # Print the total proportions
        rows = X.shape[0]
        print()
        for l, vals in X.iteritems():
            ratio = 1.0 * vals.dropna().count() / rows
            print(f"{l}: {(100 * ratio):.1f}% present")
        print()

        # Print the per subject proportions
        df = X.groupby(["subject_id"]).count()
        for k in [1, 2, 3]:
            print(f"% of subjects had at least {k} present")
            d = df > k
            d = d.sum(axis=0)
            d = d / len(df)
            d = d.reset_index()
            for index, row in d.iterrows():
                print(f"{index}: {(100 * row[0]):.1f}%")
            print()

    if verbose:
        print("Done!")


if __name__ == "__main__":
    main(verbose=True)
