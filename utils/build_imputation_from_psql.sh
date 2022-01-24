#!/bin/bash
#
# Build curated raw imputation dataset for MIMIC-III patient data

mkdir -p "$MIMIC_IMPUTATION_OUTPUT_DIR"

"$MIMIC_EXTRACT_CODE_DIR"/mimic_imputation_extract.py \
    --outdir "$MIMIC_IMPUTATION_OUTPUT_DIR" \
    --reload_population true \
    --reload_clinical true \
    --psql_user $DBUSER \
    --psql_password "$DBPASSWORD" \
    --psql_host $HOST \
    --psql_port $PORT \
    --loglevel INFO \
    "$@"
