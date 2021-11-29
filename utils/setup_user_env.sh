#!/bin/bash

export MIMIC_CODE_DIR=$(realpath ../../mimic-code)
export MIMIC_EXTRACT_CODE_DIR=$(realpath ../)

export MIMIC_DATA_DIR=$MIMIC_EXTRACT_CODE_DIR/data/

export MIMIC_EXTRACT_OUTPUT_DIR=$MIMIC_DATA_DIR/curated/
mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR

export DBUSER=mimic
export DBNAME=mimic
export DBPASSWORD=mimic
export SCHEMA=mimiciii
export HOST=SOCKET
export PORT=5432

if [ $HOST = SOCKET ]
then
    export DBSTRING="port=$PORT user=$DBUSER password=$DBPASSWORD dbname=$DBNAME options=--search_path=$SCHEMA"
else
    export DBSTRING="host=$HOST port=$PORT user=$DBUSER password=$DBPASSWORD dbname=$DBNAME options=--search_path=$SCHEMA"
fi
