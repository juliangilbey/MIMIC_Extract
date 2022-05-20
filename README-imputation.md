# Extracting data for imputation experiments from MIMIC-III

This is a fork of the **MIMIC-Extract**
[repository](https://github.com/MLforHealth/MIMIC_Extract) for our
imputation experiments.  This README explains how to use this fork to
regenerate our data.

1. Download the MIMIC-III dataset; this requires authentication.  To
   do so, follow the steps on [mimic.mit.edu](https://mimic.mit.edu/).
   We store the gzipped dataset in a directory, say
   `MIMIC_III_GZIPPED_DIR`, containing `ADMISSIONS.csv.gz` etc.

2. Set up a suitable PostgreSQL database.  Below, I assume that this
   is set up as a separate cluster called `mimic` running at port 5433.

3. Clone the
   [https://github.com/MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code)
   repository, go into the directory `mimic-iii/buildmimic/postgres/`
   and run the commands:

    * `make create-user DBUSER=mimic DBPASS=mimic DBPORT=5433`

        This will create a user called `mimic` to manage the database.
        (The default is to use `postgres`, but I prefer to have a
        dedicated user in a dedicated database, given that the
        password is passed in plaintext.)  Depending on your machine
        setup, you might be requested to sudo to the postgres account
        to set up this new user.

    * `make mimic-gz datadir=MIMIC_III_GZIPPED_DIR DBHOST=localhost`
        `DBPORT=5433 DBUSER=mimic DBPASS=mimic`
       
         Note that this code is all on one line, and
         `MIMIC_III_GZIPPED_DIR` should be replaced by the actual path
         to the data directory.  This will take a long time (several
         hours, most likely).
    
4. Next, go into the `mimic-iii/concepts` directory and run the
   following commands, with the host, port and so on modified as
   appropriate; each of these commands is one line.  The second one
   will likely take a long time to run.

    * `psql "host=localhost port=5433 dbname=mimic user=mimic password=mimic`
        `options=--search_path=mimiciii" -f postgres-functions.sql`

    * `DBCONNEXTRA="host=localhost port=5433 user=mimic password=mimic"`
        `bash postgres_make_concepts.sh`

5. Clone this repository
    [https://github.com/juliangilbey/MIMIC_Extract.git](https://github.com/juliangilbey/MIMIC_Extract.git)
    and switch to the branch `imputation-data`.  In the directory
    `MIMIC_Extract` do the following:
    
    a. Create a file `utils/setup_user_env_local.sh` containing the
        following lines, if necessary; these will override the defaults in
        `utils/setup_user_env.sh` (and this file is ignored by git):

            # this is the directory where extracted data will be saved
            # if the original mimic_direct_extract.py is used
            export MIMIC_EXTRACT_OUTPUT_DIR=...
    
            # this is the directory where extracted data will be saved
            # when extracting imputation data
            export MIMIC_IMPUTATION_OUTPUT_DIR=...
    
            # the following should be given appropriate values
            DBUSER=...
            DBPASSWORD=...
            HOST=...
            PORT=...

    b. Install all of the required Python packages.  The list of
        requirements is as follows:

        * `pandas`
        * `numpy`
        * `datapackage`
        * `psycopg2`
        * `typed-argument-parser`

    c. Go into utils and run the following, all on one line:

        `MIMIC_CODE_DIR=../../mimic-code/mimic-iii`
        `DBCONNEXTRA="host=localhost port=5433 user=mimic password=mimic"`
        `bash postgres_make_extended_concepts.sh`
        
        In this command, `MIMIC_CODE_DIR` points to the
        `mimic-code/mimic-iii` repository; this assumes that the
        `MIMIC_Extract` and `mimic-code` repositories are stored
        side-by-side, but the path to `mimic-iii` may need changing
        depending on your setup.

    d. Run the following (on one line as usual):

        `psql "host=localhost port=5433 user=mimic password=mimic dbname=mimic"`
        `-f niv-durations.sql`

    e. Run the following:
    
        `make build_curated_from_psql`

        It will take a fairly long time.

The resulting files are:

* `clinical_data.csv`: the raw clinical data extracted from the
  database
* `condensed_summary.csv`: patient-by-patient clinical data summaries
* `population_data.csv`: patient data including survival outcome
* `summary_data.csv`: a slightly more detailed version of
  `condensed_summary.csv` including the number of readings for each
  measurement type

Only the `condensed_summary.csv` file is required for the imputation
experiments.
