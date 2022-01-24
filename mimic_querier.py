import copy

import psycopg2
import pandas as pd  # type: ignore


# TODO(mmd): Where should this go?
# TODO(mmd): Rename
# TODO(mmd): eliminate try/except. Just use conditionals.
def get_values_by_name_from_df_column_or_index(data_df, colname):
    """ Easily get values for named field, whether a column or an index

    Returns
    -------
    values : 1D array
    """
    try:
        values = data_df[colname]
    except KeyError as e:
        if colname in data_df.index.names:
            values = data_df.index.get_level_values(colname)
        else:
            raise e
    return values


# TODO(mmd): Maybe make context manager?
class MIMIC_Querier():
    def __init__(
        self,
        exclusion_criteria_template_vars=None,
        query_args=None,  # passed wholesale to psycopg2.connect
        schema_name='mimiciii'
    ):
        """ A class to facilitate repeated Queries to a MIMIC psql database """
        if exclusion_criteria_template_vars is None:
            self.exclusion_criteria_template_vars = {}
        else:
            self.exclusion_criteria_template_vars = exclusion_criteria_template_vars
        if query_args is None:
            self.query_args = {}
        else:
            self.query_args = query_args
        self.schema_name = schema_name
        self.connected = False
        self.connection, self.cursor = None, None

    # TODO(mmd): this isn't really doing exclusion criteria.
    # Should maybe also absorb 'WHERE' clause...
    def add_exclusion_criteria_from_df(self, df, columns=[]):
        self.exclusion_criteria_template_vars.update({
            c: "','".join(
                set([str(v) for v in get_values_by_name_from_df_column_or_index(df, c)])
            ) for c in columns
        })

    def clear_exclusion_criteria(self):
        self.exclusion_criteria_template_vars = {}

    def close(self):
        if not self.connected:
            return
        self.connection.close()
        self.cursor.close()  # TODO(mmd): Maybe don't actually need this to stay open?
        self.connected = False

    def connect(self):
        self.close()
        self.connection = psycopg2.connect(**self.query_args)
        self.cursor = self.connection.cursor()
        self.cursor.execute('SET search_path TO %s' % self.schema_name)
        self.connected = True

    def query(self, query_string=None, query_file=None, extra_template_vars=None):
        number_of_queries = (query_string is not None) + (query_file is not None)
        assert number_of_queries > 0, "Must pass a query!"
        assert number_of_queries == 1, "Must only pass one query!"

        self.connect()

        if query_string is None:
            with open(query_file, mode='r') as f:
                query_string = f.read()

        template_vars = copy.copy(self.exclusion_criteria_template_vars)
        if extra_template_vars is not None:
            template_vars.update(extra_template_vars)

        query_string = query_string.format(**template_vars)
        out = pd.read_sql_query(query_string, self.connection)

        self.close()
        return out
