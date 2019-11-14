from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd


def date_string_to_datetime(date_string):
    date_object = datetime.strptime(date_string, '%Y-%m-%d')
    return date(date_object.year, date_object.month, date_object.day)


def add_interval_to_date(date_object, steps, interval):
    years = 0
    months = 0
    days = 0

    if interval == 'year':
        years = int(steps)
    elif interval == 'month':
        months = int(steps)
    elif interval == 'day':
        days = int(steps)
    else:
        raise ValueError('expected one of year, month, or day, got {}'.format(interval))

    return date_object + relativedelta(years=+years, months=+months, days=+days)


def join_vector_to_df_on_index_and_multiply_across_rows(vector, df, multiply_column):
    """join a vector to a df on the index and multiply across the row (the first step in a dot product)

    :param vector: pandas dataframe with the column you want to multiply the df by
    :param df: pandas dataframe
    :param multiply_column: column of the vector you want to multiply by
    :return: dataframe the shape of the original df, multiplied by the vector
    """
    column_names = df.columns.values
    ret = df.merge(vector[multiply_column], left_index=True, right_index=True)

    # this is where the dot multiplication happens, we multiply the distribution column to all the states
    ret[column_names] = ret.apply(
        lambda row: pd.Series([x * row[multiply_column] for x in row[column_names]]),
        axis=1,
    )

    return ret[column_names]
