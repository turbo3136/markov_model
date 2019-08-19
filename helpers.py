from datetime import datetime, date
from dateutil.relativedelta import relativedelta


def date_string_to_datetime(date_string):
    date_object = datetime.strptime(date_string, '%Y-%m-%d')
    return date(date_object.year, date_object.month, date_object.day)


def add_interval_to_date(date_object, steps, interval):
    years = 0
    months = 0
    days = 0

    if interval == 'year':
        years = steps
    elif interval == 'month':
        months = steps
    elif interval == 'days':
        days = steps
    else:
        raise ValueError('expected one of year, month, or day, got {}'.format(interval))

    return date_object + relativedelta(years=+years, months=+months, days=+days)
