# coding=UTF-8
'''
@Author: cuiqiyuan
@Date: 2019-11-19 15:48:23
@Description: file content
'''
from datetime import datetime


def cale_to_jd(time):
    fmt = '%Y%m%d'
    dt = datetime.strptime(time, fmt)
    tt = dt.timetuple()
    return tt.tm_year * 1000 + tt.tm_yday


def jd_to_cale(time):
    dt = datetime.strptime(time, '%Y%j').date()
    fmt = '%Y.%m.%d'
    return dt.strftime(fmt)

if __name__ == '__main__':
    print(cale_to_jd('20191208'))
    