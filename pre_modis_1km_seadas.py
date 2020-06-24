'''
@Author: Marvin
@Date: 2020-06-18 15:09:38
@LastEditTime: 2020-06-18 16:20:58
@LastEditors: Marvin
@Description: atmosphere correction (MODIS 1KM)
'''
import os

from utils import date_teanslator

def main(ifile, geo_file, path_out):
    '''
    @description: AQUA/MODIS SeaDAS大气校正
    @ifile {str} L1B文件
    @geo_file {str} 地理定位文件
    @path_out {str} 输出路径
    @return: 
    '''
    date_str = os.path.split(ifile)[1].split('.')[1]
    date_str = date_teanslator.jd_to_cale(date_str[1:])
    time_str = os.path.split(ifile)[1].split('.')[2]
    year = int(date_str.split('.')[0])
    month = int(date_str.split('.')[1])
    day = int(date_str.split('.')[2])
    hour = int(time_str[0:2])
    minute = int(time_str[2:])
    date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
    nrow = os.path.split(ifile)[1].split('.')[3]
    out_name = 'TERRA_MODIS_1000_L2_%s_%s_00.hdf' % (date_str, nrow)
    raster_fn_out = os.path.join(path_out, out_name)
    cmd = 'l2gen ifile=%s geofile=%s cloud_thresh=0.05 ofile=%s' % \
        (ifile, geo_file, raster_fn_out)
    os.system(cmd)
    return(raster_fn_out)
