# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: VIIRS 373m 数据预处理
@Date: 2019-08-02 23:22:35
'''
import os
import sys
import re
import simplejson
import numpy as np
import time
from osgeo import gdal, osr, ogr
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
import cv2

import utils.date_teanslator as date_teanslator


# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


def cut_data(subrange, lon_all, lat_all):
    '''
    @description: 计算影像裁切范围
    @subrange {[lon_min, lon_max, lat_min, lat_max]}: 
    @lon_all {numpy array}: 栅格经度
    @lat_all {numpy array}: 栅格纬度
    @return: 
    '''
    dist = (lon_all - subrange[0])**2 + (lat_all - subrange[3])**2
    minloca = np.argmin(dist)
    n_colm = lon_all.shape[1]
    min_loca_row = int(minloca / n_colm)  # 裁切起始行号
    min_loca_colm = minloca % n_colm  # 裁切起始列号
    dist = (lon_all - subrange[1])**2 + (lat_all - subrange[2])**2
    minloca = np.argmin(dist)
    max_loca_row = int(minloca / n_colm)  # 裁切终止行号
    max_loca_colm = minloca % n_colm  # 裁切终止列号
    if min_loca_row == max_loca_row | min_loca_colm == max_loca_colm:
        return(-1)
    else:
        if min_loca_row > max_loca_row:
            tmp = min_loca_row
            min_loca_row = max_loca_row
            max_loca_row = tmp
        if min_loca_colm > max_loca_colm:
            tmp = min_loca_colm
            min_loca_colm = max_loca_colm
            max_loca_colm = tmp
        return([min_loca_row, max_loca_row, min_loca_colm, max_loca_colm])


def arms_corr(raster_data, mtl_coef, wave_index):
    '''
    @description: 辐射定标和6S大气校正
    @raster_data {numpy array} 原始数据
    @mtl_coef {dict} 所需的校正参数，以下是必须包含的键值：
        -- altitude {float} 海拔(km)
        -- visibility {float} 能见度(km)
        -- type_aero {int} 气溶胶模型 0 无; 1 大陆; 2 近海; 3 城市; 5 沙漠;
                                    6 有机质燃烧; 7 平流层
    @wave_index {int} 波段代号(详情查看6S代码)
    @return {numpy array} 大气校正后的栅格数据
    '''
    path_6s = global_config['path_6s']
    with open(os.path.join(path_6s, 'in.txt'), 'w') as fp:
        # igeom
        fp.write('%d\n' % 0)
        # SOLZ SOLA THV PHV month day
        SOLZ = mtl_coef['solz']
        SOLA = mtl_coef['sola']
        PHV = mtl_coef['sala']
        THV = mtl_coef['salz']
        month = int(mtl_coef['month'])
        day = int(mtl_coef['day'])
        fp.write('%.2f %.2f %.2f %.2f %d %d\n' % (SOLZ, SOLA, THV, PHV, month, day))
        # atms 
        center_lat = mtl_coef['location'][1]
        if center_lat<23.5 and month<9 and month>2:
            fp.write('%d\n' % 1)
        elif center_lat>=23.5 and center_lat<66.5 and month<9 and month>2:
            fp.write('%d\n' % 2)
        elif center_lat>=23.5 and center_lat<66.5 and (month>=9 or month<=2):
            fp.write('%d\n' % 3)
        elif center_lat>=66.5 and month<9 and month>2:
            fp.write('%d\n' % 4)
        elif center_lat>=66.5 and (month>=9 or month<=2):
            fp.write('%d\n' % 5)
        else:
            print('无法确定大气模型')
            return(0)
        # 气溶胶模型
        fp.write('%d\n' % mtl_coef['aero_type'])
        # 能见度
        fp.write('%.1f\n' % mtl_coef['visibility'])
        # 高程
        fp.write('%.3f\n' % -mtl_coef['altitude'])
        # 传感器类型
        fp.write('%d\n' % -1000)
        # 波段号
        fp.write('%d\n' % wave_index)
        # 其余参数
        fp.write('%d\n' % 0) # 无方向反射
        fp.write('%d\n' % 0) # 朗伯体假设
        fp.write('%d\n' % 4) # 湖泊水体
        fp.write('%d\n' % 0) # 进行大气校正
        fp.write('%.2f\n' % 0.01) # 默认反射率
        fp.write('%d\n' % 5) # 除了0,1,2外任意设置
    os.system('cd %s && ./sixs_Lin<./in.txt>log.txt' % (path_6s))
    with open(os.path.join(path_6s, 'sixs.out'), 'r') as fp:
        for line in fp.readlines():
            if 'coefficients xa xb xc' in line:
                coefAll = re.findall(r'[\.\d]+', line)
                # y=xa*(measured radiance)-xb;  acr=y/(1.+xc*y)
                xa = float(coefAll[0])
                xb = float(coefAll[1])
                xc = float(coefAll[2])
                y = xa*raster_data - xb
                atms_corr_data = y/(1.0+xc*y)/3.14159
                break
    return(atms_corr_data)


def main0(file_ref, file_info, subrange, aerotype=1, altitude=0.01, visibility=15, path_out=None):
    '''
    @description: 主程序
    @file_ref {str}: DN数据文件
    @file_info {str}: 影像信息数据文件(包含经纬度、传感器方位信息等)
    @subrange {lon_min, lon_max, lat_min, lat_max}
    @return: None
    '''
    # 文件信息获取
    file_sd = SD(file_info)
    obj = file_sd.select('Longitude')
    lon = obj.get()
    obj = file_sd.select('Latitude')
    lat = obj.get()
    # 构造目标经纬度
    if subrange is None:
        griddata_key = False # 不使用griddata插值(考虑效率)
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        subrange = [lon_min, lon_max, lat_min, lat_max]
        xsize = lon.shape[1]
        ysize = lon.shape[0]
        xstep = (lon_max-lon_min) / xsize
        ystep = (lat_min-lat_max) / ysize
        x1d = np.linspace(subrange[0], subrange[1], xsize)
        y1d = np.linspace(subrange[3], subrange[2], ysize)
        [xx, yy] = np.meshgrid(x1d, y1d)
        cut_index = [0, ysize, 0, xsize]
        xsize0 = xsize
        ysize0 = ysize
        # 重构
        lon_1d = np.reshape(lon, lon.shape[0]*lon.shape[1])
        lat_1d = np.reshape(lat, lat.shape[0]*lat.shape[1])
        lonlat = np.vstack(([lon_1d], [lat_1d])).T
        lon_1d = None
        lat_1d = None
    else:
        griddata_key = True
        xstep = 0.375 / 111
        ystep = -0.375 / 111
        xsize = (subrange[1] - subrange[0]) * 111 / 0.375
        ysize = (subrange[3] - subrange[2]) * 111 / 0.375
        x1d = np.linspace(subrange[0], subrange[1], xsize)
        y1d = np.linspace(subrange[3], subrange[2], ysize)
        [xx, yy] = np.meshgrid(x1d, y1d)
        cut_index = cut_data(subrange, lon, lat)
        xsize0 = cut_index[3] - cut_index[2] # 裁切后的大小
        ysize0 = cut_index[1] - cut_index[0]
        lon_cut = lon[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
        lat_cut = lat[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
        lon_1d = np.reshape(lon_cut, lon_cut.shape[0]*lon_cut.shape[1])
        lat_1d = np.reshape(lat_cut, lat_cut.shape[0]*lat_cut.shape[1])
        lonlat = np.vstack(([lon_1d], [lat_1d])).T
        lon_cut = None
        lat_cut = None
        lon_1d = None
        lat_1d = None
    # 卫星方位角
    obj = file_sd.select('SatelliteAzimuthAngle')
    sala_all = obj.get()
    sala_cut = sala_all[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    sala = np.mean(sala_cut)
    if sala < 0:
        sala = sala+360
    sala_all = None
    sala_cut = None
    # 卫星天顶角
    obj = file_sd.select('SatelliteZenithAngle')
    salz_all = obj.get()
    salz_cut = salz_all[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    salz = np.mean(salz_cut)
    salz_all = None
    salz_cut = None
    # 太阳天顶角
    obj = file_sd.select('SolarZenithAngle')
    solz_all = obj.get()
    solz_cut = solz_all[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    solz = np.mean(solz_cut)
    solz_all = None
    solz_cut = None
    # 太阳方位角
    obj = file_sd.select('SolarAzimuthAngle')
    sola_all = obj.get()
    sola_cut = sola_all[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    sola = np.mean(sola_cut)
    sola_all = None
    sola_cut = None
    file_sd.end()
    file_sd = SD(file_ref)
    atms_corr_resample = np.zeros((xx.shape[0], xx.shape[1], 4))
    center_lonlat = [(subrange[0]+subrange[1])/2, (subrange[2]+subrange[3])/2]
    for i in range(4):
        obj_name = 'Radiance_I' + str(i+1)
        obj = file_sd.select(obj_name)
        data = obj.get()
        data_cut = data[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
        info = obj.attributes()
        scale = info['Scale']
        offset = info['Offset']
        radi_cali = data_cut * scale + offset
        date_str = os.path.split(file_ref)[1].split('.')[1]
        date_str = date_teanslator.jd_to_cale(date_str[1:])
        month = int(date_str.split('.')[1])
        day = int(date_str.split('.')[2])
        mtl_coef = {
            'altitude': altitude,
            'visibility': visibility,
            'aero_type': aerotype,
            'location': center_lonlat,
            'month': month,
            'day': day,
            'solz': solz,
            'sola': sola,
            'salz': salz,
            'sala': sala
        }
        atms_corr = arms_corr(radi_cali, mtl_coef, i+161)
        # 重采样
        print('I%d 大气校正 ...' % (i+1))
        atms_corr_1d = np.reshape(atms_corr, xsize0*ysize0)
        print('I%d 重采样 ...' % (i+1))
        if griddata_key:
            atms_corr_resample[:, :, i] = griddata(lonlat, atms_corr_1d, (xx, yy), method='nearest')
        else:
            # 采用最邻近填充法，效率较高，但有部分损失
            data_resize = np.zeros(xx.shape) + np.nan
            x_index = np.round((lon - lon_min)/xstep)
            y_index = ysize - 1 - np.round((lat - lat_min)/abs(ystep))
            usefall_key = np.logical_and(x_index < xsize, x_index >= 0)
            usefall_key = np.logical_and(usefall_key, y_index < ysize)
            usefall_key = np.logical_and(usefall_key, y_index >= 0)
            x_index_1d = x_index[usefall_key]
            y_index_1d = y_index[usefall_key]
            data_index_1d = atms_corr[usefall_key]
            for j in range(len(y_index[usefall_key])):
                data_resize[int(y_index_1d[j]), int(x_index_1d[j])] = data_index_1d[j]
    driver = gdal.GetDriverByName('GTiff')
    xsize = np.shape(atms_corr_resample)[1]
    ysize = np.shape(atms_corr_resample)[0]
    nbands = 4
    if '.hdf' in file_ref:
        # 输出文件名
        date_str = os.path.split(file_ref)[1].split('.')[1]
        time_str = os.path.split(file_ref)[1].split('.')[2]
        date_str = date_teanslator.jd_to_cale(date_str[1:])
        year = int(date_str.split('.')[0])
        month = int(date_str.split('.')[1])
        day = int(date_str.split('.')[2])
        hour = int(time_str[0:2])
        minute = int(time_str[2:])
        date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
        nrow = os.path.split(file_ref)[1].split('.')[3]
        out_name = 'NPP_VIIRS_375_L2_%s_%s_00.tif' % (date_str, nrow)
        raster_fn_out = os.path.join(path_out, out_name)
    else:
        print('无法识别的文件类型: %s' % os.path.split(file_ref)[1])
        return(0)
    target_ds = driver.Create(raster_fn_out, xsize, ysize, nbands, gdal.GDT_UInt16)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    geo_trans = (subrange[0], xstep, 0, subrange[3], 0, ystep)
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(nbands):
        data_tmp = atms_corr_resample[:,:,i]
        mask = np.logical_or(data_tmp >= 65530, np.isnan(data_tmp))
        data_tmp = (data_tmp * 10000).astype(np.int)
        data_tmp[mask] = 65530
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(1+1)
        band.SetNoDataValue(65530)
    target_ds = None
    return(raster_fn_out)

def main(file_ref, file_info, subrange, aerotype=1, altitude=0.01, visibility=15, path_out=None):
    '''
    @description: 主程序
    @file_ref {str}: DN数据文件
    @file_info {str}: 影像信息数据文件(包含经纬度、传感器方位信息等)
    @subrange {lon_min, lon_max, lat_min, lat_max}
    @return: None
    '''
    nbands = 4
    mask_value = 65533
    # hdf转geotif
    print('reconstruction...')
    hdf_merge = file_ref.replace('.hdf', '_merge.hdf')
    if os.path.exists(hdf_merge):
        os.system('rm %s' % hdf_merge)
    file_sd_out = SD(hdf_merge, SDC.CREATE | SDC.WRITE)
    file_sd1 = SD(file_ref)
    file_sd2 = SD(file_info)
    obj = file_sd2.select('Longitude')
    lon = obj.get()
    obj = file_sd2.select('Latitude')
    lat = obj.get()
    fields = ['Radiance_I%d' % (i+1) for i in range(nbands)]
    size = None
    for field in fields:
        obji = file_sd1.select(field)
        if size is None:
            size = obji[:].shape
        objo = file_sd_out.create(field, SDC.UINT16, size)
        objo.set(obji[:])
    obji = file_sd2.select('Longitude')
    objo = file_sd_out.create('Longitude', SDC.FLOAT32, size)
    objo.set(obji[:])
    obji = file_sd2.select('Latitude')
    objo = file_sd_out.create('Latitude', SDC.FLOAT32, size)
    objo.set(obji[:])
    obji.endaccess()
    objo.endaccess()
    file_sd_out.end()
    for i in range(nbands):
        ofile_tif = file_ref.replace('.hdf', '_band%s_reproj.tif' % str(i+1))
        cmd = '%s -geoloc -t_srs EPSG:4326 -srcnodata %s HDF4_SDS:UNKNOWN:"%s":%s %s' % (
            global_config['path_gdalwarp'], mask_value, hdf_merge, str(i), ofile_tif)
        os.system(cmd)
    # 卫星方位信息
    cut_index = cut_data(subrange, lon, lat)
    # 太阳天顶角
    obj = file_sd2.select('SolarZenithAngle')
    data = obj.get()
    data_cut = data[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    solz = np.mean(data_cut)
    # 太阳方位角
    obj = file_sd2.select('SolarAzimuthAngle')
    data = obj.get()
    data_cut = data[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    sola = np.mean(data_cut)
    # 卫星天顶角
    obj = file_sd2.select('SatelliteZenithAngle')
    data = obj.get()
    data_cut = data[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    salz = np.mean(data_cut)
    # 卫星方位角
    obj = file_sd2.select('SatelliteAzimuthAngle')
    data = obj.get()
    data_cut = data[cut_index[0]:cut_index[1], cut_index[2]:cut_index[3]]
    sala = np.mean(data_cut)
    center_lonlat = [(subrange[0]+subrange[1])/2, (subrange[2]+subrange[3])/2]
    raster = gdal.Open(file_ref.replace('.hdf', '_band1_reproj.tif'))
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    geo_trans = raster.GetGeoTransform()
    proj_ref = raster.GetProjectionRef()
    # 计算裁切范围
    target_lon_min = 116.28
    target_lon_max = 125.0
    target_lat_min = 30.0
    target_lat_max = 37.83
    colm_s = int(round((target_lon_min - geo_trans[0]) / geo_trans[1]))
    colm_e = int(round((target_lon_max - geo_trans[0]) / geo_trans[1]))
    line_s = int(round((target_lat_max - geo_trans[3]) / geo_trans[5]))
    line_e = int(round((target_lat_min - geo_trans[3]) / geo_trans[5]))
    if colm_s < 0:
        colm_s = 0
    if line_s < 0:
        line_s = 0
    if colm_e >= xsize:
        colm_e = xsize - 1
    if line_e >= ysize:
        line_e = ysize - 1
    x_1d = np.array([geo_trans[0]+i*geo_trans[1] for i in range(xsize)])
    y_1d = np.array([geo_trans[3]+i*geo_trans[5] for i in range(ysize)])
    xx, yy = np.meshgrid(x_1d, y_1d)
    xx_sub = xx[line_s:line_e, colm_s:colm_e]
    yy_sub = yy[line_s:line_e, colm_s:colm_e]
    # 文件保存所需信息
    date_str = os.path.split(file_ref)[1].split('.')[1]
    time_str = os.path.split(file_ref)[1].split('.')[2]
    date_str = date_teanslator.jd_to_cale(date_str[1:])
    year = int(date_str.split('.')[0])
    month = int(date_str.split('.')[1])
    day = int(date_str.split('.')[2])
    hour = int(time_str[0:2])
    minute = int(time_str[2:])
    date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
    nrow = os.path.split(file_ref)[1].split('.')[3]
    out_name = 'NPP_VIIRS_375_L2_%s_%s_00.tif' % (date_str, nrow)
    raster_fn_out = os.path.join(path_out, out_name)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(raster_fn_out, xsize, ysize, nbands, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    # 大气校正
    for i in range(nbands):
        obj_name = 'Radiance_I' + str(i+1)
        obj = file_sd1.select(obj_name)
        raster = gdal.Open(file_ref.replace('.hdf', '_band%s_reproj.tif' % str(i+1)))
        data = raster.GetRasterBand(1).ReadAsArray()
        print('重采样:Band %s' % (i+1))
        data_sub = data[line_s:line_e, colm_s:colm_e]
        blank_key = data_sub == mask_value
        # OpenCV形态学处理
        blank_key[:, 0] = 0
        blank_key[:, -1] = 0
        blank_key[0, :] = 0
        blank_key[-1, :] = 0
        labels_struct = cv2.connectedComponentsWithStats(
            blank_key.astype(np.uint8), connectivity=4)
        for i_label in range(1, labels_struct[0]):
            if labels_struct[2][i_label][4] > 1e5:
                blank_key[labels_struct[1] == i_label] = 0
        lon_blank = xx_sub[blank_key]
        lat_blank = yy_sub[blank_key]
        valid_key = np.logical_not(blank_key)
        lon_valid = xx_sub[valid_key]
        lat_valid = yy_sub[valid_key]
        lonlat = np.vstack((lon_valid, lat_valid)).T
        data_valid = data_sub[valid_key]
        data_blank = griddata(lonlat, data_valid, (lon_blank, lat_blank), method='nearest')
        data_sub[blank_key] = data_blank
        data[line_s:line_e, colm_s:colm_e] = data_sub
        mask = data == mask_value
        print('I%d 辐射定标和大气校正 ...' % (i+1))
        info = obj.attributes()
        scale = info['Scale']
        offset = info['Offset']
        radi_cali = data.astype(float) * scale + offset
        date_str = os.path.split(file_ref)[1].split('.')[1]
        date_str = date_teanslator.jd_to_cale(date_str[1:])
        month = int(date_str.split('.')[1])
        day = int(date_str.split('.')[2])
        mtl_coef = {
            'altitude': altitude,
            'visibility': visibility,
            'aero_type': aerotype,
            'location': center_lonlat,
            'month': month,
            'day': day,
            'solz': solz,
            'sola': sola,
            'salz': salz,
            'sala': sala
        }
        atms_corr = arms_corr(radi_cali, mtl_coef, i+161)
        # save
        data_tmp = (atms_corr * 10000).astype(np.int)
        data_tmp[mask] = mask_value
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(i+1)
        band.SetNoDataValue(mask_value)
    target_ds = None
    file_sd1.end()
    file_sd2.end()   
    # 删除过程文件
    os.system('rm %s' % hdf_merge)
    for i in range(nbands):
        os.system('rm %s' % (file_ref.replace('.hdf', '_band%s_reproj.tif' % str(i+1))))
    return(raster_fn_out)
