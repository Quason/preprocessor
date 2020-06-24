# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: Sentinel-3 SLSTR数据预处理
@Date: 2019-08-04 12:20:56
'''
import os
import sys
import re
import simplejson
import numpy as np
import time
import h5py
import zipfile
import shutil
from osgeo import gdal, osr
from scipy.interpolate import griddata

import utils.img_cut as img_cut
import utils.date_teanslator as date_teanslator
import utils.calc_sola_position as calc_sola_position

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


wavelength_low = [0.393, 0.407, 0.437, 0.485, 0.505, 0.555, 0.615, 0.660, 0.670, 0.678,
                0.703, 0.75, 0.76, 0.7625, 0.766, 0.771, 0.855, 0.88, 0.895, 0.93, 1.0]
wavelength_up = [0.4075, 0.4175, 0.4475, 0.495, 0.515, 0.565, 0.625, 0.67, 0.6765, 0.685,
                0.71375, 0.7575, 0.762, 0.766, 0.76875, 0.78625, 0.875, 0.89, 0.905, 0.95, 1.04]
PI = 3.14159
tau_r = [0.359559, 0.317976, 0.237668, 0.15546, 0.13194, 0.0900236, 0.0594837, 0.0447581,
        0.0423839, 0.0406457, 0.0345252, 0.0269168, 0.0259299, 0.0255209, 0.0251199,
        0.0235926, 0.0154609, 0.0140998, 0.0131764, 0.0110591, 0.00796062]


def main(ifile, shp_file, center_lonlat, cut_range=None,
        aerotype=1, altitude=0.01, visibility=15, band_need=['all'],
        path_out=None):
    nbands = 3
    # 文件解压
    print('文件解压...')
    fz = zipfile.ZipFile(ifile, 'r')
    for file in fz.namelist():
        fz.extract(file, os.path.split(ifile)[0])
    path_name = os.path.split(ifile)[1]
    path_name = path_name.replace('.zip', '.SEN3')
    file_path = os.path.join(os.path.split(ifile)[0], path_name)
    file_list = os.listdir(file_path)
    print('数据预处理...')
    # 通过文件名获取传感器日期
    path_name = os.path.split(file_path)[1]
    if path_name == '':
        print('文件夹错误: %s' % path_name)
        exit(0)
    else:
        date_str = re.findall(r'\d+', path_name)[2]
    if 'geodetic_in.nc' in file_list:
        f_coordinates = h5py.File(os.path.join(file_path, 'geodetic_in.nc'))
        lon = f_coordinates['longitude_in']
        lat = f_coordinates['latitude_in']
        lon = lon[:, :]
        lon = lon.astype(float) * 1e-6
        lat = lat[:, :]
        lat = lat.astype(float) * 1e-6
        x_1d = np.linspace(0, lon.shape[1]-1, lon.shape[1])
        y_1d = np.linspace(0, lon.shape[0]-1, lon.shape[0])
        [xx, yy] = np.meshgrid(x_1d, y_1d)
        distance = ((center_lonlat[0] - lon)**2 +
                    (center_lonlat[1] - lat)**2)**0.5 * 111
        f_coordinates.close()
        if cut_range is None:
            griddata_key = False
            lon_min = np.min(lon)
            lon_max = np.max(lon)
            lat_min = np.min(lat)
            lat_max = np.max(lat)
            size_x = lon.shape[1]
            size_y = lon.shape[0]
            lon_target_1d = np.linspace(lon_min, lon_max, size_x)
            lat_target_1d = np.linspace(lat_min, lat_max, size_y)
            [lon_target, lat_target] = np.meshgrid(lon_target_1d, lat_target_1d) # 目标插值网格
            ul_x = 0
            ul_y = 0
            lr_x = size_x
            lr_y = size_y
            step_x = (lon_max-lon_min) / size_x
            step_y = (lat_max-lat_min) / size_y
            cut_range = [lon_min, lon_max, lat_min, lat_max]
        else:
            griddata_key = True
            step_x = 0.009
            step_y = 0.009
            # 确定裁切角点坐标
            distance = ((cut_range[0] - 0.2 - lon)**2 +
                        (cut_range[3] + 0.2 - lat)**2)**0.5 * 111
            ul_x = np.mean(xx[distance < 0.5])
            ul_y = np.mean(yy[distance < 0.5])
            if np.isnan(ul_x):
                ul_x = 0
            else:
                ul_x = int(ul_x)
            if np.isnan(ul_y):
                ul_y = 0
            else:
                ul_y = int(ul_y)
            distance = ((cut_range[1] + 0.2 - lon)**2 +
                        (cut_range[2] - 0.2 - lat)**2)**0.5 * 111
            lr_x = np.mean(xx[distance < 0.5])
            lr_y = np.mean(yy[distance < 0.5])
            if np.isnan(lr_x):
                lr_x = distance.shape[1]
            else:
                lr_x = int(lr_x)
            if np.isnan(lr_y):
                lr_y = distance.shape[0]
            else:
                lr_y = int(lr_y)
            # 构建目标插值网格
            target_size_x = int((cut_range[1] - cut_range[0]) / step_x)
            target_size_y = int((cut_range[3] - cut_range[2]) / step_y)
            lon_target_1d = np.linspace(cut_range[0], cut_range[1], target_size_x)
            lat_target_1d = np.linspace(cut_range[3], cut_range[2], target_size_y)
            [lon_target, lat_target] = np.meshgrid(lon_target_1d, lat_target_1d)
    else:
        print('文件缺失')
        return(0)
    size_x = lon_target.shape[1]
    size_y = lon_target.shape[0]
    rrs_join = np.zeros([size_y, size_x, nbands])
    for i_band in range(nbands):
        Oa_index = 'S%d' % (i_band+7)
        print(Oa_index)
        if (Oa_index in band_need) or ('all' in band_need):
            f_name = os.path.join(file_path, '%s_BT_in.nc' % Oa_index)
            fp_h5 = h5py.File(f_name, 'r')
            data = fp_h5['%s_BT_in' % Oa_index]
            data = data[ul_y:lr_y, ul_x:lr_x].astype(float)
            offset = fp_h5['%s_BT_in' % Oa_index].attrs['add_offset']
            scale = fp_h5['%s_BT_in' % Oa_index].attrs['scale_factor']
            # 辐射定标
            Lr = data * scale + offset
            atms_corr = Lr
            # 重采样
            atms_corr_1d = np.reshape(atms_corr, atms_corr.shape[0]*atms_corr.shape[1])
            lon_cut = lon[ul_y:lr_y, ul_x:lr_x]
            lat_cut = lat[ul_y:lr_y, ul_x:lr_x]
            lon_cut_1d = np.reshape(lon_cut, lon_cut.shape[0]*lon_cut.shape[1])
            lat_cut_1d = np.reshape(lat_cut, lat_cut.shape[0]*lat_cut.shape[1])
            x1y1 = np.vstack([[lon_cut_1d], [lat_cut_1d]]).T
            if griddata_key:
                rrs_join[:, :, i_band] = griddata(x1y1, atms_corr_1d, (lon_target, lat_target), method='linear')
            else:
                data_resize = np.zeros(xx.shape) + np.nan
                x_index = np.round((lon - lon_min)/step_x)
                y_index = size_y - 1 - np.round((lat - lat_min)/abs(step_y))
                usefall_key = np.logical_and(x_index < size_x, x_index >= 0)
                usefall_key = np.logical_and(usefall_key, y_index < size_y)
                usefall_key = np.logical_and(usefall_key, y_index >= 0)
                x_index_1d = x_index[usefall_key]
                y_index_1d = y_index[usefall_key]
                data_index_1d = atms_corr[usefall_key]
                for j in range(len(y_index[usefall_key])):
                    data_resize[int(y_index_1d[j]), int(x_index_1d[j])] = data_index_1d[j]
                rrs_join[:, :, i_band] = data_resize
    driver = gdal.GetDriverByName('GTiff')
    name_short = os.path.split(ifile)[1]
    name_short_split = name_short.split('_')
    for item in name_short_split:
        if len(item) == 15:
            date_str = item.replace('T', '')
            date_str = str(int(date_str) + 80000) # 转换为北京时间
            break
    if name_short_split[0] == 'S3A':
        satellite_code = 'A'
    elif name_short_split[0] == 'S3B':
        satellite_code = 'B'
    else:
        satellite_code = ''
    nrow = name_short_split[-8]
    ncolm = name_short_split[-7]
    name_out = 'Sentinel3%s_SLSTR_1000_L2_%s_%s_%s.tif' % (satellite_code, date_str, nrow, ncolm)
    raster_fn_out = os.path.join(path_out, name_out)
    target_ds = driver.Create(raster_fn_out, size_x, size_y, nbands, gdal.GDT_UInt16)
    geo_trans = (cut_range[0], step_x, 0, cut_range[3], 0, -step_y)
    target_ds.SetGeoTransform(geo_trans)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetProjection(proj_ref)
    for i in range(nbands):
        data_tmp = rrs_join[:, :, i]
        mask = np.isnan(data_tmp)
        data_tmp = (data_tmp * 100).astype(np.int)
        data_tmp[mask] = 65530
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(1+1)
        band.SetNoDataValue(65530)
    target_ds = None
    # 删除解压文件
    fp_h5.close()
    shutil.rmtree(file_path)


def stack_rgb(ifile):
    raster = gdal.Open(ifile)
    xsize = raster.GetRasterBand(1).ReadAsArray().shape[1]
    ysize = raster.GetRasterBand(1).ReadAsArray().shape[0]
    nbands = 5
    if '.tif' in ifile:
        raster_fn_out = ifile.replace('.tif', '_rgb.tif')
    else:
        print('无法识别的文件名: %s' % os.path.split(ifile)[1])
        exit(0)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(raster_fn_out, xsize, ysize, nbands, gdal.GDT_Float64)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjectionRef())
    target_ds.GetRasterBand(1).WriteArray(raster.GetRasterBand(4).ReadAsArray())
    target_ds.GetRasterBand(2).WriteArray(raster.GetRasterBand(6).ReadAsArray())
    target_ds.GetRasterBand(3).WriteArray(raster.GetRasterBand(8).ReadAsArray())
    target_ds.GetRasterBand(4).WriteArray(raster.GetRasterBand(17).ReadAsArray())
    target_ds.GetRasterBand(5).WriteArray(raster.GetRasterBand(21).ReadAsArray())
    target_ds = None
    raster = None
