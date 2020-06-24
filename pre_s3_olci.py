# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: Sentinel-3 OLCI数据预处理
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
from pyhdf.SD import SD, SDC

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


def arms_corr(raster_data, mtl_coef, band_index):
    '''
    @description: 辐射定标和6S大气校正
    @raster_data {numpy array} 原始数据
    @mtl_coef {dict} 所需的校正参数，以下是必须包含的键值：
        -- altitude {float} 海拔(km)
        -- visibility {float} 能见度(km)
        -- type_aero {int} 气溶胶模型 0 无; 1 大陆; 2 近海; 3 城市; 5 沙漠;
                                    6 有机质燃烧; 7 平流层
    @band_index {int}
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
        fp.write('%.2f %.2f %.2f %.2f %d %d\n' %
                 (SOLZ, SOLA, THV, PHV, month, day))
        # atms
        center_lat = mtl_coef['location'][1]
        if center_lat < 23.5 and month < 9 and month > 2:
            fp.write('%d\n' % 1)
        elif center_lat >= 23.5 and center_lat < 66.5 and month < 9 and month > 2:
            fp.write('%d\n' % 2)
        elif center_lat >= 23.5 and center_lat < 66.5 and (month >= 9 or month <= 2):
            fp.write('%d\n' % 3)
        elif center_lat >= 66.5 and month < 9 and month > 2:
            fp.write('%d\n' % 4)
        elif center_lat >= 66.5 and (month >= 9 or month <= 2):
            fp.write('%d\n' % 5)
        else:
            print('无法确定大气模型')
            exit(0)
        # 气溶胶模型
        fp.write('%d\n' % mtl_coef['aero_type'])
        # 能见度
        fp.write('%.1f\n' % mtl_coef['visibility'])
        # 高程
        fp.write('%.3f\n' % -mtl_coef['altitude'])
        # 传感器类型
        fp.write('%d\n' % -1000)
        # 波段号
        fp.write('-2\n')
        fp.write('%.3f %.3f\n' % (wavelength_low[band_index], wavelength_up[band_index]))
        # 其余参数
        fp.write('%d\n' % 0)  # 无方向反射
        fp.write('%d\n' % 0)  # 朗伯体假设
        fp.write('%d\n' % 4)  # 湖泊水体
        fp.write('%d\n' % 0)  # 进行大气校正
        fp.write('%.2f\n' % 0.01)  # 默认反射率
        fp.write('%d\n' % 5)  # 除了0,1,2外任意设置
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
                atms_corr_data = atms_corr_data / PI / np.exp(-0.5*tau_r[band_index]/np.cos(SOLZ/180*PI))
                break
    return(atms_corr_data)


def main(ifile, shp_file, center_lonlat, cut_range=None,
        aerotype=1, altitude=0.01, visibility=15, band_need=['all'],
        path_out=None):
    nbands = 21
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
        time_str = re.findall(r'\d+', path_name)[3]
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        date = '%d/%d/%d %d:%d:00' % (year, month, day, hour, minute)
        sola_position = calc_sola_position.main(center_lonlat[0], center_lonlat[1], date)
        solz = sola_position[0]
        sola = sola_position[1]

    if 'tie_geometries.nc' in file_list:
        f_geometries = h5py.File(os.path.join(file_path, 'tie_geometries.nc'), 'r')
        sala = f_geometries['OAA']
        salz = f_geometries['OZA']
        sala_all = sala[:, :]
        salz_all = salz[:, :]
        f_geometries.close()
    if 'geo_coordinates.nc' in file_list:
        f_coordinates = h5py.File(os.path.join(file_path, 'geo_coordinates.nc'), 'r')
        lon = f_coordinates['longitude'][:]
        lat = f_coordinates['latitude'][:]
        lon = lon.astype(float) * 1e-6
        lat = lat.astype(float) * 1e-6
        x_1d = np.linspace(0, lon.shape[1]-1, lon.shape[1])
        y_1d = np.linspace(0, lon.shape[0]-1, lon.shape[0])
        [xx, yy] = np.meshgrid(x_1d, y_1d)
        distance = ((center_lonlat[0] - lon)**2 +
                    (center_lonlat[1] - lat)**2)**0.5 * 111
        location_x = np.mean(xx[distance < 1])
        location_y = int(np.mean(yy[distance < 1]))
        location_x = int(np.mean((location_x / lon.shape[1]) * 77))
        sala = sala_all[location_y, location_x] * 1e-6
        if sala < 0:
            sala = 360 + sala
        salz = salz_all[location_y, location_x] * 1e-6
        f_coordinates.close()
        size_x = lon.shape[1]
        size_y = lon.shape[0]
        ul_x = 0
        ul_y = 0
        lr_x = size_x
        lr_y = size_y
    else:
        print('文件缺失')
        return(0)

    size_x = np.shape(lon)[1]
    size_y = np.shape(lat)[0]
    rrs_join = None
    # mask array
    f_name = os.path.join(file_path, 'Oa01_radiance.nc')
    fp_h5 = h5py.File(f_name, 'r')
    data = fp_h5['Oa01_radiance']
    data = data[ul_y:lr_y, ul_x:lr_x]
    hdf_merge = os.path.join(file_path, 'reproj.hdf')
    file_sd_out = SD(hdf_merge, SDC.CREATE | SDC.WRITE)
    size = np.shape(data)
    objo = file_sd_out.create('Oa%s', SDC.UINT16, size)
    objo.set(data)
    objo = file_sd_out.create('longitude', SDC.FLOAT32, size)
    objo.set(lon.astype(np.float32))
    objo = file_sd_out.create('latitude', SDC.FLOAT32, size)
    objo.set(lat.astype(np.float32))
    objo.endaccess()
    file_sd_out.end()
    os.system('cd %s && gdalwarp -t_srs EPSG:4326 HDF4_SDS:UNKNOWN:"%s":0 %s' % (
        file_path, 'reproj.hdf', 'reproj.tif'))
    raster_tmp = gdal.Open(os.path.join(file_path, 'reproj.tif'))
    geo_trans_dst = raster_tmp.GetGeoTransform()
    mask_array = raster_tmp.GetRasterBand(1).ReadAsArray()
    mask_array = np.logical_or(mask_array==65535, mask_array==0) 
    raster_tmp = None
    os.remove(hdf_merge)
    os.remove(os.path.join(file_path, 'reproj.tif'))

    for i_band in range(nbands):
        Oa_index = 'Oa%02d' % (i_band+1)
        print(Oa_index)
        if (Oa_index in band_need) or ('all' in band_need):
            f_name = os.path.join(file_path, '%s_radiance.nc' % Oa_index)
            fp_h5 = h5py.File(f_name, 'r')
            data = fp_h5['%s_radiance' % Oa_index]
            data = data[ul_y:lr_y, ul_x:lr_x].astype(float)
            offset = fp_h5['%s_radiance' % Oa_index].attrs['add_offset']
            scale = fp_h5['%s_radiance' % Oa_index].attrs['scale_factor']
            # 辐射定标
            Lr = data * scale + offset
            # 大气校正
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
            atms_corr = arms_corr(Lr, mtl_coef, i_band)
            # 重采样
            hdf_merge = os.path.join(file_path, 'reproj.hdf')
            file_sd_out = SD(hdf_merge, SDC.CREATE | SDC.WRITE)
            size = np.shape(atms_corr)
            objo = file_sd_out.create('Oa%s', SDC.FLOAT32, size)
            objo.set(atms_corr.astype(np.float32))
            objo = file_sd_out.create('longitude', SDC.FLOAT32, size)
            objo.set(lon.astype(np.float32))
            objo = file_sd_out.create('latitude', SDC.FLOAT32, size)
            objo.set(lat.astype(np.float32))
            objo.endaccess()
            file_sd_out.end()
            os.system('cd %s && gdalwarp -t_srs EPSG:4326 HDF4_SDS:UNKNOWN:"%s":0 %s' % (
                file_path, 'reproj.hdf', 'reproj.tif'))
            raster_tmp = gdal.Open(os.path.join(file_path, 'reproj.tif'))
            geo_trans_dst = raster_tmp.GetGeoTransform()
            if rrs_join is None:
                rrs_join = np.zeros([raster_tmp.RasterYSize, raster_tmp.RasterXSize, nbands])
            rrs_join[:, :, i_band] = raster_tmp.GetRasterBand(1).ReadAsArray()
            raster_tmp = None
            os.remove(hdf_merge)
            os.remove(os.path.join(file_path, 'reproj.tif'))
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
    name_out = 'Sentinel3%s_OLCI_300_L2_%s_%s_%s.tif' % (satellite_code, date_str, nrow, ncolm)
    raster_fn_out = os.path.join(path_out, name_out)
    target_ds = driver.Create(raster_fn_out, np.shape(rrs_join)[1], np.shape(rrs_join)[0], nbands, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_trans_dst)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetProjection(proj_ref)
    for i in range(nbands):
        data_tmp = rrs_join[:, :, i]
        data_tmp = (data_tmp * 10000).astype(np.int)
        data_tmp[mask_array] = 65530
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(1+1)
        band.SetNoDataValue(65530)
    target_ds = None
    # 删除解压文件
    fp_h5.close()
    shutil.rmtree(file_path)
