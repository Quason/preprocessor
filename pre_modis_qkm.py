# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: EOS/MODIS 数据处理
@Date: 2019-08-02 23:22:35
'''
import os
import sys
import re
import simplejson
import numpy as np
import time
import math
from osgeo import gdal
from pyhdf.SD import SD

import utils.img_cut as img_cut
import utils.date_teanslator as date_teanslator
import utils.calc_sola_position as calc_sola_position


# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


modis_band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8', 'B9', 'B10', 'B11',
            'B12', 'B13lo', 'B13hi', 'B14lo', 'B14hi', 'B15', 'B17', 'B18', 'B19',
            'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29',
            'B30', 'B31', 'B32', 'B33', 'B34', 'B35', 'B36']


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


def satellite_position(ifile, shp_file):
    '''
    @description: 相对精确的天顶角方位角计算
    @ifile {str}: L1B swath文件 
    @shp_file {str}: 研究区域矢量文件
    @return: {salz, sala}
    '''
    # 设置环境变量
    os.environ['MRTDATADIR'] = global_config['MRTDATADIR']
    os.environ['PGSHOME'] = global_config['PGSHOME']
    os.environ['MRTBINDIR'] = global_config['MRTBINDIR']
    run_path = os.path.split(ifile)[0]
    heg_bin = os.path.join(global_config['MRTBINDIR'], 'hegtool')
    os.system('cd %s && %s -h %s > heg.log' % (run_path, heg_bin, ifile))
    info_file = os.path.join(run_path, 'HegHdr.hdr')
    if not(os.path.exists(info_file)):
        print('获取文件信息出错：%s' % ifile)
        return(0)
    else:
        lat_min = None
        lat_max = None
        lon_min = None
        lon_max = None
        pixel_x = None
        pixel_y = None
        with open(info_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if 'SWATH_LAT_MIN' in line:
                    lat_min = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LAT_MAX' in line:
                    lat_max = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LON_MIN' in line:
                    lon_min = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LON_MAX' in line:
                    lon_max = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_X_PIXEL_RES_DEGREES' in line:
                    pixel_x = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_Y_PIXEL_RES_DEGREES' in line:
                    pixel_y = float(re.findall(r'[\d.]+', line)[0])
        if lat_min and lat_max and lon_min and lon_max and pixel_x and pixel_y:
            prm_file = os.path.join(run_path, 'HegSwath.prm')
            # 卫星天顶角、方位角重投影
            print('reprojection SALZ ...')
            if '.hdf' in ifile:
                out_file = ifile.replace('.hdf', '_SALZ.tif')
                if os.path.exists(out_file):
                    os.remove(out_file)
            else:
                print('无法识别的文件格式：%s' % os.path.split(ifile)[1])
            with open(prm_file, 'wb') as fp:
                fp.write(b'\nNUM_RUNS = 1\n\n')
                fp.write(b'BEGIN\n')
                fp.write(bytes('INPUT_FILENAME = %s\n' % ifile, 'utf-8'))
                fp.write(b'OBJECT_NAME = MODIS_SWATH_Type_L1B\n')
                fp.write(b'FIELD_NAME = SensorZenith|\n')
                fp.write(b'BAND_NUMBER = 1\n')
                fp.write(bytes('OUTPUT_PIXEL_SIZE_X = %f\n' % pixel_x, 'utf-8'))
                fp.write(bytes('OUTPUT_PIXEL_SIZE_Y = %f\n' % pixel_y, 'utf-8'))
                fp.write(bytes('SPATIAL_SUBSET_UL_CORNER = ( %f %f )\n' % (lat_max, lon_min), 'utf-8'))
                fp.write(bytes('SPATIAL_SUBSET_LR_CORNER = ( %f %f )\n' % (lat_min, lon_max), 'utf-8'))
                fp.write(b'OUTPUT_PROJECTION_TYPE = GEO\n')
                fp.write(bytes('OUTPUT_FILENAME = %s\n' % out_file, 'utf-8'))
                fp.write(b'OUTPUT_TYPE = GEO\n')
                fp.write(b'END\n\n')
            swtif_bin = os.path.join(global_config['MRTBINDIR'], 'swtif')
            os.system('cd %s && %s -P HegSwath.prm > heg.log' % (run_path, swtif_bin))
            print('reprojection SALA ...')
            if '.hdf' in ifile:
                out_file = ifile.replace('.hdf', '_SALA.tif')
                if os.path.exists(out_file):
                    os.remove(out_file)
            else:
                print('无法识别的文件格式：%s' % os.path.split(ifile)[1])
            with open(prm_file, 'wb') as fp:
                fp.write(b'\nNUM_RUNS = 1\n\n')
                fp.write(b'BEGIN\n')
                fp.write(bytes('INPUT_FILENAME = %s\n' % ifile, 'utf-8'))
                fp.write(b'OBJECT_NAME = MODIS_SWATH_Type_L1B\n')
                fp.write(b'FIELD_NAME = SensorAzimuth|\n')
                fp.write(b'BAND_NUMBER = 1\n')
                fp.write(bytes('OUTPUT_PIXEL_SIZE_X = %f\n' % pixel_x, 'utf-8'))
                fp.write(bytes('OUTPUT_PIXEL_SIZE_Y = %f\n' % pixel_y, 'utf-8'))
                fp.write(bytes('SPATIAL_SUBSET_UL_CORNER = ( %f %f )\n' % (lat_max, lon_min), 'utf-8'))
                fp.write(bytes('SPATIAL_SUBSET_LR_CORNER = ( %f %f )\n' % (lat_min, lon_max), 'utf-8'))
                fp.write(b'OUTPUT_PROJECTION_TYPE = GEO\n')
                fp.write(bytes('OUTPUT_FILENAME = %s\n' % out_file, 'utf-8'))
                fp.write(b'OUTPUT_TYPE = GEO\n')
                fp.write(b'END\n\n')
            swtif_bin = os.path.join(global_config['MRTBINDIR'], 'swtif')
            os.system('cd %s && %s -P HegSwath.prm > heg.log' % (run_path, swtif_bin))
        else:
            print('[Error] 未获得全部的所需信息')
            return(0)
        # 计算研究区域的平均太阳天顶角和方位角
        salz_file = ifile.replace('.hdf', '_SALZ.tif')
        out_file = salz_file.replace('.tif', '_cut.tif')
        cut_range = img_cut.main(salz_file, shp_file=shp_file, out_file=out_file)
        raster = gdal.Open(salz_file)
        salz_all = raster.GetRasterBand(1).ReadAsArray()
        salz_all = salz_all[cut_range[2]:cut_range[3], cut_range[0]:cut_range[1]]
        salz = np.mean(salz_all) * 0.01
        sala_file = ifile.replace('.hdf', '_SALA.tif')
        raster = gdal.Open(sala_file)
        sala_all = raster.GetRasterBand(1).ReadAsArray()
        sala_all = sala_all[cut_range[2]:cut_range[3], cut_range[0]:cut_range[1]]
        sala = np.mean(sala_all) * 0.01
        raster = None
        # 删除过程文件
        file_name = ifile.replace('.hdf', '_SALA.tif')
        file_name_met = file_name.replace('.tif', '.tif.met')
        if os.path.exists(file_name) and os.path.exists(file_name_met):
            os.remove(file_name)
            os.remove(file_name_met)
        file_name = ifile.replace('.hdf', '_SALZ.tif')
        file_name_met = file_name.replace('.tif', '.tif.met')
        if os.path.exists(file_name) and os.path.exists(file_name_met):
            os.remove(file_name)
            os.remove(file_name_met)
        file_name = ifile.replace('.hdf', '_SALZ_cut.tif')
        if os.path.exists(file_name):
            os.remove(file_name)
        log_list = ['heg.log', 'swtif.log', 'HegSwath.prm', 'HegHdr.hdr', 'hegtool.log']
        for item in log_list:
            file_name = os.path.join(run_path, item)
            if os.path.exists(file_name):
                os.remove(file_name)
        return({'salz':salz, 'sala':sala})


def satellite_position_rough(ifile, center_lonlat):
    '''
    @description: 相对精确的天顶角方位角计算
    @ifile {str}: L1B swath文件 
    @return: {salz, sala}
    '''
    SD_file = SD(ifile)
    sds_obj_lon = SD_file.select('Longitude')
    lon = sds_obj_lon.get()
    sds_obj_lat = SD_file.select('Latitude')
    lat = sds_obj_lat.get()
    SD_file.end()
    alpha = 18 / 180 * 3.14
    center_lon = np.mean(lon)
    center_lat = np.mean(lat)
    center_lat = center_lonlat[1]
    dist_lon = abs(center_lonlat[0] - center_lon)
    dist_lat = abs(center_lonlat[1] - center_lat)
    dist = dist_lon/math.cos(alpha) + math.sin(alpha)*(dist_lat-dist_lon*math.tan(alpha))
    salz = math.degrees(math.atan(dist * 111 / 705))
    return(salz, 180.0)


def main(ifile, center_lonlat, salz, sala, satellite,
            aerotype=1, altitude=0.01, visibility=15,
            band_need=['all'], shp_file=None,
            path_out_radi=None, path_out_6s=None):
    '''
    @description: 
    @ifile {str}: L1B swath文件 
    @shp_file {str}: 研究区域矢量文件
    @center_lonlat {[lon, lat]}: 中心经纬度
    @aerotype {int}: 气溶胶类型(默认大陆型)
    @altitude {float}: 海拔(km)
    @visibility {float}: 能见度(km)
    @band_need {list}: 所选波段, all表示默认全部处理
    @path_out_radi {str}: 辐射定标结果文件输出路径
    @path_out_6s {str}: 大气校正结果文件输出路径
    @return: 
    '''
    # 设置环境变量
    os.environ['MRTDATADIR'] = global_config['MRTDATADIR']
    os.environ['PGSHOME'] = global_config['PGSHOME']
    os.environ['MRTBINDIR'] = global_config['MRTBINDIR']
    run_path = os.path.split(ifile)[0]
    # step 1: 获取hdf信息
    SD_file = SD(ifile)
    sds_obj = SD_file.select('EV_250_RefSB')
    sds_info = sds_obj.attributes()
    scales = sds_info['radiance_scales']
    offsets = sds_info['radiance_offsets']
    nbands = 2
    SD_file.end()
    date_str = os.path.split(ifile)[1].split('.')[1]
    time_str = os.path.split(ifile)[1].split('.')[2]
    date_str = date_teanslator.jd_to_cale(date_str[1:])
    year = int(date_str.split('.')[0])
    month = int(date_str.split('.')[1])
    day = int(date_str.split('.')[2])
    hour = int(time_str[0:2])
    minute = int(time_str[2:])
    date = '%d/%d/%d %d:%d:00' % (year, month, day, hour, minute)
    sola_position = calc_sola_position.main(center_lonlat[0], center_lonlat[1], date)
    solz = sola_position[0]
    sola = sola_position[1]
    heg_bin = os.path.join(global_config['MRTBINDIR'], 'hegtool')
    os.system('cd %s && %s -h %s > heg.log' % (run_path, heg_bin, ifile))
    info_file = os.path.join(run_path, 'HegHdr.hdr')
    if not(os.path.exists(info_file)):
        print('获取文件信息出错：%s' % ifile)
        return(0)
    else:
        lat_min = None
        lat_max = None
        lon_min = None
        lon_max = None
        pixel_x = None
        pixel_y = None
        with open(info_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if 'SWATH_LAT_MIN' in line:
                    lat_min = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LAT_MAX' in line:
                    lat_max = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LON_MIN' in line:
                    lon_min = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_LON_MAX' in line:
                    lon_max = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_X_PIXEL_RES_DEGREES' in line:
                    pixel_x = float(re.findall(r'[\d.]+', line)[0])
                elif 'SWATH_Y_PIXEL_RES_DEGREES' in line:
                    pixel_y = float(re.findall(r'[\d.]+', line)[0])
        if lat_min and lat_max and lon_min and lon_max and pixel_x and pixel_y:
            prm_file = os.path.join(run_path, 'HegSwath.prm')
            for i_band in range(nbands):
                modis_band = 'B%d' % (i_band+1)
                if (modis_band in band_need) or ('all' in band_need):
                    print('reprojection %s ...' % modis_band)
                    out_file = ifile.replace('.hdf', '_reproj_' + modis_band + '.tif')
                    if '.hdf' in ifile:
                        if os.path.exists(out_file):
                            os.remove(out_file)
                    else:
                        print('无法识别的文件格式：%s' % os.path.split(ifile)[1])
                    with open(prm_file, 'wb') as fp:
                        fp.write(b'\nNUM_RUNS = 1\n\n')
                        fp.write(b'BEGIN\n')
                        fp.write(bytes('INPUT_FILENAME = %s\n' % ifile, 'utf-8'))
                        fp.write(b'OBJECT_NAME = MODIS_SWATH_Type_L1B\n')
                        fp.write(b'FIELD_NAME = EV_250_RefSB|\n')
                        fp.write(bytes('BAND_NUMBER = %d\n' % (i_band+1), 'utf-8'))
                        fp.write(bytes('OUTPUT_PIXEL_SIZE_X = %f\n' % pixel_x, 'utf-8'))
                        fp.write(bytes('OUTPUT_PIXEL_SIZE_Y = %f\n' % pixel_y, 'utf-8'))
                        fp.write(bytes('SPATIAL_SUBSET_UL_CORNER = ( %f %f )\n' % (lat_max, lon_min), 'utf-8'))
                        fp.write(bytes('SPATIAL_SUBSET_LR_CORNER = ( %f %f )\n' % (lat_min, lon_max), 'utf-8'))
                        fp.write(b'OUTPUT_PROJECTION_TYPE = GEO\n')
                        fp.write(bytes('OUTPUT_FILENAME = %s\n' % out_file, 'utf-8'))
                        fp.write(b'OUTPUT_TYPE = GEO\n')
                        fp.write(b'END\n\n')
                    swtif_bin = os.path.join(global_config['MRTBINDIR'], 'swtif')
                    os.system('cd %s && %s -P HegSwath.prm > heg.log' % (run_path, swtif_bin))
        else:
            print('[Error] 未获得全部的所需信息')
            return(0)
        # 裁切与合并
        if 'all' in band_need:
            raster_file = ifile.replace('.hdf', '_reproj_B1.tif')
        else:
            raster_file = ifile.replace('.hdf', '_reproj_' + band_need[0] + '.tif')
        if not(os.path.exists(raster_file)):
            return('')
        out_file = raster_file.replace('.tif', '_cut.tif')
        cut_range = img_cut.main(raster_file, shp_file=shp_file, out_file=out_file)
        # 计算研究区域的平均太阳天顶角和方位角
        xsize = cut_range[1] - cut_range[0]
        ysize = cut_range[3] - cut_range[2]
        # 影像裁切
        print('裁切...')
        for i_band in range(nbands):
            modis_band = 'B%d' % (i_band+1)
            if (modis_band in band_need) or ('all' in band_need):
                raster_file = ifile.replace('.hdf', '_reproj_' + modis_band + '.tif')
                out_file = raster_file.replace('.tif', '_cut.tif')
                img_cut.main(raster_file, sub_lim=cut_range, out_file=out_file)
        # 辐射定标
        print('辐射定标...')
        raster = gdal.Open(raster_file)
        date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
        nrow = os.path.split(ifile)[1].split('.')[3]
        if satellite == 'AQUA':
            out_name = 'AQUA_MODIS_250_L2_%s_%s_00.tif' % (date_str, nrow)
        elif satellite == 'TERRA':
            out_name = 'TERRA_MODIS_250_L2_%s_%s_00.tif' % (date_str, nrow)
        else:
            print('[Error] 无法识别卫星类型')
            return(0)
        if os.path.exists(path_out_radi):
            raster_fn_out_radi = os.path.join(path_out_radi, out_name)
            driver = gdal.GetDriverByName('GTiff')
            target_ds = driver.Create(raster_fn_out_radi, xsize, ysize, nbands, gdal.GDT_Int16)
            target_ds.SetGeoTransform(raster.GetGeoTransform())
            target_ds.SetProjection(raster.GetProjectionRef())
            nan_mask = None
            for i_band in range(nbands):
                modis_band = 'B%d' % (i_band+1)
                if (modis_band in band_need) or ('all' in band_need):
                    raster_file = ifile.replace('.hdf', '_reproj_' + modis_band + '.tif')
                    out_file = raster_file.replace('.tif', '_cut.tif')
                    raster_cut = gdal.Open(out_file)
                    raster_data = (raster_cut.GetRasterBand(1).ReadAsArray()).astype(np.int)
                    if nan_mask is None:
                        nan_mask = raster_data > 65530
                    raster_data[nan_mask] = -9999
                    target_ds.GetRasterBand(i_band+1).WriteArray(raster_data)
                    target_ds.GetRasterBand(i_band+1).SetNoDataValue(-9999)
                    raster_cut = None
            target_ds = None
        else:
            print('[Warning] without radiation-correction output!')
        # 大气校正
        print('大气校正 ...')
        driver = gdal.GetDriverByName('GTiff')
        date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
        nrow = os.path.split(ifile)[1].split('.')[3]
        raster_fn_out_6s = os.path.join(path_out_6s, out_name)
        target_ds = driver.Create(raster_fn_out_6s, xsize, ysize, nbands, gdal.GDT_Int16)
        target_ds.SetGeoTransform(raster.GetGeoTransform())
        target_ds.SetProjection(raster.GetProjectionRef())
        nan_mask = None
        for i_band in range(nbands):
            modis_band = 'B%d' % (i_band+1)
            if (modis_band in band_need) or ('all' in band_need):
                raster_file = ifile.replace('.hdf', '_reproj_' + modis_band + '.tif')
                out_file = raster_file.replace('.tif', '_cut.tif')
                raster_cut = gdal.Open(out_file)
                raster_data = raster_cut.GetRasterBand(1).ReadAsArray()
                if nan_mask is None:
                    nan_mask = raster_data > 65530
                raster_data = raster_data.astype(float)
                raster_data = scales[i_band] * (raster_data - offsets[i_band])
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
                wave_index = i_band + 42
                data_tmp = (arms_corr(raster_data, mtl_coef, wave_index) * 10000).astype(np.int)
                data_tmp[nan_mask] = -9999
                target_ds.GetRasterBand(i_band+1).WriteArray(data_tmp)
                target_ds.GetRasterBand(i_band+1).SetNoDataValue(-9999)
                raster_cut = None
        target_ds = None
        raster = None
        # 删除过程文件
        for item in modis_band_list:
            file_name = ifile.replace('.hdf', '_reproj_' + item + '.tif')
            file_name_met = file_name.replace('.tif', '.tif.met')
            file_name_cut = ifile.replace('.hdf', '_reproj_' + item + '_cut.tif')
            if os.path.exists(file_name) and os.path.exists(file_name_met):
                os.remove(file_name)
                os.remove(file_name_met)
            if os.path.exists(file_name_cut):
                os.remove(file_name_cut)
        log_list = ['heg.log', 'swtif.log', 'HegSwath.prm', 'HegHdr.hdr', 'hegtool.log']
        for item in log_list:
            file_name = os.path.join(run_path, item)
            if os.path.exists(file_name):
                os.remove(file_name)
        return(raster_fn_out_6s)
