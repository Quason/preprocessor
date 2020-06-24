# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: VIIRS 数据处理
@Date: 2019-08-02 23:22:35
'''
import os
import sys
import re
import simplejson
import numpy as np
import time
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


viirs_band_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M8', 'M9',
                    'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16']

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


def main_sa_position(ifile, shp_file, cut_range=None):
    '''
    @description: 提取卫星天顶角和方位角信息
    @ifile {str}: L1B swath文件 
    @shp_file {str}: 研究区域矢量文件
    @cut_range {[lon_min, lon_max, lat_min, lat_max}: 裁切经纬度范围
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
            if not(cut_range is None):
                lat_min = cut_range[2]
                lat_max = cut_range[3]
                lon_min = cut_range[0]
                lon_max = cut_range[1]
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
                fp.write(b'OBJECT_NAME = VIIRS_EV_750M_SDR\n')
                fp.write(b'FIELD_NAME = SatelliteZenithAngle|\n')
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
                fp.write(b'OBJECT_NAME = VIIRS_EV_750M_SDR\n')
                fp.write(b'FIELD_NAME = SatelliteAzimuthAngle|\n')
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
        salz = np.mean(salz_all)
        sala_file = ifile.replace('.hdf', '_SALA.tif')
        raster = gdal.Open(sala_file)
        sala_all = raster.GetRasterBand(1).ReadAsArray()
        sala_all = sala_all[cut_range[2]:cut_range[3], cut_range[0]:cut_range[1]]
        sala = np.mean(sala_all)
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


def main(ifile, shp_file, center_lonlat, salz, sala, cut_range=None,
        aerotype=1, altitude=0.01, visibility=15, band_need=['all'],
        path_out=None):
    '''
    @description: 
    @ifile {str}: L1B swath文件 
    @shp_file {str}: 研究区域矢量文件
    @center_lonlat {[lon, lat]}: 中心经纬度
    @aerotype {int}: 气溶胶类型(默认大陆型)
    @altitude {float}: 海拔(km)
    @visibility {float}: 能见度(km)
    @return: 
    '''
    # 设置环境变量
    os.environ['MRTDATADIR'] = global_config['MRTDATADIR']
    os.environ['PGSHOME'] = global_config['PGSHOME']
    os.environ['MRTBINDIR'] = global_config['MRTBINDIR']
    run_path = os.path.split(ifile)[0]
    # step 1: 获取hdf信息
    nbands = 16
    SD_file = SD(ifile)
    scales = []
    offsets = []
    for i_band in range(nbands):
        if i_band+1 == 9:
            sds_obj = SD_file.select('Reflectance_M%d' % (i_band+1))
        elif i_band+1 >= 12:
            sds_obj = SD_file.select('BrightnessTemperature_M%d' % (i_band+1))
        else:
            sds_obj = SD_file.select('Radiance_M%d' % (i_band+1))
        sds_info = sds_obj.attributes()
        if 'Scale' in sds_info:
            scales.append(sds_info['Scale'])
        else:
            scales.append(1)
        if 'Offset' in sds_info:
            offsets.append(sds_info['Offset'])
        else:
            offsets.append(0)
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
            if not(cut_range is None):
                lat_min = cut_range[2]
                lat_max = cut_range[3]
                lon_min = cut_range[0]
                lon_max = cut_range[1]
        if lat_min and lat_max and lon_min and lon_max and pixel_x and pixel_y:
            prm_file = os.path.join(run_path, 'HegSwath.prm')
            # 反射波段值重投影
            for i_band in range(nbands):
                viirs_band = 'M%d' % (i_band+1)
                if (viirs_band in band_need) or ('all' in band_need):
                    print('reprojection %s ...' % viirs_band)
                    out_file = ifile.replace('.hdf', '_reproj_' + viirs_band + '.tif')
                    if '.hdf' in ifile:
                        if os.path.exists(out_file):
                            os.remove(out_file)
                    else:
                        print('无法识别的文件格式：%s' % os.path.split(ifile)[1])
                    with open(prm_file, 'wb') as fp:
                        fp.write(b'\nNUM_RUNS = 1\n\n')
                        fp.write(b'BEGIN\n')
                        fp.write(bytes('INPUT_FILENAME = %s\n' % ifile, 'utf-8'))
                        fp.write(b'OBJECT_NAME = VIIRS_EV_750M_SDR\n')
                        if viirs_band == 'M9':
                            fp.write(bytes('FIELD_NAME = Reflectance_M%d|\n' % (i_band+1), 'utf-8'))
                        elif viirs_band == 'M15' or viirs_band == 'M16':
                            fp.write(bytes('FIELD_NAME = BrightnessTemperature_M%d|\n' % (i_band+1), 'utf-8'))
                        else:
                            fp.write(bytes('FIELD_NAME = Radiance_M%d|\n' % (i_band+1), 'utf-8'))
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
        # 裁切与合并
        if 'all' in band_need:
            raster_file = ifile.replace('.hdf', '_reproj_M1.tif')
        else:
            raster_file = ifile.replace('.hdf', '_reproj_' + band_need[0] + '.tif')
        out_file = raster_file.replace('.tif', '_cut.tif')
        cut_range = img_cut.main(raster_file, shp_file=shp_file, out_file=out_file)
        xsize = cut_range[1] - cut_range[0]
        ysize = cut_range[3] - cut_range[2]
        data_join = np.zeros([ysize, xsize, nbands])
        print('大气校正 ...')
        for i_band in range(nbands):
            viirs_band = 'M%d' % (i_band+1)
            if (viirs_band in band_need) or ('all' in band_need):
                raster_file = ifile.replace('.hdf', '_reproj_' + viirs_band + '.tif')
                out_file = raster_file.replace('.tif', '_cut.tif')
                img_cut.main(raster_file, sub_lim=cut_range, out_file=out_file)
                # 辐射定标和大气校正
                raster = gdal.Open(out_file)
                raster_data = raster.GetRasterBand(1).ReadAsArray()
                raster_data = raster_data.astype(float)
                mask = raster_data >= 65530
                raster_data = scales[i_band] * raster_data + offsets[i_band]
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
                if i_band <= 11:
                    wave_index = i_band + 149
                else:
                    wave_index = None
                if not(wave_index is None):
                    if viirs_band == 'M9':
                        tmp = raster_data
                    else:
                        tmp = arms_corr(raster_data, mtl_coef, wave_index)
                    # tmp = raster_data
                    tmp[mask] = 65535
                    data_join[:, :, i_band] = tmp
                else:
                    tmp = raster_data
                    tmp[mask] = 65535
                    data_join[:, :, i_band] = tmp
        driver = gdal.GetDriverByName('GTiff')
        if 'all' in band_need:
            raster_file = ifile.replace('.hdf', '_reproj_M1_cut.tif')
        else:
            raster_file = ifile.replace('.hdf', '_reproj_' + band_need[0] + '_cut.tif')
        raster = gdal.Open(raster_file)
        # 输出文件名
        date_str = '%d%02d%02d%02d%02d%02d' % (year, month, day, hour+8, minute, 0)
        nrow = os.path.split(ifile)[1].split('.')[3]
        out_name = 'NPP_VIIRS_750_L2_%s_%s_00.tif' % (date_str, nrow)
        if path_out is None:
            raster_fn_out = os.path.join(os.path.split(ifile)[0], out_name)
        else:
            raster_fn_out = os.path.join(path_out, out_name)
        target_ds = driver.Create(raster_fn_out, xsize, ysize, nbands, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(raster.GetGeoTransform())
        target_ds.SetProjection(raster.GetProjectionRef())
        for i in range(nbands):
            if i < 11:
                target_ds.GetRasterBand(i+1).WriteArray((data_join[:, :, i]*10000).astype(np.int))
            else:
                target_ds.GetRasterBand(i+1).WriteArray((data_join[:, :, i]*100).astype(np.int))
            band = target_ds.GetRasterBand(i+1)
            band.SetNoDataValue(65535)
        target_ds = None
        raster = None
        # 删除过程文件
        for item in viirs_band_list:
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
        return(raster_fn_out)


def stack_rgb(ifile):
    raster = gdal.Open(ifile)
    xsize = raster.GetRasterBand(1).ReadAsArray().shape[1]
    ysize = raster.GetRasterBand(1).ReadAsArray().shape[0]
    nbands = 6
    if '.tif' in ifile:
        raster_fn_out = ifile.replace('.tif', '_rgb.tif')
    else:
        print('无法识别的文件名: %s' % os.path.split(ifile)[1])
        return(0)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(raster_fn_out, xsize, ysize, nbands, gdal.GDT_Float64)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjectionRef())
    target_ds.GetRasterBand(1).WriteArray(raster.GetRasterBand(2).ReadAsArray())
    target_ds.GetRasterBand(2).WriteArray(raster.GetRasterBand(4).ReadAsArray())
    target_ds.GetRasterBand(3).WriteArray(raster.GetRasterBand(5).ReadAsArray())
    target_ds.GetRasterBand(4).WriteArray(raster.GetRasterBand(7).ReadAsArray())
    target_ds.GetRasterBand(5).WriteArray(raster.GetRasterBand(9).ReadAsArray()) # 卷云波段
    target_ds.GetRasterBand(6).WriteArray(raster.GetRasterBand(10).ReadAsArray())
    
    target_ds = None
    raster = None


if __name__ == '__main__':
    path_in = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\201805\NPP\L1B'
    path_out = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\201805\NPP\L2'
    shp_file = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\shp\taihu-wgs84\taihu.shp'
    center_lonlat = [120.17, 31.2]
    cut_range = [119.8, 120.7, 30.8, 31.6]

    name_list = os.listdir(path_in)
    cnt_sum = 0
    cnt = 0
    for item in name_list:
        if '.hdf' in item:
            cnt_sum += 1
    for item in name_list:
        if '.hdf' in item:
            sensor_position = main_sa_position(ifile=os.path.join(path_in, item),
                                                cut_range=cut_range, shp_file=shp_file)
            salz = sensor_position['salz']
            sala = sensor_position['sala']
            if sala < 0:
                sala = sala + 360
            main(ifile=os.path.join(path_in, item),
                shp_file=shp_file, center_lonlat=center_lonlat, aerotype=1,
                altitude=0.01, visibility=15, salz=salz, sala=sala, cut_range=cut_range,
                band_need=['M2', 'M4', 'M5', 'M7', 'M9', 'M10'], path_out=path_out)
            ifile = os.path.join(path_out, item.replace('.hdf', '_cut_corr.tif'))
            stack_rgb(ifile)
            cnt += 1
            print('第%d个已经完成(共%d个)' % (cnt, cnt_sum))
