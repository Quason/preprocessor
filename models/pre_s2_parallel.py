"""water environment index inversion based on Sentinel2
"""
import os
import platform
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
import sys
import multiprocessing

import numpy as np
import simplejson
import cv2.cv2 as cv
import skimage.io
from osgeo import gdal, osr

from utils import calc_sola_position, coord_trans, img_cut

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
if platform.platform()[0:3] == 'Lin':
    jsonFile = os.path.join(rootpath, 'global_configure.json')
else:
    jsonFile = os.path.join(rootpath, 'global_configure_local.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


wave_center = {'B01':0.443, 'B02':0.49, 'B03':0.56, 'B04':0.665, 'B05':0.705,
                'B06':0.74, 'B07':0.783, 'B08':0.842, 'B8A':0.865, 'B09':0.945,
                'B10':1.375, 'B11':1.61, 'B12':2.19}
wave_width = {'B01':0.021, 'B02':0.066, 'B03':0.036, 'B04':0.031, 'B05':0.015,
                'B06':0.015, 'B07':0.02, 'B08':0.106, 'B8A':0.021, 'B09':0.02,
                'B10':0.031, 'B11':0.091, 'B12':0.175}
tau_r = {'B01':0.23546, 'B02':0.15546, 'B03':0.0900236, 'B04':0.0447581, 'B05':0.0353252,
                'B06':0.0290359, 'B07':0.0231095, 'B08':0.017236, 'B8A':0.0154609, 'B09':0.0108254,
                'B10':0.00239831, 'B11':0.00127373, 'B12':0.000371274}
bands_10 = ['B02', 'B03', 'B04', 'B08']
bands_20 = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
bands_60 = ['B01', 'B09', 'B10']
PI = 3.14159


def raster2tif(raster, geo_trans, proj_ref, file_out, type='float'):
    """save ndarray as GeoTiff
    """
    driver = gdal.GetDriverByName('GTiff')
    if len(raster.shape) == 2:
        nbands = 1
    else:
        nbands = raster.shape[2]
    if type == 'uint8':
        target_ds = driver.Create(
            file_out, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Byte)
        mask_value = None
    elif type == 'int':
        target_ds = driver.Create(
            file_out, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Int16)
        mask_value = -9999
    else:
        target_ds = driver.Create(
            file_out, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Float32)
        mask_value = -9999
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    if nbands == 1:
        target_ds.GetRasterBand(1).WriteArray(raster)
        if mask_value is not None:
            target_ds.GetRasterBand(1).SetNoDataValue(mask_value)
    else:
        for i in range(nbands):
            target_ds.GetRasterBand(i+1).WriteArray(raster[:,:,i])
            if mask_value is not None:
                target_ds.GetRasterBand(i+1).SetNoDataValue(mask_value)
    target_ds = None


def parse_xml(tree, tag): 
    tag_index = 0
    child = tree.findall(tag[tag_index][0])
    tag_index += 1
    while tag_index < len(tag):
        if len(list(child[0])) != 0:
            child = child[0].findall(tag[tag_index][0])
            attri = tag[tag_index][1]
            if not(attri is None):
                del_item = []
                for item in child:
                    for key in attri:
                            if key in item.attrib:
                                if item.attrib[key] != attri[key]:
                                    del_item.append(item)
                for item in del_item:
                    child.remove(item)
            tag_index += 1
        else:
            return(None)
    return child[0].text


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
        SOLZ = float(mtl_coef['solz'])
        SOLA = float(mtl_coef['sola'])
        PHV = float(mtl_coef['sala'])
        THV = float(mtl_coef['salz'])
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
        wave_lower = wave_center[wave_index] - wave_width[wave_index]/2
        wave_upper = wave_center[wave_index] + wave_width[wave_index]/2
        fp.write('%.3f %.3f\n' % (wave_lower, wave_upper))
        # 其余参数
        fp.write('%d\n' % 0) # 无方向反射
        fp.write('%d\n' % 0) # 朗伯体假设
        fp.write('%d\n' % mtl_coef['target_type'])
        fp.write('%d\n' % 0) # 进行大气校正
        fp.write('%.2f\n' % 0.01) # 默认反射率
        fp.write('%d\n' % 5) # 除了0,1,2外任意设置
    if platform.platform()[0:3] == 'Lin':
        os.system('cd %s && ./sixs_Lin<./in.txt>log.txt' % (path_6s))
    else:
        driver = path_6s[0:2]
        os.system('%s && cd %s && sixs.exe<./in.txt>log.txt' % (driver, path_6s))
    with open(os.path.join(path_6s, 'sixs.out'), 'r') as fp:
        for line in fp.readlines():
            if 'coefficients xap xb xc' in line:
                coefAll = re.findall(r'[\.\d]+', line)
                # y=xa*(measured radiance)-xb;  acr=y/(1.+xc*y)
                xa = float(coefAll[0])
                xb = float(coefAll[1])
                xc = float(coefAll[2])
                y = xa*raster_data - xb
                atms_corr_data = y/(1.0+xc*y)
                atms_corr_data = atms_corr_data / PI / np.exp(-0.5*tau_r[wave_index]/np.cos(SOLZ/180*PI))
                break
    return(atms_corr_data)


def radi_atms(item, data_dir, dst_dir, sixs_config):
    wave_index = re.findall(r'B[\dA]+', item)
    if len(wave_index) == 1:
        print('preprocess: %s' % wave_index[0])
        wave_index = wave_index[0]
    else:
        return 0
    raster = gdal.Open(os.path.join(data_dir, item))
    dataset = raster.GetRasterBand(1)
    if wave_index in bands_20:
        data = dataset.ReadAsArray(
            buf_xsize=raster.RasterXSize * 2,
            buf_ysize=raster.RasterYSize * 2
        )
    elif wave_index in bands_60:
        data = dataset.ReadAsArray(
            buf_xsize=raster.RasterXSize * 6,
            buf_ysize=raster.RasterYSize * 6
        )
    else:
        data = dataset.ReadAsArray()
    nan_mask = data == 0
    radi_cali = data.astype(float) / 10000
    mtl_coef = {
        'altitude': sixs_config['altitude'],
        'visibility': sixs_config['visibility'],
        'aero_type': sixs_config['aero_type'],
        'target_type': sixs_config['target_type'],
        'location': sixs_config['location'],
        'month': sixs_config['month'],
        'day': sixs_config['day'],
        'solz': sixs_config['solz'],
        'sola': sixs_config['sola'],
        'salz': sixs_config['salz'],
        'sala': sixs_config['sala']
    }
    if wave_index == 'B10':
        res_atms_corr = radi_cali
    else:
        res_atms_corr = arms_corr(radi_cali, mtl_coef, wave_index) * 10000.0
    geo_trans_dst = list(raster.GetGeoTransform())
    # 20m和60m的数据需要重采样
    if wave_index in bands_20:
        geo_trans_dst[1] /= 2
        geo_trans_dst[5] /= 2
    elif wave_index in bands_60:
        geo_trans_dst[1] /= 6
        geo_trans_dst[5] /= 6
    file_out = os.path.join(dst_dir, item).replace('.jp2', '') + '_L2.tif'
    res_atms_corr = res_atms_corr.astype(np.int)
    res_atms_corr[nan_mask] = -9999
    raster2tif(res_atms_corr, geo_trans_dst, raster.GetProjection(), file_out, 'int')
    raster = None


def main(ifile, aero_type=1, target_type=4, altitude=0.01, visibility=15,
        dst_dir=None, cpu_cores=1):
    """main function

    Args:
        ifile (string): source L1C file
        aero_type (int, optional) 0无 1大陆 2近海 3城市
        target_type (int, optional) 目标地物类型 1植被 2清洁水体 3沙漠 4浑浊水体
        altitude (float, optional) 海拔
        visibility (float, optional) 能见度
    """
    dst_file = []
    if os.path.isfile(ifile):
        fz = zipfile.ZipFile(ifile, 'r')
        for file in fz.namelist():
            fz.extract(file, os.path.split(ifile)[0])
        path_name = os.path.split(fz.namelist()[0])[0]
        path_in = os.path.join(os.path.split(ifile)[0], path_name)
    else:
        path_in = ifile
    if path_in[-1] == '\\' or path_in[-1] == '/':
        path_in = path_in[0:-1]
    # 卫星经纬度范围
    path_xml = os.listdir(os.path.join(path_in, 'GRANULE'))[0]
    file_mtd = os.path.join(path_in, 'GRANULE', path_xml, 'MTD_TL.xml')
    with open(file_mtd, 'r') as fp:
        for line in fp:
            res = re.findall(r'psd-\d+', line)
            if len(res) != 0:
                psd_str = '{https://%s.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd}Geometric_Info' % res[0]
                break
    tree = ET.parse(file_mtd)
    ulx = parse_xml(tree, [[psd_str, None],
                            ['Tile_Geocoding', None],
                            ['Geoposition', None],
                            ['ULX', None]])
    ulx = int(ulx)
    uly = parse_xml(tree, [[psd_str, None],
                            ['Tile_Geocoding', None],
                            ['Geoposition', None],
                            ['ULY', None]])
    uly = int(uly)
    # EPSG代码
    epsg = parse_xml(tree, [[psd_str, None],
                            ['Tile_Geocoding', None],
                            ['HORIZONTAL_CS_CODE', None]])
    epsg = re.findall(r'\d+', epsg)
    epsg = int(epsg[0])
    # 60m分辨率的行列数
    nrows = parse_xml(tree, [[psd_str, None],
                            ['Tile_Geocoding', None],
                            ['Size', {'resolution': '60'}],
                            ['NROWS', None]])
    nrows = int(nrows)
    ncols = parse_xml(tree, [[psd_str, None],
                            ['Tile_Geocoding', None],
                            ['Size', {'resolution': '60'}],
                            ['NCOLS', None]])
    ncols = int(ncols)
    # 观测天顶角和方位角
    salz = parse_xml(tree, [[psd_str, None],
                            ['Tile_Angles', None],
                            ['Mean_Viewing_Incidence_Angle_List', None],
                            ['Mean_Viewing_Incidence_Angle', None],
                            ['ZENITH_ANGLE', None]])
    sala = parse_xml(tree, [[psd_str, None],
                            ['Tile_Angles', None],
                            ['Mean_Viewing_Incidence_Angle_List', None],
                            ['Mean_Viewing_Incidence_Angle', None],
                            ['AZIMUTH_ANGLE', None]])
    # 太阳天顶角和方位角
    path_in_short = os.path.basename(path_in)
    date_str = path_in_short.split('_')[2]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[11:13])
    second = int(date_str[13:15])
    date = '%d/%d/%d %d:%d:%d' % (year, month, day, hour, minute, second)
    center_lonlat = coord_trans.trans(epsg, 4326, ulx+(ncols/2)*60, uly-(nrows/2)*60)
    [solz, sola] = calc_sola_position.main(center_lonlat[0], center_lonlat[1], date)
    data_dir = os.path.join(path_in, 'GRANULE', path_xml, 'IMG_DATA')
    data_list = os.listdir(data_dir)
    # 辐射定标和大气校正
    sixs_config = {
        'altitude': altitude,
        'visibility': visibility,
        'aero_type': aero_type,
        'target_type': target_type,
        'location': center_lonlat,
        'month': month,
        'day': day,
        'solz': solz,
        'sola': sola,
        'salz': salz,
        'sala': sala
    }
    if cpu_cores > 1:
        task_num = int(len(data_list) / cpu_cores + 0.5)
        for i in range(task_num):
            data_list_sub = data_list[i*cpu_cores:(i+1)*cpu_cores]
            pool = multiprocessing.Pool(processes=cpu_cores)
            for item in data_list_sub:
                pool.apply_async(radi_atms, (item, data_dir, dst_dir, sixs_config))
            pool.close()
            pool.join()
            print('part %d is done!' % (i+1))
    else:
        for item in data_list:
            radi_atms(item, data_dir, dst_dir, sixs_config)
        
    # 删除解压文件
    if os.path.isfile(ifile):
        shutil.rmtree(path_in)
