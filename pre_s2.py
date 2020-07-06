# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: 哨兵2A数据预处理
@Date: 2019-07-27 16:24:55
'''
import numpy as np
from osgeo import gdal, osr
import os
import re
import simplejson
import zipfile
import shutil
import xml.etree.ElementTree as ET
import platform

from utils import img_cut
from utils import calc_sola_position
from utils import coord_trans


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
PI = 3.14159


def parse_xml(tree, tag):
    '''
    @description: 从xml文件中读取影像相关信息
    @tree {}: element tree
    @tag {str}: 标签
    '''
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


def main(ifile, shp_file, path_out_10m, path_out_20m, path_out_60m,
        aero_type=1, target_type=2, altitude=0.01, visibility=15):
    print('文件解压...')
    fz = zipfile.ZipFile(ifile, 'r')
    for file in fz.namelist():
        fz.extract(file, os.path.split(ifile)[0])
    path_name = os.path.split(fz.namelist()[0])[0]
    path_in = os.path.join(os.path.split(ifile)[0], path_name)
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
    # 计算裁切范围-10m
    if shp_file is None:
        tif_tmp = 'tif_tmp.tif'
    else:
        tif_tmp = shp_file.replace('.shp', '_tmp.tif')
    target_ds = gdal.GetDriverByName('GTiff').Create(tif_tmp, ncols*6, nrows*6, 1, gdal.GDT_Byte)
    geo_trans = (ulx, 10, 0, uly, 0, -10)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    data_tmp = np.zeros([nrows, ncols]).astype(np.uint8)
    target_ds.GetRasterBand(1).WriteArray(data_tmp)
    target_ds = None
    if shp_file is None:
        out_file = 'tif_mask.tif'
    else:
        out_file = shp_file.replace('.shp', '_mask.tif')
    cut_range_10m = img_cut.main(tif_tmp, shp_file=shp_file, out_file=out_file)
    os.remove(out_file)
    os.remove(tif_tmp)
    # 计算裁切范围-20m
    target_ds = gdal.GetDriverByName('GTiff').Create(tif_tmp, ncols*3, nrows*3, 1, gdal.GDT_Byte)
    geo_trans = (ulx, 20, 0, uly, 0, -20)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    data_tmp = np.zeros([nrows, ncols]).astype(np.uint8)
    target_ds.GetRasterBand(1).WriteArray(data_tmp)
    target_ds = None
    cut_range_20m = img_cut.main(tif_tmp, shp_file=shp_file, out_file=out_file)
    os.remove(out_file)
    os.remove(tif_tmp)
    # 计算裁切范围-60m
    target_ds = gdal.GetDriverByName('GTiff').Create(tif_tmp, ncols, nrows, 1, gdal.GDT_Byte)
    geo_trans = (ulx, 60, 0, uly, 0, -60)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    data_tmp = np.zeros([nrows, ncols]).astype(np.uint8)
    target_ds.GetRasterBand(1).WriteArray(data_tmp)
    target_ds = None
    cut_range_60m = img_cut.main(tif_tmp, shp_file=shp_file, out_file=out_file)
    os.remove(out_file)
    os.remove(tif_tmp)
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
    path_in_short = os.path.split(path_in)[1]
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
    path_data = os.path.join(path_in, 'GRANULE', path_xml, 'IMG_DATA')
    data_list = os.listdir(path_data)
    # 首次循环：识别波段
    for item in data_list:
        if 'B01' in item:
            name_b1 = item
            break
    name_10m = [name_b1.replace('B01','B02'), name_b1.replace('B01','B03'), name_b1.replace('B01','B04'),
                name_b1.replace('B01','B08')]
    name_20m = [name_b1.replace('B01','B02'), name_b1.replace('B01','B03'), name_b1.replace('B01','B04'),
                name_b1.replace('B01','B05'), name_b1.replace('B01','B06'), name_b1.replace('B01','B07'),
                name_b1.replace('B01','B08'), name_b1.replace('B01','B8A'), name_b1.replace('B01','B11'),
                name_b1.replace('B01','B12')]
    name_60m = [name_b1, name_b1.replace('B01','B02'), name_b1.replace('B01','B03'), name_b1.replace('B01','B04'),
                name_b1.replace('B01','B05'), name_b1.replace('B01','B06'), name_b1.replace('B01','B07'),
                name_b1.replace('B01','B08'), name_b1.replace('B01','B8A'), name_b1.replace('B01','B09'),
                name_b1.replace('B01','B10'), name_b1.replace('B01','B11'), name_b1.replace('B01','B12')]
    # 10m波段
    print('10m ...')
    file_out_short = os.path.split(path_in)[1]
    file_out_split = file_out_short.split('_')
    date_str = file_out_split[2].replace('T', '')
    date_str = str(int(date_str) + 80000) # 转换为北京时间
    nrow = file_out_split[3].replace('N', '')
    npath = file_out_split[4].replace('R', '')
    if file_out_split[0] == 'S2A':
        satellite_code = 'A'
    elif file_out_split[0] == 'S2B':
        satellite_code = 'B'
    else:
        satellite_code = ''
    name_short = 'Sentinel2%s_MSI_10_L2_%s_%s_%s.tif' % (satellite_code, date_str, nrow, npath)
    file_out = os.path.join(path_out_10m, name_short)
    nbands = len(name_10m)
    xsize = cut_range_10m[1] - cut_range_10m[0]
    ysize = cut_range_10m[3] - cut_range_10m[2]
    target_ds = gdal.GetDriverByName('GTiff').Create(file_out, xsize, ysize, nbands, gdal.GDT_Int16)
    geo_trans = (ulx+cut_range_10m[0]*10, 10, 0, uly-cut_range_10m[2]*10, 0, -10)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    nband = 0
    for item in name_10m:
        raster = gdal.Open(os.path.join(path_data, item))
        data = raster.GetRasterBand(1).ReadAsArray()
        nan_mask = data == 0
        raster = None
        radi_cali = data.astype(float) / 10000
        # 大气校正
        mtl_coef = {
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
        wave_index = item[-7:-4]
        res_atms_corr = arms_corr(radi_cali, mtl_coef, wave_index)
        # 裁切
        res_atms_corr = res_atms_corr[cut_range_10m[2]:cut_range_10m[3], cut_range_10m[0]:cut_range_10m[1]]
        res_atms_corr = res_atms_corr * 10000
        res_atms_corr[nan_mask] = -9999
        target_ds.GetRasterBand(nband+1).WriteArray((res_atms_corr).astype(np.int))
        target_ds.GetRasterBand(nband+1).SetNoDataValue(-9999)
        nband = nband + 1
        print('%s处理完毕...' % wave_index)
    target_ds = None
    # 20m波段
    print('20m ...')
    name_short_20m = name_short.replace('_10_', '_20_')
    file_out = os.path.join(path_out_20m, name_short_20m)
    nbands = len(name_20m)
    xsize = cut_range_20m[1] - cut_range_20m[0]
    ysize = cut_range_20m[3] - cut_range_20m[2]
    target_ds = gdal.GetDriverByName('GTiff').Create(file_out, xsize, ysize, nbands, gdal.GDT_Int16)
    geo_trans = (ulx+cut_range_20m[0]*20, 20, 0, uly-cut_range_20m[2]*20, 0, -20)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    nband = 0
    for item in name_20m:
        raster = gdal.Open(os.path.join(path_data, item))
        data = raster.GetRasterBand(1).ReadAsArray()
        # 降采样
        if item in name_10m:
            data_resample = np.zeros([int(data.shape[0]/2), int(data.shape[1]/2)])
            for i in range(2):
                for j in range(2):
                    data_resample = data_resample + data[i::2, j::2]
            data = data_resample / 4
        nan_mask = data == 0
        raster = None
        radi_cali = data.astype(float) / 10000
        # 大气校正
        mtl_coef = {
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
        wave_index = item[-7:-4]
        res_atms_corr = arms_corr(radi_cali, mtl_coef, wave_index)
        # 裁切
        res_atms_corr = res_atms_corr[cut_range_20m[2]:cut_range_20m[3], cut_range_20m[0]:cut_range_20m[1]]
        res_atms_corr = res_atms_corr * 10000
        res_atms_corr[nan_mask] = -9999
        target_ds.GetRasterBand(nband+1).WriteArray((res_atms_corr).astype(np.int))
        target_ds.GetRasterBand(nband+1).SetNoDataValue(-9999)
        nband = nband + 1
        print('%s处理完毕...' % wave_index)
    target_ds = None
    # 60m波段
    print('60m ...')
    name_short_60m = name_short.replace('_10_', '_60_')
    file_out = os.path.join(path_out_60m, name_short_60m)
    nbands = len(name_60m)
    xsize = cut_range_60m[1] - cut_range_60m[0]
    ysize = cut_range_60m[3] - cut_range_60m[2]
    target_ds = gdal.GetDriverByName('GTiff').Create(file_out, xsize, ysize, nbands, gdal.GDT_Int16)
    geo_trans = (ulx+cut_range_60m[0]*60, 60, 0, uly-cut_range_60m[2]*60, 0, -60)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    nband = 0
    for item in name_60m:
        raster = gdal.Open(os.path.join(path_data, item))
        data = raster.GetRasterBand(1).ReadAsArray()
        # 降采样
        if item in name_10m:
            data_resample = np.zeros([int(data.shape[0]/6), int(data.shape[1]/6)])
            for i in range(6):
                for j in range(6):
                    data_resample = data_resample + data[i::6, j::6]
            data = data_resample / 36
        elif item in name_20m:
            data_resample = np.zeros([int(data.shape[0]/3), int(data.shape[1]/3)])
            for i in range(3):
                for j in range(3):
                    data_resample = data_resample + data[i::3, j::3]
            data = data_resample / 9
        nan_mask = data == 0
        raster = None
        radi_cali = data.astype(float) / 10000
        # 大气校正
        mtl_coef = {
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
        wave_index = item[-7:-4]
        if wave_index == 'B10':
            res_atms_corr = radi_cali
        else:
            res_atms_corr = arms_corr(radi_cali, mtl_coef, wave_index)
        # 裁切
        res_atms_corr = res_atms_corr[cut_range_60m[2]:cut_range_60m[3], cut_range_60m[0]:cut_range_60m[1]]
        res_atms_corr = res_atms_corr * 10000
        res_atms_corr[nan_mask] = -9999
        target_ds.GetRasterBand(nband+1).WriteArray((res_atms_corr).astype(np.int))
        target_ds.GetRasterBand(nband+1).SetNoDataValue(-9999)
        nband = nband + 1
        print('%s处理完毕...' % wave_index)
    target_ds = None
    # 删除解压文件
    shutil.rmtree(path_in)
