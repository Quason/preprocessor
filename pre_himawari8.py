# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: Himawari 8(2km)数据预处理
@Date: 2019-07-29 14:14:43
'''
from osgeo import gdal, osr
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
import numpy as np
import re
import h5py
import os
import json

import utils.calc_sola_position as calc_sola_position

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = json.load(fp)

wavelength_center = [0.471, 0.51, 0.639, 0.857, 1.61, 2.26, 3.885, 6.243, 6.941,
                    7.347, 8.593, 9.637, 10.407, 11.24, 12.381, 13.281]
wavelength_width = [0.04, 0.02, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
tau_r = [0.1845, 0.13194, 0.05262, 0.0160511, 0.00127373, 0.000329065, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0] # 瑞利光学厚度
path_6s = global_config['path_6s']
PI = 3.14159


month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def get_hima_info(file_L1B, center_location, sub_range):
    '''
    @description: 从L1B文件中获取数据信息
    @file_L1B {str} L1B数据文件
    @file_info {str} 卫星常规信息文件
    @center_location {[lon, lat]} 研究区域中心经纬度
    @sub_range {[x_start, x_end, y_start, y_end]} 研究区域范围
    @return {dict} info文件
    '''
    info_dict = {}
    f_L1B = h5py.File(file_L1B, 'r')
    data_salz = f_L1B['SAZ']
    data_salz = data_salz[:]
    salz_cut = data_salz[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]] * 0.01
    info_dict['salz'] = np.mean(salz_cut)
    data_sala = f_L1B['SAA']
    data_sala = data_sala[:]
    sala_cut = data_sala[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]] * 0.01
    info_dict['sala'] = np.mean(sala_cut)
    data_solz = f_L1B['SOZ']
    data_solz = data_solz[:]
    solz_cut = data_solz[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]] * 0.01
    info_dict['solz'] = np.mean(solz_cut)
    data_sola = f_L1B['SOA']
    data_sola = data_sola[:]
    sola_cut = data_sola[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]] * 0.01
    info_dict['sola'] = np.mean(sola_cut)
    info_dict['location'] = center_location
    file_short = os.path.split(file_L1B)[1]
    str_date = file_short.split('_')[2]
    if len(str_date) != 8:
        print('无法识别的数据名')
        return(0)
    info_dict['month'] = int(str_date[4:6])
    info_dict['day'] = int(str_date[6:8])
    return(info_dict)


def radi_arms_corr(raster_data, mtl_coef, wave_index):
    '''
    @description: 辐射定标和6S大气校正
    @raster_data {numpy array} 辐射定标后的数据
    @mtl_coef 传感器信息，包括以下内容：
        -- altitude {float} 海拔(km)
        -- visibility {float} 能见度(km)
        -- type_atms {int} 大气模型 0 无; 1 热带; 2 中纬度夏季; 3 中纬度冬季;
            4 亚热带夏季; 5 亚热带冬季; 6 US 62
        -- type_aero {int} 气溶胶模型 0 无; 1 大陆; 2 近海; 3 城市; 5 沙漠;
            6 有机质燃烧; 7 平流层
        -- tau 瑞利光学厚度
    @wave_index {int} 波段下标(从0开始)
    @return {numpy array} 大气校正后的栅格数据
    '''
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
        target_type = mtl_coef['target_type']
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
        fp.write('%.1f\n' % mtl_coef['visibility']) # 需根据条件改变
        # 高程
        fp.write('%.3f\n' % -mtl_coef['altitude']) # 需根据条件改变
        # 传感器类型
        fp.write('%d\n' % -1000)
        # 波段号(-2表示自己输入范围)
        fp.write('%d\n' % -2)
        wavelength_down = wavelength_center[wave_index] - wavelength_width[wave_index]/2
        wavelength_up = wavelength_center[wave_index] + wavelength_width[wave_index]/2
        fp.write('%.4f %.4f\n' % (wavelength_down, wavelength_up))
        # 其余参数
        fp.write('%d\n' % 0) # 无方向反射
        fp.write('%d\n' % 0) # 朗伯体假设
        fp.write('%d\n' % target_type) # 地物类型
        fp.write('%d\n' % 0) # 进行大气校正
        fp.write('%.2f\n' % 0.01) # 默认反射率
        fp.write('%d\n' % 5) # 除了0,1,2外任意设置
    os.system('cd %s && ./sixs_Lin<./in.txt>log.txt' % (path_6s))
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
                atms_corr_data = atms_corr_data / PI / np.exp(-0.5*mtl_coef['tau']/np.cos(SOLZ/180*PI))
                break
    return(atms_corr_data)


def main(file_L1B, alt, visib, aero_type, target_type, target_range, path_out):
    '''
    @description: 文件裁切
    @file_L1B {str} 原始数据文件
    @alt {float} 海拔(km)
    @visib {float} 能见度(km)
    @aero_type {int} 气溶胶类型
    @target_type {int} 地物类型 1:植被 2:清洁水体 3:沙漠 4:湖泊水体
    @target_range {[lon_start, lon_end, lat_start, lat_end]} 目标插值网格范围
    @file_out {str} 输出文件
    '''
    # 计算影像日期
    nbands = 16
    name_out = os.path.split(file_L1B)[1]
    date = name_out.split('_')[2] + name_out.split('_')[3]
    date = str(int(date) + 800)
    name_out = 'Himawari8_AHI_2000_L2_%s00_00_00.tif' % (date)
    file_out = os.path.join(path_out, name_out)
    f_info = h5py.File(file_L1B, 'r')
    data_lon = f_info['longitude']
    data_lat = f_info['latitude']
    # 计算目标网格对应的起止行列号
    x_start = round((target_range[0] - data_lon[0]) / 0.02)
    x_end = round((target_range[1] - data_lon[0]) / 0.02)
    y_start = round((data_lat[0] - target_range[3]) / 0.02)
    y_end = round((data_lat[0] - target_range[2]) / 0.02)
    xsize_target = x_end - x_start
    ysize_target = y_end - y_start
    sub_range = [int(y_start), int(y_end), int(x_start), int(x_end)]
    # 读取原始数据
    rrs_interp = np.zeros([int(ysize_target), int(xsize_target), nbands])
    f_Lr = h5py.File(file_L1B, 'r')
    center_location = [(target_range[0]+target_range[1])/2, (target_range[2]+target_range[3])/2]
    info_dict = get_hima_info(file_L1B, center_location, sub_range)
    info_dict['altitude'] = alt
    info_dict['visibility'] = visib
    info_dict['target_type'] = target_type
    info_dict['aero_type'] = aero_type
    for i in range(nbands):
        if i < 6:
            band_index = 'albedo_%02d' % (i+1)
            Lr = f_Lr[band_index]
            Lr = Lr[:]
            Lr_cut = Lr[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]]
            Lr_cut = Lr_cut.astype(float) * 1e-4 # 辐射定标
            info_dict['tau'] = tau_r[i]
            cos_solz = np.cos(info_dict['solz']/180*PI)
            rrs_interp[:, :, i] = radi_arms_corr(Lr_cut/cos_solz, mtl_coef=info_dict, wave_index=i) * 10000
            # rrs_interp[:, :, i] = Lr_cut
        else:
            band_index = 'tbb_%02d' % (i+1)
            Lr = f_Lr[band_index]
            Lr = Lr[:]
            Lr_cut = Lr[sub_range[0]:sub_range[1], sub_range[2]:sub_range[3]]
            rrs_interp[:, :, i] = Lr_cut * 0.01 + 273.15 # 辐射定标
        print('第%d波段处理完毕' % (i+1))
    # 数据保存
    rrs_interp = rrs_interp.astype(np.int)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(file_out, rrs_interp.shape[1], rrs_interp.shape[0],
                                rrs_interp.shape[2], gdal.GDT_UInt16)
    degree_step = 0.02
    geo_trans = (data_lon[x_start], degree_step, 0, data_lat[y_start], 0, -degree_step)
    target_ds.SetGeoTransform(geo_trans)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetProjection(proj_ref)
    for nband in range(nbands):
        data_tmp = rrs_interp[:, :, nband]
        target_ds.GetRasterBand(nband+1).WriteArray(data_tmp)
    target_ds = None
    f_Lr.close()
    f_info.close()
