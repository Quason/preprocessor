# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: GOCI数据预处理
@Date: 2019-07-29 14:14:43
'''
from osgeo import gdal, osr
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
import numpy as np
import re
import h5py
import os
import simplejson

import utils.calc_sola_position as calc_sola_position

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)

wavelength_down = [0.402, 0.433, 0.48, 0.545, 0.65, 0.675, 0.735, 0.845]
wavelength_up = [0.422, 0.453, 0.5, 0.565, 0.67, 0.685, 0.755, 0.885]
tau_r = [0.317976, 0.23546, 0.15546, 0.0933786, 0.0461493, 0.0408885, 0.0282559, 0.0154609] # 瑞利光学厚度
path_6s = global_config['path_6s']
PI = 3.14159


month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def get_goci_info(file_L1B, goci_info, center_location, sub_range):
    '''
    @description: 从L1B文件中获取数据信息
    @file_L1B {str} L1B数据文件
    @file_info {str} 卫星常规信息文件
    @center_location {[lon, lat]} 研究区域中心经纬度
    @sub_range {[x_start, x_end, y_start, y_end]} 研究区域范围
    @return {dict} info文件
    '''
    info_dict = {}
    f_info = h5py.File(goci_info, 'r')
    data_salz = f_info['THV Image Pixel Values']
    salz_cut = data_salz[sub_range[2]:sub_range[3], sub_range[0]:sub_range[1]]
    info_dict['salz'] = 90.0 - np.mean(salz_cut)
    data_sala = f_info['PHV Image Pixel Values']
    sala_cut = data_sala[sub_range[2]:sub_range[3], sub_range[0]:sub_range[1]]
    info_dict['sala'] = np.mean(sala_cut)
    f_L1B = h5py.File(file_L1B, 'r')
    center_time = f_L1B['HDFEOS']['POINTS']['Ephemeris'].attrs['Scene center time']
    center_time = center_time.decode('utf-8')
    date_split = re.findall(r'\d+', center_time)
    month_str = re.findall(r'[A-Z]+', center_time)[0]
    if month_str in month_dict:
        info_dict['month'] = month_dict[month_str]
    else:
        print('无法识别的月份代码: %s' % month_str)
    info_dict['day'] = int(date_split[0])
    date = '%s/%s/%s %s:%s:%s' % (date_split[1], month_dict[month_str], date_split[0],
                                date_split[2], date_split[3], date_split[4])
    sola_position = calc_sola_position.main(center_location[0], center_location[1], date)
    info_dict['solz'] = sola_position[0]
    info_dict['sola'] = sola_position[1]
    info_dict['location'] = center_location
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
        fp.write('%.4f %.4f\n' % (wavelength_down[wave_index], wavelength_up[wave_index]))
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
            if 'coefficients xa xb xc' in line:
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


def main(file_L1B, goci_info, alt, visib, aero_type, target_type, sub_range, target_range, path_out):
    '''
    @description: 文件裁切
    @file_L1B {str} 原始数据文件
    @goci_info {str} goci信息文件
    @alt {float} 海拔(km)
    @visib {float} 能见度(km)
    @aero_type {int} 气溶胶类型
    @target_type {int} 地物类型 1:植被 2:清洁水体 3:沙漠 4:湖泊水体
    @sub_range {[x_start, x_end, y_start, y_end]} 裁切范围
    @target_range {[lon_start, lon_end, lat_start, lat_end]} 目标插值网格范围
    @file_out {str} 输出文件
    '''
    # 计算影像日期
    name_out = os.path.split(file_L1B)[1]
    date = name_out.split('_')[4][0:8]
    hour = int(name_out.split('_')[4][8:10]) + 8
    name_out = 'COMS_GOCI_500_L2_%s%02d3000_00_00.tif' % (date, hour)
    file_out = os.path.join(path_out, name_out)
    f_info = h5py.File(goci_info, 'r')
    data_lon = f_info['Longitude Image Pixel Values']
    lon_cut = data_lon[sub_range[2]:sub_range[3], sub_range[0]:sub_range[1]]
    lon_cut_1d = lon_cut.reshape(lon_cut.shape[0]*lon_cut.shape[1])
    data_lat = f_info['Latitude Image Pixel Values']
    lat_cut = data_lat[sub_range[2]:sub_range[3], sub_range[0]:sub_range[1]]
    lat_cut_1d = lat_cut.reshape(lat_cut.shape[0]*lat_cut.shape[1])
    f_info.close()
    # 构建目标插值网格
    xsize_target = round((target_range[1]-target_range[0])*111/0.5)
    ysize_target = round((target_range[3]-target_range[2])*111/0.5)
    x1d_target = np.linspace(target_range[0], target_range[1], xsize_target)
    y1d_target = np.linspace(target_range[3], target_range[2], ysize_target)
    [xx_interp, yy_interp] = np.meshgrid(x1d_target, y1d_target)
    # 读取原始数据并插值
    rrs_interp = np.zeros([ysize_target, xsize_target, 8])
    f_Lr = h5py.File(file_L1B, 'r')
    center_location = [(target_range[0]+target_range[1])/2, (target_range[2]+target_range[3])/2]
    info_dict = get_goci_info(file_L1B, goci_info, center_location, sub_range)
    info_dict['altitude'] = alt
    info_dict['visibility'] = visib
    info_dict['target_type'] = target_type
    info_dict['aero_type'] = aero_type
    num_interp = xx_interp.shape[0] * xx_interp.shape[1]
    for i in range(8):
        band_index = 'Band ' + str(i+1) + ' Image Pixel Values'
        Lr = f_Lr['HDFEOS']['GRIDS']['Image Data']['Data Fields'][band_index]
        Lr_cut = Lr[sub_range[2]:sub_range[3], sub_range[0]:sub_range[1]]
        Lr_cut = Lr_cut.astype(float) * 1e-6 # 辐射定标
        Lr_cut[0, :] = -999
        Lr_cut[-1, :] = -999
        Lr_cut[:, 0] = -999
        Lr_cut[:, -1] = -999
        Lr_cut_1d = Lr_cut.reshape(Lr_cut.shape[0]*Lr_cut.shape[1])
        xy = np.vstack((lon_cut_1d, lat_cut_1d)).T
        if num_interp < 250000:
            print('重采样(cubic)...')
            Lr_interp = griddata(xy, Lr_cut_1d, (xx_interp, yy_interp), method='cubic')
        else:
            print('重采样(nearest)...')
            Lr_interp = griddata(xy, Lr_cut_1d, (xx_interp, yy_interp), method='nearest')
        # 大气校正
        info_dict['tau'] = tau_r[i]
        rrs_interp[:, :, i] = radi_arms_corr(Lr_interp, mtl_coef=info_dict, wave_index=i)
        print('第%d波段处理完毕' % (i+1))
    # 数据保存
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(file_out, rrs_interp.shape[1], rrs_interp.shape[0],
                                rrs_interp.shape[2], gdal.GDT_UInt16)
    degree_step = 0.5/111
    geo_trans = (target_range[0], degree_step, 0, target_range[3], 0, -degree_step)
    target_ds.SetGeoTransform(geo_trans)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    proj_ref = raster_srs.ExportToWkt()
    target_ds.SetProjection(proj_ref)
    for nband in range(8):
        data_tmp = (rrs_interp[:, :, nband]*10000).astype(np.int)
        mask_key = data_tmp < -99
        data_tmp[mask_key] = 65530
        target_ds.GetRasterBand(nband+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(nband+1)
        band.SetNoDataValue(65530)
    target_ds = None
    f_Lr.close()
            

if __name__ == '__main__':
    file_L1B = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\201904\GOCI\COMS_GOCI_L1B_GA_20190415031643.he5'
    goci_info = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\goci_info.h5'
    path_out = os.path.split(file_L1B)[0]
    sub_range = [865, 1015, 3590, 3740]
    target_range = [119.891896, 120.64138, 30.928214, 31.548645]
    name_out = os.path.split(file_L1B)[1]
    main(file_L1B=file_L1B, goci_info=goci_info, alt=0.01, visib=40.0,
        aero_type=1, target_type=2, sub_range=sub_range, target_range=target_range, path_out=path_out)
