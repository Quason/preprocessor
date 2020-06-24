# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: ladsat 8数据预处理: 辐射定标+6S大气校正
@Date: 2019-07-23 08:27:23
'''
import numpy as np
from osgeo import gdal
import os
import skimage.io
import sys
import re
import tarfile
import shutil
import simplejson
import time

import utils.img_cut as img_cut

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)

path_6s = global_config['path_6s']
wavelength_down = [0.438, 0.455, 0.537, 0.640, 0.854, 1.604, 2.146, 0.500, 1.368]
wavelength_up   = [0.448, 0.510, 0.585, 0.670, 0.874, 1.641, 2.279, 0.680, 1.379]

def read_coef_MTL(MTLfile):
    '''
    @description: 读取MLT文件中的辐射定标参数
    '''
    with open(MTLfile, 'r') as fp:
        for line in fp.readlines():
            if 'RADIANCE_MULT_BAND_1 ' in line:
                mult_B1 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_2 ' in line:
                mult_B2 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_3 ' in line:
                mult_B3 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_4 ' in line:
                mult_B4 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_5 ' in line:
                mult_B5 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_6 ' in line:
                mult_B6 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_7 ' in line:
                mult_B7 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_8 ' in line:
                mult_B8 = float(line.split(' ')[-1])
            elif 'REFLECTANCE_MULT_BAND_9 ' in line:
                mult_B9 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_10 ' in line:
                mult_B10 = float(line.split(' ')[-1])
            elif 'RADIANCE_MULT_BAND_11 ' in line:
                mult_B11 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_1 ' in line:
                add_B1 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_2 ' in line:
                add_B2 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_3 ' in line:
                add_B3 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_4 ' in line:
                add_B4 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_5 ' in line:
                add_B5 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_6 ' in line:
                add_B6 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_7 ' in line:
                add_B7 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_8 ' in line:
                add_B8 = float(line.split(' ')[-1])
            elif 'REFLECTANCE_ADD_BAND_9 ' in line:
                add_B9 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_10 ' in line:
                add_B10 = float(line.split(' ')[-1])
            elif 'RADIANCE_ADD_BAND_11 ' in line:
                add_B11 = float(line.split(' ')[-1])
            elif 'SUN_AZIMUTH' in line:
                sun_azimuth = float(line.split(' ')[-1])
            elif 'SUN_ELEVATION' in line:
                sun_zenith = 90 - float(line.split(' ')[-1])
            elif 'DATE_ACQUIRED = ' in line:
                dateStr = line.split(' ')[-1]
            elif 'CORNER_UL_LAT_PRODUCT' in line:
                lat_ul = float(line.split(' ')[-1])
            elif 'CORNER_UL_LON_PRODUCT' in line:
                lon_ul = float(line.split(' ')[-1])
            elif 'CORNER_UR_LAT_PRODUCT' in line:
                lat_ur = float(line.split(' ')[-1])
            elif 'CORNER_UR_LON_PRODUCT' in line:
                lon_ur = float(line.split(' ')[-1])
            elif 'CORNER_LL_LAT_PRODUCT' in line:
                lat_ll = float(line.split(' ')[-1])
            elif 'CORNER_LL_LON_PRODUCT' in line:
                lon_ll = float(line.split(' ')[-1])
            elif 'CORNER_LR_LAT_PRODUCT' in line:
                lat_lr = float(line.split(' ')[-1])
            elif 'CORNER_LR_LON_PRODUCT' in line:
                lon_lr = float(line.split(' ')[-1])
            elif 'K1_CONSTANT_BAND_10' in line:
                K1_band10 = float(line.split(' ')[-1])
            elif 'K2_CONSTANT_BAND_10' in line:
                K2_band10 = float(line.split(' ')[-1])
            elif 'K1_CONSTANT_BAND_11' in line:
                K1_band11 = float(line.split(' ')[-1])
            elif 'K2_CONSTANT_BAND_11' in line:
                K2_band11 = float(line.split(' ')[-1])
    coef = {}
    coef['mult'] = [mult_B1, mult_B2, mult_B3, mult_B4, mult_B5, mult_B6,
                    mult_B7, mult_B8, mult_B9, mult_B10, mult_B11]
    coef['add'] = [add_B1, add_B2, add_B3, add_B4, add_B5, add_B6,
                    add_B7, add_B8, add_B9, add_B10, add_B11]
    coef['sun'] = [sun_zenith, sun_azimuth]
    coef['date'] = dateStr
    coef['location'] = [np.mean([lon_ul, lon_ur, lon_ll, lon_lr]), np.mean([lat_ul, lat_ur, lat_ll, lat_lr])]
    coef['K1_B10'] = K1_band10
    coef['K2_B10'] = K2_band10
    coef['K1_B11'] = K1_band11
    coef['K2_B11'] = K2_band11
    return(coef)


def radi_arms_corr(raster_data, MTLfile, altitude, visibility, type_aero, wave_index):
    '''
    @description: 辐射定标和6S大气校正
    @raster_data {numpy array} 辐射定标后的数据
    @MTLfile {str} landsat 8 MTL文件
    @altitude {float} 海拔(km)
    @visibility {float} 能见度(km)
    @type_atms {int} 大气模型 0 无; 1 热带; 2 中纬度夏季; 3 中纬度冬季;
     4 亚热带夏季; 5 亚热带冬季; 6 US 62
    @type_aero {int} 气溶胶模型 0 无; 1 大陆; 2 近海; 3 城市; 5 沙漠;
     6 有机质燃烧; 7 平流层
    @wave_index {int} 波段下标(从0开始)
    @retur {numpy array} 大气校正后的栅格数据
    '''
    print('辐射定标...')
    mtl_coef = read_coef_MTL(MTLfile)
    radi_cali_data = raster_data * mtl_coef['mult'][wave_index] + mtl_coef['add'][wave_index]
    if wave_index < 7:
        print('大气校正...')
        with open(os.path.join(path_6s, 'in.txt'), 'w') as fp:
            # igeom
            fp.write('%d\n' % 0)
            # SOLZ SOLA THV PHV month day
            SOLZ = mtl_coef['sun'][0]
            SOLA = mtl_coef['sun'][1]
            PHV = 0
            THV = 0
            month = int(mtl_coef['date'][5:7])
            day = int(mtl_coef['date'][8:10])
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
            fp.write('%d\n' % type_aero)
            # 能见度
            fp.write('%.1f\n' % visibility) # 需根据条件改变
            # 高程
            fp.write('%.3f\n' % -altitude) # 需根据条件改变
            # 传感器类型
            fp.write('%d\n' % -1000)
            # if wave_index >= 1 and wave_index <= 6:
            if False:
                fp.write('%d\n' % (wave_index + 136))
            else:
                # 波段号(-2表示自己输入范围)
                fp.write('%d\n' % -2)
                fp.write('%.4f %.4f\n' % (wavelength_down[wave_index], wavelength_up[wave_index]))
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
                    y = xa*radi_cali_data - xb
                    atms_corr_data = y/(1.0+xc*y)/3.14159
                    break
    elif wave_index == 8:
        atms_corr_data = radi_cali_data
    elif wave_index == 9:
        atms_corr_data = mtl_coef['K2_B10'] / np.log(mtl_coef['K1_B10']/radi_cali_data+1)
    elif wave_index == 10:
        atms_corr_data = mtl_coef['K2_B11'] / np.log(mtl_coef['K1_B11']/radi_cali_data+1)
    return(atms_corr_data)
    

def main(ifile, path_out_30, shp_file=None, altitude=0.01, visibility=40.0, type_aero=1):
    '''
    @description: 
    @ifile {str} tar.gz原始数据文件
    @path_out_30 {str} 预处理后的多光谱文件输出路径
    @shp_file {str} 矢量文件路径
    @altitude {float} 海拔(km)
    @visibility {float} 能见度(km)
    @type_aero {int} 气溶胶类型(3默认为城市)
    '''
    tf = tarfile.open(ifile)
    path_short = os.path.split(ifile)[1]
    path_short = path_short.replace('.tar.gz', '')
    file_path = os.path.join(os.path.split(ifile)[0], path_short)
    print('文件解压...')
    tf.extractall(path=file_path)
    file_list = os.listdir(file_path)
    MTLfile = ''
    band_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for item in file_list:
        if 'MTL.txt' in item:
            MTLfile = item
            break
    raster_b1 = os.path.join(file_path, MTLfile.replace('MTL.txt', 'B2.TIF'))
    lim = img_cut.main(raster_file = raster_b1, shp_file = shp_file)
    xlim = [lim[0], lim[1]]
    ylim = [lim[2], lim[3]]
    raster_data_res = np.zeros((ylim[1]-ylim[0], xlim[1]-xlim[0], len(band_list)))
    MTLfile = os.path.join(file_path, MTLfile)
    print('预处理...')
    for i in band_list:
        if i != 7: # B8分辨率不一致，不处理
            print('裁切...')
            raster_file = os.path.join(file_path, MTLfile.replace('MTL.txt', 'B'+str(i+1)+'.TIF'))
            raster_data = skimage.io.imread(raster_file)
            # 裁切
            raster_data = raster_data[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            # 辐射定标和大气校正(6S)
            nan_key = raster_data < 1e-6
            band_data_atmscorr = radi_arms_corr(raster_data=raster_data, MTLfile=MTLfile,
                altitude=altitude, visibility=visibility, type_aero=type_aero, wave_index=i)
            band_data_atmscorr[nan_key] = 0
            raster_data_res[:,:,i] = band_data_atmscorr
        print('第%d波段已经处理完毕' % (i+1))
    # 数据保存为tif
    raster_file = os.path.join(file_path, MTLfile.replace('MTL.txt', 'B1.TIF'))
    raster = gdal.Open(raster_file)
    georef0 = raster.GetGeoTransform()
    georef1 = list(georef0)
    georef1[0] = georef0[0] + xlim[0] * georef0[1]
    georef1[3] = georef0[3] + ylim[0] * georef0[5]
    georef1 = tuple(georef1)
    # 可见光-短波红外
    ifile_name = os.path.split(ifile)[1]
    date_str = '%s103000' % (ifile_name.split('_')[3])
    nrow = ifile_name.split('_')[2][0:3]
    npath = ifile_name.split('_')[2][3:6]
    name_short = 'Landsat8_OLI_30_L2_%s_%s_%s.tif' % (date_str, nrow, npath)
    raster_file_out = os.path.join(path_out_30, name_short)
    if os.path.exists(raster_file_out): # 如果文件存在则需要删除
        os.remove(raster_file_out)
    target_ds = gdal.GetDriverByName('GTiff').Create(
        raster_file_out,
        raster_data_res.shape[1],
        raster_data_res.shape[0],
        9,
        gdal.GDT_UInt16
    )
    target_ds.SetGeoTransform(georef1)
    target_ds.SetProjection(raster.GetProjectionRef())
    for i in range(9):
        print('保存第%d波段...' % (i+1))
        if i < 7:
            data_tmp = (raster_data_res[:,:,i]*10000).astype(np.int)
        else:
            data_tmp = (raster_data_res[:,:,i+2]*100).astype(np.int)
        data_tmp[data_tmp<1e-6] = 0
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        target_ds.GetRasterBand(i+1).SetNoDataValue(0)
    target_ds = None
    # 删除解压文件
    raster = None
    shutil.rmtree(file_path)


if __name__ == '__main__':
    file_path = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\洪泽湖\LC8'
    raster_file_out = os.path.join(file_path, '20190406', 'LC08_L1TP_119038_20190415_20190423_01_T1_atmscorr.tif')
    shp_file = r'D:\Job\ImageSky\back-end\jiangsu_water_demo_model_data\shp\hongzuhu\hongzehu.shp'
    altitude = 0.01 # 海拔
    visibility = 15.0 # 能见度
    # 气溶胶模型 0 无; 1 大陆; 2 近海; 3 城市; 5 沙漠;
    aero = 1
    main(file_path, raster_file_out, shp_file, altitude=altitude, visibility=visibility, type_aero=aero)
