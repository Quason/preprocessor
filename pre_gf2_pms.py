# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: 高分2号PMS传感器数据(4m)预处理
@Date: 2019-09-24 08:34:06
'''
import os
import simplejson
import zipfile
import re
import numpy as np
import tarfile
import shutil
from osgeo import gdal

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)

wavelength_low = [0.45, 0.52, 0.63, 0.77]
wavelength_up = [0.52, 0.59, 0.69, 0.89]
PI = 3.14159
tau_r = [0.0287501, 0.028279, 0.0279022, 0.0275933]
# 辐射定标系数表
SCALES = {
    'PMS_2018': [0.1356, 0.1736, 0.1644, 0.1788],
    'PMS_latest': [0.1356, 0.1736, 0.1644, 0.1788]
}


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


def main(ifile, shp_file=None, cut_range=None, aerotype=1,
        altitude=0.01, visibility=15, path_out=None):
    nbands = 4
    # 文件解压
    print('文件解压...')
    if ifile[-3:] == '.gz':
        tf = tarfile.open(ifile)
        path_name = os.path.split(ifile)[1]
        path_name = path_name.replace('.tar.gz', '')
        file_path = os.path.join(os.path.split(ifile)[0], path_name)
        tf.extractall(path=file_path)
    elif ifile[-3:] == 'zip':
        path_name = os.path.split(ifile)[1]
        path_name = path_name.replace('.zip', '')
        file_path = os.path.join(os.path.split(ifile)[0], path_name)
        fz = zipfile.ZipFile(ifile, 'r')
        for file in fz.namelist():
            fz.extract(file, file_path)
    file_list = os.listdir(file_path)
    for item in file_list:
        if 'PMS' in item and 'MSS' in item and '.tiff' in item:
            mss_name = item
            break
    print('数据预处理...')
    # 通过文件名获取传感器日期
    path_name = os.path.split(file_path)[1]
    raster = gdal.Open(os.path.join(file_path, mss_name))
    nbands = raster.RasterCount
    # 根据文件名判断日期
    date_str = path_name.split('_')[4]
    if date_str[0:4] == '2018' and 'PMS' in path_name:
        scales = SCALES['PMS_2018']
    elif 'PMS' in path_name:
        scales = SCALES['PMS_latest']
    else:
        print('无法识别的传感器类型!')
        return(0)
    month = date_str[4:6]
    day = date_str[6:8]
    # 通过文件名获取中心经纬度
    center_lon = path_name.split('_')[2]
    center_lon = float(center_lon[1:])
    center_lat = path_name.split('_')[3]
    center_lat = float(center_lat[1:])
    center_lonlat = [center_lon, center_lat]
    # 获取太阳和卫星的方位信息
    for item in file_list:
        if 'PMS' in item and 'MSS' in item and '.xml' in item:
            xml_name = item
            break
    with open(os.path.join(file_path, xml_name), 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if '<SolarAzimuth>' in line:
                sola = re.findall(r'[\d.]+', line)
                sola = float(sola[0])
            elif '<SolarZenith>' in line:
                solz = re.findall(r'[\d.]+', line)
                solz = float(solz[0])
            elif '<SatelliteAzimuth>' in line:
                sala = re.findall(r'[\d.]+', line)
                sala = float(sala[0])
            elif '<SatelliteZenith>' in line:
                salz = re.findall(r'[\d.]+', line)
                salz = 90.0 - float(salz[0])
            elif '<CenterTime>' in line:
                time_str = re.findall(r'\d+:\d+:\d+', line)
                time_str = time_str[0].replace(':', '')
            elif '<ScenePath>' in line:
                nrow = re.findall(r'\d+', line)
                nrow = nrow[0]
            elif '<SceneRow>' in line:
                ncolm = re.findall(r'\d+', line)
                ncolm = ncolm[0]
    size_x = raster.RasterXSize
    size_y = raster.RasterYSize
    rrs_join = np.zeros([size_y, size_x, nbands])
    for i_band in range(nbands):
        data = raster.GetRasterBand(i_band+1).ReadAsArray()
        # 辐射定标
        Lr = data * scales[i_band]
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
        rrs_join[:, :, i_band] = arms_corr(Lr, mtl_coef, i_band)
    satellite_code = path_name.split('_')[0]
    sensor_code = path_name.split('_')[1]
    date_str = date_str + time_str
    name_out = '%s_%s_4_L2_%s_%s_%s.tiff' % (satellite_code, sensor_code, date_str, nrow, ncolm)
    raster_fn_out = os.path.join(path_out, name_out)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(raster_fn_out, size_x, size_y, nbands, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjectionRef())
    for i in range(nbands):
        data_tmp = rrs_join[:, :, i]
        mask = np.isnan(data_tmp)
        data_tmp = (data_tmp * 10000).astype(np.int)
        data_tmp[mask] = 65530
        target_ds.GetRasterBand(i+1).WriteArray(data_tmp)
        band = target_ds.GetRasterBand(1+1)
        band.SetNoDataValue(65530)
    target_ds = None
    # 拷贝RPB文件
    for item in file_list:
        if 'PMS' in item and 'MSS' in item and '.rpb' in item:
            rpb_name = item
            break
    rpb_name_rename = raster_fn_out.replace('.tiff', '.rpb')
    shutil.copy(os.path.join(file_path, rpb_name), rpb_name_rename)
    xml_name = xml_name.replace('.rpb', '.xml')
    xml_name_rename = raster_fn_out.replace('.tiff', '.xml')
    shutil.copy(os.path.join(file_path, xml_name), xml_name_rename)
    # 删除解压文件
    raster = None
    shutil.rmtree(file_path)
    print('处理完成!')
