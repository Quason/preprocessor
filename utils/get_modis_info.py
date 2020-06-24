import os
import simplejson
import re
import gdal
import numpy as np
import math

from tools import img_cut

# 全局配置
rootpath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.dirname(rootpath)
jsonFile = os.path.join(rootpath, 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


def main_sa_position(ifile, shp_file):
    '''
    @description: 
    @ifile {str}: L1B swath文件 
    @shp_file {str}: 研究区域矢量文件
    @return: {salz, sala}
    '''
    # 设置环境变量
    os.environ['MRTDATADIR'] = global_config['MRTDATADIR']
    os.environ['PGSHOME'] = global_config['PGSHOME']
    os.environ['MRTBINDIR'] = global_config['MRTBINDIR']
    run_path = os.path.split(ifile)[0]
    disk_id = run_path[0:2]
    heg_bin = os.path.join(global_config['MRTBINDIR'], 'hegtool')
    os.system('cd %s && %s -h %s > heg.log' % (run_path, heg_bin, ifile))
    info_file = os.path.join(run_path, 'HegHdr.hdr')
    if not(os.path.exists(info_file)):
        print('获取文件信息出错：%s' % ifile)
        exit(0)
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
            exit(0)
        # 计算研究区域的平均卫星天顶角和方位角
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

