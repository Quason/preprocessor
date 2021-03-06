'''
@Author: Marvin
@Date: 2020-06-18 15:16:42
@LastEditTime: 2020-06-18 18:34:35
@LastEditors: Marvin
@Description: 多源卫星数据预处理（水环境相关）
'''
import os
import sys
import shutil
import argparse
import simplejson
import time

import models.pre_modis_qkm as modis250m
import models.pre_modis_hkm as modis500m
import models.pre_modis_1km as modis1km
import models.pre_l8 as landsat8
import models.pre_l5 as landsat5
import models.pre_viirs_375m as npp375m
import models.pre_viirs_750m as npp750m
# import models.pre_s2 as sentinel2
# import models.pre_s2_separate as sentinel2
import models.pre_s2_parallel as sentinel2
import models.pre_s3_olci as s3_olci
import models.pre_s3_slstr as s3_slstr
import models.pre_goci as goci
import models.pre_himawari8 as himawari8
import models.pre_gf1_pms as gf1_pms
import models.pre_gf1_wfv as gf1_wfv
import models.pre_gf2_pms as gf2
import models.pre_zy3 as zy3

rootpath = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(rootpath, 'models', 'global_configure.json')
with open(jsonFile, 'r') as fp:
    global_config = simplejson.load(fp)


SATELLITE_TYPE = [
    'AQUA-MODIS-250',
    'AQUA-MODIS-500',
    'AQUA-MODIS-1KM',
    'TERRA-MODIS-250',
    'TERRA-MODIS-500',
    'TERRA-MODIS-1KM',
    'NPP-VIIRS-375',
    'NPP-VIIRS-750',
    'Landsat8',
    'Landsat5',
    'COMS-GOCI',
    'Himawari8-HLI',
    'Sentinel2',
    'Sentinel3-OLCI',
    'Sentinel3-SLSTR',
    'GF1-WFV',
    'GF1-PMS',
    'GF2-PMS',
    'ZY3-MUX'
]


def main(file_in, path_out, satellite_type, file_in_geo,
        location, aerosol, view, altitude, target, cores):
    '''
    @description: 卫星数据预处理
    @satellite_type {str} 原始数据类型
    @file_in_geo {str} 地理矢量数据用于裁切
    @location {[float, float]} 目标经纬度
    @aerosol {int} 气溶胶类型(0:无 1:大陆 2:近海 3:城市)
    @view {float} 能见度
    @altitude {float} target altitude (km)
    @cores {int} cores of CPU
    '''
    if satellite_type == 'AQUA-MODIS-1KM' or satellite_type == 'TERRA-MODIS-1KM':
        # MODIS 1km
        sensor_position = modis1km.main_sa_position(ifile=file_in, shp_file=None)
        salz = sensor_position['salz']
        sala = sensor_position['sala']
        if 'AQUA' in satellite_type:
            satellite = 'AQUA'
        else:
            satellite = 'TERRA'
        if sala < 0:
            sala = sala + 360
        modis1km.main(
            ifile=file_in, shp_file=None, satellite=satellite,
            center_lonlat=location, aerotype=aerosol, altitude=altitude,
            visibility=view, salz=salz, sala=sala, band_need=['all'],
            path_out=path_out
        )
    elif satellite_type == 'AQUA-MODIS-500' or satellite_type == 'TERRA-MODIS-500':
        # MODIS 500m
        path_out_radi = os.path.join(
            os.path.split(path_out)[0], 'RADI_CALI', os.path.split(path_out)[1])
        path_out_6s = path_out
        if 'AQUA' in satellite_type:
            satellite = 'AQUA'
        else:
            satellite = 'TERRA'
        [salz, sala] = modis500m.satellite_position_rough(file_in, location)
        if sala < 0:
            sala = sala + 360
        modis500m.main(
            ifile=file_in, shp_file=None, center_lonlat=location,
            satellite=satellite, aerotype=aerosol, altitude=altitude, 
            visibility=view, salz=salz, sala=sala, band_need=['all'],
            path_out_radi=path_out_radi, path_out_6s=path_out_6s
        )
    elif satellite_type == 'AQUA-MODIS-250' or satellite_type == 'TERRA-MODIS-250':
        # MODIS 250m
        path_out_radi = os.path.join(
            os.path.split(path_out)[0], 'RADI_CALI', os.path.split(path_out)[1])
        path_out_6s = path_out
        if 'AQUA' in satellite_type:
            satellite = 'AQUA'
        else:
            satellite = 'TERRA'
        [salz, sala] = modis250m.satellite_position_rough(file_in, location)
        if sala < 0:
            sala = sala + 360
        modis250m.main(
            ifile=file_in, shp_file=None, center_lonlat=location,
            satellite=satellite, aerotype=aerosol, altitude=altitude, 
            visibility=view, salz=salz, sala=sala, band_need=['all'],
            path_out_radi=path_out_radi, path_out_6s=path_out_6s
        )
    elif satellite_type == 'NPP-VIIRS-375':
        # NPP/VIIRS 375m
        subrange = [location[0]-0.2, location[0]+0.2, location[1]-0.2, location[1]+0.2]
        if not(file_in_geo):
            print('Invalid geofile')
            return(0)
        npp375m.main(
            file_ref=file_in, file_info=file_in_geo, subrange=subrange,
            aerotype=aerosol, altitude=altitude, visibility=view, path_out=path_out
        )
    elif satellite_type == 'NPP-VIIRS-750':
        # NPP/VIIRS 750m
        sensor_position = npp750m.main_sa_position(ifile=file_in, cut_range=None, shp_file=None)
        salz = sensor_position['salz']
        sala = sensor_position['sala']
        if sala < 0:
            sala = sala + 360
        npp750m.main(
            ifile=file_in, shp_file=None, center_lonlat=location, aerotype=aerosol,
            altitude=altitude, visibility=view, salz=salz, sala=sala, cut_range=None,
            band_need=['all'], path_out=path_out
        )
    elif satellite_type == 'Landsat8':
        # Landsat8
        landsat8.main(
            ifile=file_in, path_out_30=path_out, shp_file=None,
            altitude=altitude, visibility=view, type_aero=aerosol
        )
    elif satellite_type == 'Landsat5':
        # Landsat5
        landsat5.main(
            ifile=file_in, path_out_30=path_out, shp_file=None,
            altitude=altitude, visibility=view, type_aero=aerosol
        )
    elif satellite_type == 'COMS-GOCI-500':
        # COMS/GOCI
        goci_info = os.path.join(
            global_config['path_model_file'],
            'model',
            'static',
            'goci_info.h5'
        )
        sub_range = [0, 2000, 1500, 5500] # 江苏省行列范围
        target_range = [117.06, 125.5, 25.726, 36.05] # 江苏省区域范围
        goci.main(
            file_L1B=file_in, goci_info=goci_info, alt=0.01, visib=view,
            aero_type=aerosol, target_type=aerosol, sub_range=sub_range,
            target_range=target_range, path_out=path_out
        )
    elif satellite_type == 'Himawari8-HLI-1000':
        # Himawari8
        target_range = [115.17, 125.93, 29.2, 37.7] # 江苏省区域范围
        himawari8.main(
            file_L1B=file_in, alt=altitude, visib=view, aero_type=aerosol,
            target_type=aerosol, target_range=target_range, path_out=path_out
        )
    elif satellite_type == 'Sentinel2':
        # Sentinel2
        path_out_10m = path_out
        sentinel2.main(
            ifile=file_in,
            aero_type=aerosol,
            target_type=target,
            altitude=altitude,
            visibility=view,
            dst_dir=path_out_10m,
            cpu_cores=cores
        )
    elif satellite_type == 'Sentinel3-OLCI':
        # Sentinel3 OLCI
        s3_olci.main(
            ifile=file_in, shp_file=None, center_lonlat=location, aerotype=aerosol,
            altitude=altitude, visibility=view, cut_range=None, band_need=['all'],
            path_out=path_out
        )
    elif satellite_type == 'Sentinel3-SLSTR':
        # Sentinel3 SLSTR
        s3_slstr.main(
            ifile=file_in, shp_file=None, center_lonlat=location, aerotype=aerosol,
            altitude=altitude, visibility=view, cut_range=None, band_need=['all'],
            path_out=path_out
        )
    elif satellite_type == 'GF1-WFV':
        # 高分1号 WFV
        gf1_wfv.main(
            ifile=file_in, aerotype=aerosol, altitude=altitude,
            visibility=view, path_out=path_out
        )
    elif satellite_type == 'GF1-PMS':
        # 高分1号 PMS
        gf1_pms.main(
            ifile=file_in, aerotype=aerosol, altitude=altitude,
            visibility=view, path_out=path_out
        )
    elif satellite_type == 'GF2-PMS':
        # 高分2号
        gf2.main(
            ifile=file_in, aerotype=aerosol, altitude=altitude,
            visibility=view, path_out=path_out
        )
    elif satellite_type == 'ZY3-MUX':
        # 资源3号
        zy3.main(
            ifile=file_in, aerotype=aerosol, altitude=altitude,
            visibility=view, path_out=path_out
        )


if __name__ == '__main__':
    time_s = time.time()
    parser = argparse.ArgumentParser(description='Satellite Data Preprocess')
    parser.add_argument('--ifile', type=str, help='原始影像')
    parser.add_argument('--opath', type=str, help='输出路径')
    parser.add_argument('--type', type=str, help='影像类型')
    parser.add_argument('--geofile', type=str, default='', help='地理参考文件')
    parser.add_argument('--lon', type=float, default=120.17, help='目标中心经度')
    parser.add_argument('--lat', type=float, default=31.2, help='目标中心纬度')
    parser.add_argument('--aero', type=int, default=1, help='气溶胶类型 0无 1大陆(默认) 2近海 3城市')
    parser.add_argument('--view', type=float, default=15.0, help='能见度(默认15km)')
    parser.add_argument('--alti', type=float, default=0.01, help='地面高程(默认0.01km)')
    parser.add_argument('--target', type=int, default=4, help='地面目标物类型 1 植被 2 清洁水体 3 沙漠 4 湖泊水体(默认)')
    parser.add_argument('--cores', type=int, default=1, help='用于并行处理的核心数(默认1)')
    args = parser.parse_args()
    if not(args.type in SATELLITE_TYPE):
        print('[Error] Invalid satellite type(--type), should be one of the following:\n%s' % (
            ','.join(SATELLITE_TYPE)))
        sys.exit(0)
    main(
        file_in=args.ifile, 
        path_out=args.opath, 
        satellite_type=args.type,
        file_in_geo=args.geofile,
        location=[args.lon, args.lat],
        aerosol=args.aero,
        view=args.view,
        altitude=args.alti,
        target=args.target,
        cores=args.cores
    )
    time_e = time.time()
    print('%ds in all' % (time_e-time_s))
