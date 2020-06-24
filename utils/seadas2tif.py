# coding=UTF-8
'''
@Author: cuiqiyuan
@Date: 2019-12-31 15:34:18
@Description: file content
'''
import os
import numpy as np
from pyhdf.SD import SD
import re
import skimage.io
import h5py
from scipy.interpolate import griddata
from osgeo import gdal, osr, ogr

from tools import h5_to_geotif


def raster2tif(raster, mask_value, geotrans, projref, file_out):
    '''
    @description: numpy array保存为tif
    @raster {numpy array} 栅格数据 
    @mask {numpy array} 掩膜值
    @geotrans {any} 地理空间范围
    @projref {any} 坐标系信息
    @file_out {str} 输出文件
    '''
    driver = gdal.GetDriverByName('GTiff')
    if len(raster.shape) == 2:
        nbands = 1
    else:
        nbands = raster.shape[2]
    target_ds = driver.Create(
        file_out, raster.shape[1], raster.shape[0], nbands, gdal.GDT_Int16)
    target_ds.SetGeoTransform(geotrans)
    target_ds.SetProjection(projref)
    if nbands == 1:
        target_ds.GetRasterBand(1).WriteArray(raster[:,:,0])
        target_ds.GetRasterBand(1).SetNoDataValue(int(mask_value))
    else:
        for i in range(nbands):
            target_ds.GetRasterBand(i+1).WriteArray(raster[:,:,i])
            target_ds.GetRasterBand(i+1).SetNoDataValue(int(mask_value))
    target_ds = None


def main(ifile_seadas, tif_ex_file, data_field, lon_field, lat_field, ofile):
    '''
    @description: 将SeaDAS大气校正的结果转换为tif
    @ifile_seadas {str} SeaDAS大气校正后的文件
    @data_field {list} SeaDAS波段field
    @lon_field {str} longitude对应h5的field
    @lat_field {str} latitude对应h5的field
    @ofile {str} 融合后的输出文件
    @xy_step {}
    '''
    raster = gdal.Open(tif_ex_file)
    geo_trans = raster.GetGeoTransform()
    xy_size = [raster.RasterXSize, raster.RasterYSize]
    geo_range = [geo_trans[0], geo_trans[0]+raster.RasterXSize*geo_trans[1],
                geo_trans[3]+geo_trans[5]*raster.RasterYSize, geo_trans[3]]
    fp = h5py.File(ifile_seadas, 'r')
    lon0 = fp[lon_field]
    lon0 = lon0[:]
    lat0 = fp[lat_field]
    lat0 = lat0[:]
    lon_1d = np.linspace(geo_range[0], geo_range[1], xy_size[0])
    lat_1d = np.linspace(geo_range[3], geo_range[2], xy_size[1])
    [lon, lat] = np.meshgrid(lon_1d, lat_1d)
    resolution_x = (geo_range[1] - geo_range[0]) / (xy_size[0] - 1)
    resolution_y = (geo_range[2] - geo_range[3]) / (xy_size[1] - 1)
    res = (np.zeros([lon.shape[0], lon.shape[1], len(data_field)])).astype(int)
    n = 0
    for item in data_field:
        data0 = fp[item]
        mask_value = data0.attrs['_FillValue'][0]
        try:
            scale = data0.attrs['scale_factor']
            offset = data0.attrs['add_offset']
        except:
            scale = 1
            offset = 0
        data0 = data0[:]
        data0[0, :] = mask_value
        data0[-1, :] = mask_value
        data0[:, 0] = mask_value
        data0[:, -1] = mask_value
        print('重投影(%s)...' % item)
        lon1d = np.reshape(lon0, lon0.shape[0]*lon0.shape[1])
        lat1d = np.reshape(lat0, lat0.shape[0]*lat0.shape[1])
        data1d = np.reshape(data0, data0.shape[0]*data0.shape[1])
        lonlat = (np.vstack([[lon1d], [lat1d]])).T
        data_tmp = griddata(lonlat, data1d, (lon, lat), method='nearest')
        data_tmp1 = (data_tmp*scale + offset) * 10000
        data_tmp1[data_tmp==mask_value] = -32767
        res[:,:,n] = data_tmp1
        n += 1
    # 文件保存
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(4326)
    proj_ref = raster_srs.ExportToWkt()
    raster2tif(res, -32767, geo_trans, proj_ref, ofile)
    return(ofile)
