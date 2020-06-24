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
from osgeo import gdal, osr, ogr

from tools import h5_to_geotif


def shp2mask(raster_fn, vector_fn, raster_fn_out):
    """ 将矢量数据转换为水文模型的目标网格
    Args:
        raster_fn: 输入栅格
        vector_fn: 矢量文件(.shp)全路径
        raster_fn_out: 掩膜文件栅格
    Returns: None
    Raises: .shp路径下需要有.prj投影文件
    """
    # 新建掩抹文件驱动
    raster = gdal.Open(raster_fn)
    tifData = raster.GetRasterBand(1).ReadAsArray()
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_out, tifData.shape[1], tifData.shape[0], 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjectionRef())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0]) # 栅格化函数
    target_ds = None
    raster = None


def main(ifile_6s, ifile_seadas, fuse_band, fuse_index, lon_field, lat_field,
    shp_file, file_out):
    '''
    @ifile_6s {str} 6s大气校正后的文件
    @ifile_seadas {str} SeaDAS大气校正后的文件
    @fuse_band {list} SeaDAS波段field
    @fuse_index {list} SeaDAS对应的6S的index
    @lon_field {str} longitude对应h5的field
    @lat_field {str} latitude对应h5的field
    @shp_file {str} 海陆分解文件矢量
    @file_out {str} 融合后的输出文件
    '''
    raster_6s = gdal.Open(ifile_6s)
    geo_trans = raster_6s.GetGeoTransform()
    proj_ref = raster_6s.GetProjectionRef()
    xy_size = [raster_6s.RasterXSize, raster_6s.RasterYSize]
    geo_range = [
        geo_trans[0],
        geo_trans[0]+xy_size[0]*geo_trans[1],
        geo_trans[3]+xy_size[1]*geo_trans[5],
        geo_trans[3]
    ]
    # 区分海陆
    mask_file = shp_file.replace('.shp', '_mask.tif')
    shp2mask(ifile_6s, shp_file, mask_file)
    key_land = skimage.io.imread(mask_file) == 0
    target_ds = gdal.GetDriverByName('GTiff').Create(
        file_out,
        raster_6s.RasterXSize,
        raster_6s.RasterYSize,
        raster_6s.RasterCount,
        raster_6s.GetRasterBand(1).DataType
    )
    for i in range(raster_6s.RasterCount):
        if i in fuse_index:
            print('%s ...' % fuse_band[fuse_index.index(i)])
            data_field = fuse_band[fuse_index.index(i)]
            data_seadas_tif0 = h5_to_geotif.main(
                ifile_seadas,
                data_field,
                lon_field,
                lat_field,
                geo_range,
                xy_size
            )
            data_seadas_tif = data_seadas_tif0['data']
            data_6s = raster_6s.GetRasterBand(i+1).ReadAsArray()
            
            mosaic_data_band = data_seadas_tif
            key_seadas_nan = data_seadas_tif==data_seadas_tif0['mask']
            mosaic_data_band[key_seadas_nan] = data_6s[key_seadas_nan]
            
            # mosaic_data_band = np.zeros([data_seadas_tif.shape[0], data_seadas_tif.shape[1], 2])
            # data_seadas_tif[data_seadas_tif==data_seadas_tif0['mask']] = -999
            # mosaic_data_band[:, :, 0] = data_seadas_tif
            # mosaic_data_band[:, :, 1] = data_6s
            # mosaic_data_band = np.nanmax(mosaic_data_band, 2)
            
            target_ds.GetRasterBand(i+1).WriteArray(mosaic_data_band)
            target_ds.GetRasterBand(i+1).SetNoDataValue(2**16-1)
        else:
            data_6s = raster_6s.GetRasterBand(i+1).ReadAsArray()
            target_ds.GetRasterBand(i+1).WriteArray(data_6s)
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    target_ds = None
    return(file_out)
