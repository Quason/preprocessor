# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: 根据shp计算tif的裁切范围
@Date: 2019-07-17 09:40:48
'''
import numpy as np
from osgeo import gdal, osr, ogr
import skimage.io
import os
import re

def coord_trans(EPSGs, EPSGt, x, y):
    '''
    @description: 投影坐标转地理坐标
    @EPSGs: 原始坐标系的EPSG代码
    @EPSGt: 目标坐标系的EPSG代码
    @x: 原始x坐标
    @y: 原始y坐标
    '''
    sys0 = osr.SpatialReference()
    sys0.ImportFromEPSG(EPSGs) # 投影坐标系
    sys1 = osr.SpatialReference()
    sys1.ImportFromEPSG(EPSGt) # 地理坐标系
    ct = osr.CoordinateTransformation(sys0, sys1)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def main(raster_file, shp_file='', sub_lim=None, corner=None, out_file=None):
    '''
    @description: 根据矢量或经纬度范围裁切栅格
    @raster_file {str} 原始栅格文件
    @shp_file {str} 裁切矢量
    @corner {[lon_lu, lat_lu, lon_rd, lat_rd]} 角点经纬度
    @sub_lim {[x_min, x_max, y_min, y_max]} 自区间范围
    @out_file {str} 输出文件
    @return: 
    '''
    raster = gdal.Open(raster_file)
    if not(shp_file):
        # 建立掩膜文件
        if '.TIF' in raster_file:
            raster_file_out = raster_file.replace('.TIF', '_mask.TIF')
        elif '.tif' in raster_file:
            raster_file_out = raster_file.replace('.tif', '_mask.tif')
        else:
            print('Error 未知的文件类型: %s' % os.path.split[raster_file][-1])
            exit(0)
        tifData = raster.GetRasterBand(1).ReadAsArray()
        target_ds = gdal.GetDriverByName('GTiff').Create(
            raster_file_out, tifData.shape[1], tifData.shape[0], 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(raster.GetGeoTransform())
        target_ds.SetProjection(raster.GetProjectionRef())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(255)
        source_ds = ogr.Open(shp_file)
        source_layer = source_ds.GetLayer()
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0])  # 栅格化函数
        target_ds = None
        # 计算裁切范围
        mask = skimage.io.imread(raster_file_out)
        mask_cp = mask.copy()
        mask_cp[mask == 255] = 0
        mask_cp[mask == 0] = 1
        sum_x = np.sum(mask_cp, axis = 1)
        sum_y = np.sum(mask_cp, axis = 0)
        cut_x_b = 0
        cut_x_e = mask.shape[1]
        cut_y_b = 0
        cut_y_e = mask.shape[0]
        for i in range(len(sum_x)):
            if sum_x[i] != 0:
                cut_y_b = i
                break
        for i in range(len(sum_x)-1, 0, -1):
            if sum_x[i] != 0:
                cut_y_e = i
                break
        for i in range(len(sum_y)):
            if sum_y[i] != 0:
                cut_x_b = i
                break
        for i in range(len(sum_y)-1, 0, -1):
            if sum_y[i] != 0:
                cut_x_e = i
                break
        os.remove(raster_file_out)
    elif not(corner is None):
        proj = raster.GetProjectionRef()
        geo_trans = raster.GetGeoTransform()
        EPSG_t = int(re.findall(r'\d+', re.findall(r'"EPSG","\d+"', proj)[-1])[-1])
        EPSG_s = 4326
        [x_lu, y_lu] = coord_trans(EPSG_s, EPSG_t, corner[0], corner[1])
        [x_rd, y_rd] = coord_trans(EPSG_s, EPSG_t, corner[2], corner[3])
        cut_x_b = round((x_lu - geo_trans[0]) / geo_trans[1])
        cut_x_e = round((x_rd - geo_trans[0]) / geo_trans[1])
        cut_y_b = round((y_lu - geo_trans[3]) / geo_trans[5])
        cut_y_e = round((y_rd - geo_trans[3]) / geo_trans[5])
    elif not(sub_lim is None):
        cut_x_b = sub_lim[0]
        cut_x_e = sub_lim[1]
        cut_y_b = sub_lim[2]
        cut_y_e = sub_lim[3]
    else:
        cut_x_b = 0
        cut_x_e = raster.RasterXSize
        cut_y_b = 0
        cut_y_e = raster.RasterYSize
    if out_file:
        nbands = raster.RasterCount
        target_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, cut_x_e-cut_x_b, cut_y_e-cut_y_b, nbands, raster.GetRasterBand(1).DataType)
        geo_trans0 = raster.GetGeoTransform()
        geo_trans1 = list(geo_trans0).copy()
        geo_trans1[0] = geo_trans0[0] + cut_x_b * geo_trans0[1]
        geo_trans1[3] = geo_trans0[3] + cut_y_b * geo_trans0[5]
        target_ds.SetGeoTransform(geo_trans1)
        target_ds.SetProjection(raster.GetProjectionRef())
        # 如果都是边界空白区则不需要计算
        band_data = raster.GetRasterBand(1).ReadAsArray()
        band_data = band_data[cut_y_b:cut_y_e, cut_x_b:cut_x_e]
        # 以下命令过于占用内存，需要被优化，暂时都是不做裁切，忽略
        # if np.nanstd(band_data) < 0.001:
        #     return(None)
        for i in range(nbands):
            band_data = raster.GetRasterBand(i+1).ReadAsArray()
            band_data = band_data[cut_y_b:cut_y_e, cut_x_b:cut_x_e]
            target_ds.GetRasterBand(i+1).WriteArray(band_data)
        target_ds = None
    return([cut_x_b, cut_x_e, cut_y_b, cut_y_e])


if __name__ == '__main__':
    raster_file = r'D:\Job\ImageSky\proj\yihulianghai\daihai\data\radi_atms_corr\LC08_L1TP_126032_20180920_20180928_01_T1_rad_6s.tif'
    shp_file = r'D:\Job\ImageSky\proj\yihulianghai\daihai\Daihai_buffer\Daihai_WR.shp'
    out_file = raster_file.replace('.tif', '_cut.tif')
    print(main(raster_file=raster_file, shp_file=shp_file, sub_lim=None, out_file=out_file))
