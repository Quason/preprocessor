# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: 地理坐标和投影坐标的相互转换
@Date: 2019-07-05 15:08:36
'''
from osgeo import osr

def trans(EPSGs, EPSGt, x, y):
    '''
    @description: 投影坐标转地理坐标
    @EPSGs: 原始坐标系
    @EPSGt: 目标坐标系
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


if __name__ == '__main__':
    print(trans(32651, 4326, 295065, 3471165))
