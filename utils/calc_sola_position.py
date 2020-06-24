# coding=UTF-8
'''
@Author: cuiqiyuan
@Description: 计算太阳天顶角和方位角
@Date: 2019-07-30 15:26:29
'''
import ephem

def main(lon, lat, date):
    '''
    @lon {float} 观测者经度
    @lat {float} 观测者纬度
    @date {yyyy/mm/dd hh:mm:ss} 观测时间
    @return: [solz, sola]
    '''    
    gatech = ephem.Observer()
    gatech.lon = str(lon)
    gatech.lat = str(lat)
    gatech.date = date
    sun = ephem.Sun()
    sun.compute(gatech)
    solz = str(sun.alt)
    sola = str(sun.az)
    solz_split = solz.split(':')
    solz = float(solz_split[0]) + float(solz_split[1])/60 + float(solz_split[2])/3600
    sola_split = sola.split(':')
    sola = float(sola_split[0]) + float(sola_split[1])/60 + float(sola_split[2])/3600
    return([90-solz, sola])


if __name__ == '__main__':
    print(main(120, 31, '2018/7/30 7:0:0'))
    