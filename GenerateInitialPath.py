#初期パスを生成するファイル

import numpy as np
from scipy import interpolate
from param import Parameter as p

########
#WayPointから3次スプライン関数を生成し、状態量をサンプリングする
########

#3次スプライン関数の生成
def cubic_spline():   
    x, y = [], []
    for i in range(len(p.WayPoint)):
        x.append(p.WayPoint[i][0])
        y.append(p.WayPoint[i][1])
        
    tck,u = interpolate.splprep([x,y], k=3, s=0) 
    u = np.linspace(0, 1, num=p.N, endpoint=True)
    spline = interpolate.splev(u, tck)
    cubicX = spline[0]
    cubicY = spline[1]
    return cubicX, cubicY

#直線補間の生成
def interp_1d(waypoint):
    x1, y1 = waypoint[0][0], waypoint[0][1]
    x2, y2 = waypoint[1][0], waypoint[1][1]
    
    x_list = []
    y_list = []
    
    for i in range(p.N):
        x = x1 + (x2 - x1) * i / (p.N - 1)
        y = y1 + (y2 - y1) * i / (p.N - 1)
        
        x_list.append(x)
        y_list.append(y)
        
    return x_list, y_list
    

#3次スプライン関数の生成(経路が関数の引数として与えられる場合)
def cubic_spline_by_waypoint(waypoint):   
    if len(waypoint) == 2:
        k = 1
    elif len(waypoint) == 3:
        k = 2
    else:
        k = 3
    
    x, y = [], []
    for i in range(len(waypoint)):
        x.append(waypoint[i][0])
        y.append(waypoint[i][1])
        
    tck,u = interpolate.splprep([x,y], k=k, s=0) 
    u = np.linspace(0, 1, num=p.N, endpoint=True)
    spline = interpolate.splev(u, tck)
    cubicX = spline[0]
    cubicY = spline[1]
    return cubicX, cubicY

#x, yからΘとφを生成する
def generate_initialpath(cubicX, cubicY):
    #nd.arrayに変換
    x = np.array(cubicX)
    y = np.array(cubicY)
    
    #x, yの差分を計算
    deltax = np.diff(x)
    deltay = np.diff(y)
    
    #x, y の差分からthetaを計算
    #theta[0]を初期値に置き換え、配列の最後に終端状態を追加
    theta = np.arctan(deltay / deltax)
    theta[0] = p.initial_theta
    theta = np.append(theta, p.terminal_theta)
    
    #thetaの差分からphiを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    deltatheta = np.diff(theta)
    phi = deltatheta / p.dt
    phi[0] = p.initial_phi
    phi = np.append(phi, p.terminal_phi)
    
    #x,yの差分からvを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    v = np.sqrt((deltax ** 2 + deltay ** 2) / p.dt)
    v[0] = p.initial_v
    v = np.append(v, p.terminal_v)
    return x, y, theta, phi, v


