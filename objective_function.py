from param import Parameter as p
import numpy as np

def objective_function(x):
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += (trajectory_matrix[3, i] ** 2 / p.phi_max ** 2) + (trajectory_matrix[4, i] ** 2 / p.v_max ** 2) 
    
    return sum / p.N

def objective_function2(x):
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += (trajectory_matrix[3, i] ** 2 / p.phi_max ** 2) * (trajectory_matrix[4, i] ** 2 / p.v_max ** 2) 
    
    return sum / p.N

def jac_of_objective_function(x):
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    jac_f = np.zeros((p.M, p.N))

    for i in range(p.N):
        #phiの微分
        jac_f[3, i] = (trajectory_matrix[3, i] * 2) / (p.N * (p.phi_max ** 2))  
    
        #vの微分
        jac_f[4, i] = (trajectory_matrix[4, i] * 2) / (p.N * (p.v_max ** 2)) 

    #ベクトルに直す
    jac_f = jac_f.flatten()
    
    return jac_f