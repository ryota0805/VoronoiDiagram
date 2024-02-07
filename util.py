from param import Parameter as p
import numpy as np

########
#設計変数の行列(M×N)をベクトル(1×MN)に変換する関数
########
def matrix_to_vector(trajectory_matrix):
    
    trajectory_vector = trajectory_matrix.flatten()
    
    return trajectory_vector

########
#設計変数のベクトル(1×MN)を行列(M×N)をに変換する関数
########
def vector_to_matrix(trajectory_vector):
    
    trajectory_matrix = trajectory_vector.reshape(p.M, int(len(trajectory_vector)/p.M))
    
    return trajectory_matrix


########
#最適化結果を各変数のベクトルに変換
########
def generate_result(trajectory_vector):
    
    trajectory_matrix = trajectory_vector.reshape(p.M, int(len(trajectory_vector)/p.M))
    x, y, theta, phi, v = trajectory_matrix[0], trajectory_matrix[1], trajectory_matrix[2], trajectory_matrix[3], trajectory_matrix[4]
    
    return x, y, theta, phi, v

    