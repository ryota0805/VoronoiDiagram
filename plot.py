import matplotlib.pyplot as plt
import matplotlib.patches as patches
from param import Parameter as p
import util
import numpy as np
import env


########
#壁と障害物の配置し表示する関数
########
def vis_env():
    fig, ax = plt.subplots()
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    """
    #startとgoalを配置
    ax.scatter([p.initial_x], [p.initial_y], marker='v', color='green', label='start')
    ax.scatter([p.terminal_x], [p.terminal_y], marker='^', color='green', label='goal')
    """
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    plt.show()
    
    return None
    
    
########    
#経路を環境に表示する関数
########
def vis_path(trajectory_vector):
    fig, ax = plt.subplots()
    
    #vectorをmatrixに変換
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    x, y = trajectory_matrix[0], trajectory_matrix[1]
    
    ax.scatter(x, y, marker='x', color='red', s=5)
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    #startとgoalを配置
    ax.scatter([x[0]], [y[0]], marker='v', color='green', label='start')
    ax.scatter([x[-1]], [y[-1]], marker='^', color='green', label='goal')
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    ax.legend(loc="best")
    plt.show()
    
    return None

########
#2本のpathを比較する関数
########
def compare_path(trajectory_vector1, trajectory_vector2):
    fig, ax = plt.subplots()
    
    #2本のpathを配置
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    x1, y1 = trajectory_matrix1[0], trajectory_matrix1[1]
    ax.scatter(x1, y1, marker='x', color='red', s=5, label='Initial path')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    x2, y2 = trajectory_matrix2[0], trajectory_matrix2[1]
    ax.scatter(x2, y2, marker='x', color='blue', s=5, label='Optimized path')
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    #startとgoalを配置
    ax.scatter([x1[0]], [y1[0]], marker='v', color='green', label='start')
    ax.scatter([x1[-1]], [y1[-1]], marker='^', color='green', label='goal')
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    ax.legend(loc="best")
    plt.show()
    
    return None
    
########
#姿勢thetaのグラフを生成
########
def vis_history_theta(trajectory_vector, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    
    theta = trajectory_matrix[2]
    
    ax.plot(theta, color='blue', label=r'$\theta$[rad]')
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$\theta$[rad]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        theta_max_list = [p.theta_max for i in range(p.N)]
        theta_min_list = [p.theta_min for i in range(p.N)]
        ax.plot(theta_max_list, color='green', linestyle='-.')
        ax.plot(theta_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    

########
#姿勢の比較
########
def compare_history_theta(trajectory_vector1, trajectory_vector2, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    theta1 = trajectory_matrix1[2]
    ax.plot(theta1,  color='red',  label='Initial')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    theta2 = trajectory_matrix2[2]
    ax.plot(theta2,  color='blue',  label='Optimized')
    
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$\theta$[rad]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        theta_max_list = [p.theta_max for i in range(p.N)]
        theta_min_list = [p.theta_min for i in range(p.N)]
        ax.plot(theta_max_list, color='green', linestyle='-.')
        ax.plot(theta_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    
 
########
#ステアリング角phiのグラフを生成
########
def vis_history_phi(trajectory_vector, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    
    phi = trajectory_matrix[3]
    
    ax.plot(phi, color='blue', label=r'$\phi$[rad]')
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$\phi$[rad]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        phi_max_list = [p.phi_max for i in range(p.N)]
        phi_min_list = [p.phi_min for i in range(p.N)]
        ax.plot(phi_max_list, color='green', linestyle='-.')
        ax.plot(phi_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    

########
#ステアリング角の比較
########
def compare_history_phi(trajectory_vector1, trajectory_vector2, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    phi1 = trajectory_matrix1[3]
    ax.plot(phi1,  color='red',  label='Initial')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    phi2 = trajectory_matrix2[3]
    ax.plot(phi2,  color='blue',  label='Optimized')
    
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$\phi$[rad]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        phi_max_list = [p.phi_max for i in range(p.N)]
        phi_min_list = [p.phi_min for i in range(p.N)]
        ax.plot(phi_max_list, color='green', linestyle='-.')
        ax.plot(phi_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    

########
#速さvのグラフを生成
########
def vis_history_v(trajectory_vector, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    
    v = trajectory_matrix[4]
    
    ax.plot(v, color='blue', label=r'$v$[m/s]')
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$v$[m/s]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        v_max_list = [p.v_max for i in range(p.N)]
        v_min_list = [p.v_min for i in range(p.N)]
        ax.plot(v_max_list, color='green', linestyle='-.')
        ax.plot(v_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    

########
#速さの比較
########
def compare_history_v(trajectory_vector1, trajectory_vector2, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    v1 = trajectory_matrix1[4]
    ax.plot(v1,  color='red',  label='Initial')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    v2 = trajectory_matrix2[4]
    ax.plot(v2,  color='blue',  label='Optimized')
    
    ax.set_xlabel(r'$t$[s]')
    ax.set_ylabel(r'$v$[m/s]')
    ax.legend(loc='upper right')
    
    #thetaの範囲を追加
    if range_flag:
        v_max_list = [p.v_max for i in range(p.N)]
        v_min_list = [p.v_min for i in range(p.N)]
        ax.plot(v_max_list, color='green', linestyle='-.')
        ax.plot(v_min_list, color='green', linestyle='-.')
    else:
        pass
    
    plt.show()
    
########
#2本のpathを比較する関数(障害物が正方形の場合)
########
def compare_path_rec(trajectory_vector1, trajectory_vector2):
    fig, ax = plt.subplots()
    
    #2本のpathを配置
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    x1, y1 = trajectory_matrix1[0], trajectory_matrix1[1]
    ax.scatter(x1, y1, marker='x', color='red', s=5, label='Initial path')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    x2, y2 = trajectory_matrix2[0], trajectory_matrix2[1]
    ax.scatter(x2, y2, marker='x', color='blue', s=5, label='Optimized path')
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obstacle_list = env_data.obs_circle
    
    #wallを配置
    for k in range(len(obstacle_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obstacle_list)):
        x_o, y_o, r_o = obstacle_list[k][0], obstacle_list[k][1], obstacle_list[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
        
    #startとgoalを配置
    ax.scatter([p.initial_x], [p.initial_y], marker='v', color='green', label='start')
    ax.scatter([p.terminal_x], [p.terminal_y], marker='^', color='green', label='goal')
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    #ax.legend(loc="best")
    plt.show()
    
    return None

def generate_network(trajectory_vectors, success_list):
    fig, ax = plt.subplots()
    
    for trajectory_vector, success in zip(trajectory_vectors, success_list):
        if success == True:
            trajectory_matrix = util.vector_to_matrix(trajectory_vector)
            x, y = trajectory_matrix[0], trajectory_matrix[1]
            ax.plot(x, y, color='red', marker='', zorder=1)
            ax.scatter([x[0]], [y[0]], marker='.', color='green', zorder=2)
            ax.scatter([x[-1]], [y[-1]], marker='.', color='green', zorder=2)
        else:
            trajectory_matrix = util.vector_to_matrix(trajectory_vector)
            x, y = trajectory_matrix[0], trajectory_matrix[1]
            ax.plot(x, y, color='blue', marker='', zorder=1)
            ax.scatter([x[0]], [y[0]], marker='.', color='green', zorder=2)
            ax.scatter([x[-1]], [y[-1]], marker='.', color='green', zorder=2)
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    #ax.legend(loc="best")
    plt.show()
    
    return None