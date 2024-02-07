from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import env
import math
from param import Parameter as p
import utils
import GenerateInitialPath
import objective_function
import constraints
import util
import scipy.optimize as optimize
import plot

#障害物情報からpointsを設定する
env_data = env.Env()

points = []
wall_list = env_data.obs_boundary
obs_circle = env_data.obs_circle
obs_rectangle = env_data.obs_rectangle
wall_div_num = 100
rectangle_div_num = 10
circle_div_num = 72

x_range = env_data.x_range
y_range = env_data.y_range

points.append((x_range[0], y_range[0]))
points.append((x_range[1], y_range[0]))
points.append((x_range[0], y_range[1]))
points.append((x_range[1], y_range[1]))

width = x_range[1] - x_range[0]
height = y_range[1] - y_range[0]

for i in range(1, wall_div_num):
    x = x_range[0] + width * i / wall_div_num
    y = y_range[0]
    points.append((x, y))
    
for i in range(1, wall_div_num):
    x = x_range[0] 
    y = y_range[0] + height * i / wall_div_num
    points.append((x, y))
    
for i in range(1, wall_div_num):
    x = x_range[0] + width * i / wall_div_num
    y = y_range[1] 
    points.append((x, y))

for i in range(1, wall_div_num):
    x = x_range[1] 
    y = y_range[0] + height * i / wall_div_num
    points.append((x, y))
    
for i in range(len(obs_circle)):
    x0, y0, r0 = obs_circle[i][0], obs_circle[i][1], obs_circle[i][2]
    for j in range(circle_div_num):
        x = x0 + r0 * math.cos(2 * math.pi * j / circle_div_num)
        y = y0 + r0 * math.sin(2 * math.pi * j / circle_div_num)
        points.append((x, y))
        
for i in range(len(obs_rectangle)):
    x_left_down = obs_rectangle[i][0]
    y_left_down = obs_rectangle[i][1]
    
    x_left_up = obs_rectangle[i][0] 
    y_left_up = obs_rectangle[i][1] + obs_rectangle[i][3]
    
    x_right_down = obs_rectangle[i][0] + obs_rectangle[i][2]
    y_right_down = obs_rectangle[i][1]
    
    x_right_up = obs_rectangle[i][0] + obs_rectangle[i][2]
    y_right_up = obs_rectangle[i][1] + obs_rectangle[i][3]
    
    points.append((x_left_down, y_left_down))
    points.append((x_left_up, y_left_up))
    points.append((x_right_down, y_right_down))
    points.append((x_right_up, y_right_up))
        
    width = obs_rectangle[i][2]
    height = obs_rectangle[i][3]
    
    for i in range(1, rectangle_div_num):
        x = x_left_down + width * i / rectangle_div_num
        y = y_left_down
        points.append((x, y))
    
    for i in range(1, rectangle_div_num):
        x = x_left_down 
        y = y_left_down + height * i / rectangle_div_num
        points.append((x, y))
        
    for i in range(1, rectangle_div_num):
        x = x_left_down + width * i / rectangle_div_num
        y = y_left_up 
        points.append((x, y))

    for i in range(1, rectangle_div_num):
        x = x_right_down 
        y = y_left_down + height * i / rectangle_div_num
        points.append((x, y))
    
vor = Voronoi(points)

# エッジ情報を表示
for ridge_points in vor.ridge_vertices:
    if -1 not in ridge_points:  # ボロノイ図の外側のエッジは -1 で表現されるためスキップ
        start_point = vor.vertices[ridge_points[0]]
        end_point = vor.vertices[ridge_points[1]]
        print(f"Start Point: {start_point}, End Point: {end_point}")
        

fig, ax = plt.subplots()
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
    


fig = voronoi_plot_2d(vor, ax=ax)
ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
ax.set_aspect('equal')

plt.show()

print(vor.vertices)            
print(vor.ridge_vertices)


edge = utils.generate_valid_edge(vor.ridge_vertices, vor.vertices)
print("ここから")
print(edge)

fig, ax = plt.subplots()
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
    
for i in range(len(edge)):
    ax.plot([vor.vertices[edge[i][0]][0], vor.vertices[edge[i][1]][0]], [vor.vertices[edge[i][0]][1], vor.vertices[edge[i][1]][1]], color='r', marker='')
    
ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
ax.set_aspect('equal')

plt.show()

print("ここから")
graph = utils.generate_adjacency_list(edge, vor.vertices)
print(graph)

intersection_index = []

for i in range(len(graph)):
    if len(graph[i]) >= 3:
        intersection_index.append(i)
        

for i in intersection_index:
    print("Index:{} ({}, {})".format(i, vor.vertices[i][0], vor.vertices[i][1]))
    
intersection_list_x=[]
intersection_list_y=[]
for i in intersection_index:
    intersection_list_x.append(vor.vertices[i][0])
    intersection_list_y.append(vor.vertices[i][1])

fig, ax = plt.subplots()
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
    
for i in range(len(edge)):
    ax.plot([vor.vertices[edge[i][0]][0], vor.vertices[edge[i][1]][0]], [vor.vertices[edge[i][0]][1], vor.vertices[edge[i][1]][1]], color='r', marker='', zorder=1)
    
ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
ax.set_aspect('equal')

    
ax.scatter(intersection_list_x,intersection_list_y,zorder=2)
plt.show()

network = []

for i in range(0, len(intersection_index)-1):
    for j in range(i+1, len(intersection_index)):
        start = intersection_index[i]
        goal = intersection_index[j]
        _, shortest_path = utils.dijkstra(graph, start, goal)
        network.append(shortest_path)

print(len(network))

trajectory_vectors = []
for index_path in network:
    waypoint = []
    for index in index_path:
        vertice = [vor.vertices[index][0], vor.vertices[index][1]]
        waypoint.append(vertice)
        
    print(waypoint)
    if waypoint[0][0] < waypoint[-1][0]:
        pass
    else:
        waypoint.reverse()
        
    p.initial_x, p.terminal_x = waypoint[0][0], waypoint[-1][0]
    p.initial_y, p.terminal_y = waypoint[0][1], waypoint[-1][1]
    
    distance = ((p.initial_x - p.terminal_x) ** 2 + (p.initial_y - p.terminal_y) ** 2) ** 0.5
    p.N = int(distance)
    if p.N == 1:
        p.N = 2
    print(p.N)
    cubicX, cubicY = GenerateInitialPath.cubic_spline_by_waypoint(waypoint)
    x, y, theta, phi, v = GenerateInitialPath.generate_initialpath(cubicX, cubicY)
    theta, phi, v = np.array([0.1]*p.N), np.array([0.1]*p.N), np.array([0.1]*p.N)
    trajectory_matrix = np.array([x, y, theta, phi, v])
    trajectory_vector = util.matrix_to_vector(trajectory_matrix)
    
    
    #目的関数の設定
    func = objective_function.objective_function
    jac_of_objective_function = objective_function.jac_of_objective_function

    #制約条件の設定
    cons = constraints.generate_cons_with_jac()
    #cons = constraints.generate_constraints()

    #変数の範囲の設定
    bounds = constraints.generate_bounds()

    #オプションの設定
    options = {'maxiter':10000}
    
    #最適化を実行
    result = optimize.minimize(func, trajectory_vector, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)
    #result = optimize.minimize(func, trajectory_vector, method='SLSQP', constraints=cons, bounds=bounds, options=options)
    print(result)
    #plot.compare_path(trajectory_vector, result.x)
    #plot.compare_history_theta(trajectory_vector, result.x, range_flag = True)
    #plot.compare_history_phi(trajectory_vector, result.x, range_flag = True)
    #plot.compare_history_v(trajectory_vector, result.x, range_flag = True)
    trajectory_vectors.append(result.x)
    
plot.generate_network(trajectory_vectors)

