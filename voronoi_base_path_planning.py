import env
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import utils
import GenerateInitialPath
import objective_function
import constraints
from param import Parameter as p
import util
import scipy.optimize as optimize
import plot
import csv


class VoronoiSQPbasePathPlanning:
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_circle = env_data.obs_circle
    obs_rectangle = env_data.obs_rectangle
    wall_div_num = 100
    rectangle_div_num = 20
    circle_div_num = 72
        
    def __init__(self, start, goal, theta):
        self.start = start
        self.goal = goal
        self.theta = theta
        
    def generate_voronoi(self):
        points = []
        x_range = self.env_data.x_range 
        y_range = self.env_data.y_range 
        
        points.append((x_range[0], y_range[0]))
        points.append((x_range[1], y_range[0]))
        points.append((x_range[0], y_range[1]))
        points.append((x_range[1], y_range[1]))
        
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        for i in range(1, self.wall_div_num):
            x = x_range[0] + width * i / self.wall_div_num
            y = y_range[0]
            points.append((x, y))
            
        for i in range(1, self.wall_div_num):
            x = x_range[0] 
            y = y_range[0] + height * i / self.wall_div_num
            points.append((x, y))
            
        for i in range(1, self.wall_div_num):
            x = x_range[0] + width * i / self.wall_div_num
            y = y_range[1] 
            points.append((x, y))

        for i in range(1, self.wall_div_num):
            x = x_range[1] 
            y = y_range[0] + height * i / self.wall_div_num
            points.append((x, y))
            
        for i in range(len(self.obs_circle)):
            x0, y0, r0 = self.obs_circle[i][0], self.obs_circle[i][1], self.obs_circle[i][2]
            for j in range(self.circle_div_num):
                x = x0 + r0 * math.cos(2 * math.pi * j / self.circle_div_num)
                y = y0 + r0 * math.sin(2 * math.pi * j / self.circle_div_num)
                points.append((x, y))
                
        for i in range(len(self.obs_rectangle)):
            x_left_down = self.obs_rectangle[i][0]
            y_left_down = self.obs_rectangle[i][1]
            
            x_left_up = self.obs_rectangle[i][0] 
            y_left_up = self.obs_rectangle[i][1] + self.obs_rectangle[i][3]
            
            x_right_down = self.obs_rectangle[i][0] + self.obs_rectangle[i][2]
            y_right_down = self.obs_rectangle[i][1]
            
            x_right_up = self.obs_rectangle[i][0] + self.obs_rectangle[i][2]
            y_right_up = self.obs_rectangle[i][1] + self.obs_rectangle[i][3]
            
            points.append((x_left_down, y_left_down))
            points.append((x_left_up, y_left_up))
            points.append((x_right_down, y_right_down))
            points.append((x_right_up, y_right_up))
                
            width = self.obs_rectangle[i][2]
            height = self.obs_rectangle[i][3]
            
            for i in range(1, self.rectangle_div_num):
                x = x_left_down + width * i / self.rectangle_div_num
                y = y_left_down
                points.append((x, y))
            
            for i in range(1, self.rectangle_div_num):
                x = x_left_down 
                y = y_left_down + height * i / self.rectangle_div_num
                points.append((x, y))
                
            for i in range(1, self.rectangle_div_num):
                x = x_left_down + width * i / self.rectangle_div_num
                y = y_left_up 
                points.append((x, y))

            for i in range(1, self.rectangle_div_num):
                x = x_right_down 
                y = y_left_down + height * i / self.rectangle_div_num
                points.append((x, y))

        vor = Voronoi(points)
            
        return vor
    
    def generate_graph(self, vor):
        edge = utils.generate_valid_edge(vor.ridge_vertices, vor.vertices)
        graph = utils.generate_adjacency_list(edge, vor.vertices)
        
        return graph
    
    def generate_network(self, graph, vor):
        intersection_index = []

        for i in range(len(graph)):
            if len(graph[i]) >= 3:
                intersection_index.append(i)
            
        intersection_list_x=[]
        intersection_list_y=[]
        for i in intersection_index:
            intersection_list_x.append(vor.vertices[i][0])
            intersection_list_y.append(vor.vertices[i][1])
        
        network = []

        for i in range(0, len(intersection_index)-1):
            for j in range(i+1, len(intersection_index)):
                start = intersection_index[i]
                goal = intersection_index[j]
                _, shortest_path = utils.dijkstra(graph, start, goal)
                network.append(shortest_path)
                
        trajectory_vectors = []
        success_list = []
        counter = 0
        for index_path in network:
            counter += 1
            waypoint = []
            for index in index_path:
                vertice = [vor.vertices[index][0], vor.vertices[index][1]]
                waypoint.append(vertice)

            if waypoint[0][0] < waypoint[-1][0]:
                pass
            else:
                waypoint.reverse()
                
            p.initial_x, p.terminal_x = waypoint[0][0], waypoint[-1][0]
            p.initial_y, p.terminal_y = waypoint[0][1], waypoint[-1][1]
            
            distance = ((p.initial_x - p.terminal_x) ** 2 + (p.initial_y - p.terminal_y) ** 2) ** 0.5
            p.N = int(distance)

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
            trajectory_vectors.append(result.x)
            success_list.append(result.success)
            print("{}/{}".format(counter, len(network)))
            
        return trajectory_vectors, success_list
    
    
    def path_planning(self, trajectory_vectors):
        start_distance = []
        goal_distance = []

        for trajectory_vector in trajectory_vectors:
            x, y, _, _ ,_ = util.generate_result(trajectory_vector)
            s_distance = (self.start[0]- x[0]) ** 2 + (self.start[1] - y[0]) ** 2
            g_distance = (self.goal[0]- x[-1]) ** 2 + (self.goal[1] - y[-1]) ** 2
            start_distance.append(s_distance)
            goal_distance.append(g_distance)
            
        start_index = start_distance.index(min(start_distance))
        goal_index = goal_distance.index(min(goal_distance))

        xs, ys, _, _ ,_ = util.generate_result(trajectory_vectors[start_index])
        xg, yg, _, _ ,_ = util.generate_result(trajectory_vectors[goal_index])

        for trajectory_vector in trajectory_vectors:
            x, y, _, _ ,_ = util.generate_result(trajectory_vector)
            if x[0] == xs[0] and y[0] == y[0] and x[-1] == xg[-1] and y[-1] == yg[-1]:
                middle_trajectory_vector = trajectory_vector
            else:
                pass

        
        ####start->middle
        xs, ys, theta_s, _ ,_ = util.generate_result(middle_trajectory_vector)
        p.initial_x, p.terminal_x = self.start[0], xs[0]
        p.initial_y, p.terminal_y = self.start[1], ys[0]
        p.initial_theta, p.terminal_theta = self.theta[0], theta_s[0] 
        p.set_cons['initial_theta'] = True
        p.set_cons['terminal_theta'] = True
        distance = ((p.initial_x - p.terminal_x) ** 2 + (p.initial_y - p.terminal_y) ** 2) ** 0.5
        p.N = int(distance * 1.5)

        #初期経路生成
        x_list, y_list = GenerateInitialPath.interp_1d()
        x, y, theta, phi, v = GenerateInitialPath.generate_initialpath(x_list, y_list)
        #theta, phi, v = np.array([0.1]*p.N), np.array([0.1]*p.N), np.array([0.1]*p.N)
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
        
        start_trajectory_vector = result.x
        
        ####middle->goal
        xg, yg, theta_g, _ ,_ = util.generate_result(middle_trajectory_vector)
        p.initial_x, p.terminal_x = xg[-1], self.goal[0]
        p.initial_y, p.terminal_y = yg[-1], self.goal[1]
        p.initial_theta, p.terminal_theta = theta_g[-1], theta[1]
        p.set_cons['initial_theta'] = True
        p.set_cons['terminal_theta'] = True
        
        distance = ((p.initial_x - p.terminal_x) ** 2 + (p.initial_y - p.terminal_y) ** 2) ** 0.5
        p.N = int(distance * 1.5)

        #初期経路生成
        x_list, y_list = GenerateInitialPath.interp_1d()
        x, y, theta, phi, v = GenerateInitialPath.generate_initialpath(x_list, y_list)
        #theta, phi, v = np.array([0.1]*p.N), np.array([0.1]*p.N), np.array([0.1]*p.N)
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
        goal_trajectory_vector = result.x
        
        return middle_trajectory_vector, start_trajectory_vector, goal_trajectory_vector
    

    
def main():
    start = [-2.5, 9]
    goal = [32, -6]
    theta = [0, 0]
    
    planner = VoronoiSQPbasePathPlanning(start, goal, theta)
    vor = planner.generate_voronoi()
    graph = planner.generate_graph(vor)
    trajectory_vectors, _ = planner.generate_network(graph, vor)
    middle_path, start_path, goal_path = planner.path_planning(trajectory_vectors)
    
    all_path = [middle_path, start_path, goal_path]
    plot.generate_network(trajectory_vectors)
    plot.generate_network(all_path)
    x, y, _, _, _ = util.generate_result(middle_path)
    way_point = []
    for i in range(len(x)):
        way_point.append([x[i], y[i]])
    print(way_point)
#main()

def save_network():
    start = [-2.5, 9]
    goal = [32, -6]
    theta = [0, 0]

    planner = VoronoiSQPbasePathPlanning(start, goal, theta)
    vor = planner.generate_voronoi()
    graph = planner.generate_graph(vor)
    trajectory_vectors, success_list = planner.generate_network(graph, vor)
    plot.generate_network(trajectory_vectors, success_list)
    csv_path = "network_complex2.csv"
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(trajectory_vectors)
    print(success_list)
save_network()

List = [True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, 
 True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, 
 True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, 
 True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, 
 True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, 
 False, True, True, True, True, False, True, True, True, True, True, True, False, True, True, 
 True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, 
 True, True, True, True, True, False, True, True, True, True, True, True, True, True, True]
