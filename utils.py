import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import env
import math
from param import Parameter as p
import heapq

#edgeの両端点のどちらかが無限遠(-1)か障害物の領域内にあればFalse,そうでなければTrueを返す
def generate_valid_edge(ridge_vertices, vertices):
    edge = []
    
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    for edge_index in ridge_vertices:
        flag = True
       
        if -1 in edge_index:
            flag = False
            continue
        
        vertice1 = vertices[edge_index[0]]
        vertice2 = vertices[edge_index[1]]
        
        for i in range(len(obs_rectangle)):
            if (obs_rectangle[i][0] < vertice1[0] < obs_rectangle[i][0] + obs_rectangle[i][2]) and (obs_rectangle[i][1] < vertice1[1] < obs_rectangle[i][1] + obs_rectangle[i][3]):
                flag = False
            
            elif (obs_rectangle[i][0] < vertice2[0] < obs_rectangle[i][0] + obs_rectangle[i][2]) and (obs_rectangle[i][1] < vertice2[1] < obs_rectangle[i][1] + obs_rectangle[i][3]):
                flag = False
        
        for i in range(len(obs_circle)):
            x0, y0, r = obs_circle[i][0], obs_circle[i][1], obs_circle[i][2]
            if ((vertice1[0] - x0) ** 2 + (vertice1[1] - y0) ** 2 < r ** 2) or ((vertice2[0] - x0) ** 2 + (vertice2[1] - y0) ** 2 < r ** 2):
                flag = False
                
        if flag == True:
            edge.append(edge_index)
            
            
    return edge


#重み付き隣接リスト型のグラフを作成する関数
def generate_adjacency_list(edge, vertices):
    graph = [[] for _ in range(len(vertices))]
    
    for edge_index in edge:
        vertice1 = vertices[edge_index[0]]
        vertice2 = vertices[edge_index[1]]
        
        distance = ((vertice2[0] - vertice1[0]) ** 2 + (vertice2[1] - vertice1[1]) ** 2) ** 0.5
        
        graph[edge_index[0]].append([edge_index[1], distance])
        graph[edge_index[1]].append([edge_index[0], distance])  # 有向グラフなら消す
        
    return graph


def dijkstra(graph, start, goal):
    # グラフの頂点数
    num_vertices = len(graph)
    
    # 各頂点までの最短距離を無限大に初期化
    distance = [float('inf')] * num_vertices
    
    # 始点から始点への距離は0に設定
    distance[start] = 0
    
    # プライオリティキューを初期化し、始点を追加
    priority_queue = [(0, start)]
    
    # 各頂点への最短経路を保存するリスト
    shortest_path = [None] * num_vertices
    
    while priority_queue:
        # プライオリティキューから最も距離が短い頂点を取得
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # ゴールに到達した場合、最短経路を構築して返す
        if current_vertex == goal:
            path = []
            while current_vertex is not None:
                path.append(current_vertex)
                current_vertex = shortest_path[current_vertex]
            return distance[goal], list(reversed(path))
        
        # 現在の頂点からの距離が既知の最短距離よりも長ければスキップ
        if current_distance > distance[current_vertex]:
            continue
        
        # 隣接する頂点を探索
        for neighbor, weight in graph[current_vertex]:
            # 新しい距離を計算
            distance_to_neighbor = distance[current_vertex] + weight
            
            # より短い距離を発見した場合、更新
            if distance_to_neighbor < distance[neighbor]:
                distance[neighbor] = distance_to_neighbor
                # 最短経路情報を更新
                shortest_path[neighbor] = current_vertex
                # プライオリティキューに追加
                heapq.heappush(priority_queue, (distance[neighbor], neighbor))
    
    # ゴールに到達できなかった場合、最短距離と経路は存在しない
    return float('inf'), []