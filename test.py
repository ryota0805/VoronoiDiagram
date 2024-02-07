import heapq

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

# グラフの隣接リストを定義
graph = [
    [(1, 4), (7, 8)],
    [(0, 4), (2, 8), (7, 11)],
    [(1, 8), (3, 7), (5, 4), (8, 2)],
    [(2, 7), (4, 9), (5, 14)],
    [(3, 9), (5, 10)],
    [(2, 4), (3, 14), (6, 2)],
    [(5, 2), (7, 1), (8, 6)],
    [(0, 8), (1, 11), (6, 1), (8, 7)],
    [(2, 2), (6, 6), (7, 7)]
]

# スタート地点とゴール地点を指定
start_vertex = 0
goal_vertex = 4

# スタート地点からゴール地点までの最短距離と経路を計算
shortest_distance, shortest_path = dijkstra(graph, start_vertex, goal_vertex)

print(shortest_path)
# 結果を表示
if shortest_distance != float('inf'):
    print(f"頂点 {start_vertex} から頂点 {goal_vertex} までの最短距離: {shortest_distance}")
    print(f"最短経路: {' -> '.join(map(str, shortest_path))}")
else:
    print(f"頂点 {start_vertex} から頂点 {goal_vertex} への経路は存在しません。")