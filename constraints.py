#不等式制約、等式制約を定義する
from param import Parameter as p
import util
import numpy as np
import env

########
#制約条件を生成する関数
########
def generate_constraints():
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #最初に不等式制約(K×N個)
    cons = ()
    
    #矩形の障害物に対する不等式制約
    for k in range(len(obs_rectangle)):
        for i in range(p.N):
            cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: (((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 10 + ((2*0.8/obs_rectangle[k][3]) ** 10) * (x[i + p.N] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 10) - 1},)
    
    #円形の障害物に対する不等式制約
    for k in range(len(obs_circle)):
        for i in range(p.N):
            cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: ((x[i] - obs_circle[k][0]) ** 2 + (x[i + p.N] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + p.robot_size) ** 2},)


    #次にモデルの等式制約(3×(N-1)個)
    #x
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1] - (x[i] + x[i + 4 * p.N] * np.cos(x[i + 2 * p.N]) * p.dt)},)
        
    #y
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + p.N] - (x[i + p.N] + x[i + 4 * p.N] * np.sin(x[i + 2 * p.N]) * p.dt)},)
        
    #theta
    for i in range(p.N-1):
        cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + 2 * p.N] - (x[i + 2 * p.N] + x[i + 4 * p.N] * np.tan(x[i+ 3 * p.N]) * p.dt / p.L)},)

    #境界条件(8個)
    #境界条件が設定されている場合は制約条件に加える。
    #x初期条件
    if p.set_cons['initial_x'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[0] - p.initial_x},)
        
    #x終端条件
    if p.set_cons['terminal_x'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N - 1] - p.terminal_x},)

    #y初期条件
    if p.set_cons['initial_y'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N] - p.initial_y},)
        
    #y終端条件
    if p.set_cons['terminal_y'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N - 1] - p.terminal_y},)
        
    #theta初期条件
    if p.set_cons['initial_theta'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N] - p.initial_theta},)
        
    #theta終端条件
    if p.set_cons['terminal_theta'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N - 1] - p.terminal_theta},)
        
    #phi初期条件
    if p.set_cons['initial_phi'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N] - p.initial_phi},)
        
    #phi終端条件
    if p.set_cons['terminal_phi'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[4*p.N - 1] - p.terminal_phi},)
        
    #v初期条件
    if p.set_cons['initial_v'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[4*p.N] - p.initial_v},)
        
    #v終端条件
    if p.set_cons['terminal_v'] == False:
        pass
    else:
        cons = cons + ({'type':'eq', 'fun':lambda x: x[5*p.N - 1] - p.terminal_v},)

    return cons


########
#bounds(変数の範囲)を設定する関数
########

#変数の数だけタプルのリストとして返す関数
def generate_bounds():
    
    #boundsのリストを生成
    bounds = []
    
    #xの範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #yの範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #thetaの範囲
    for i in range(p.N):
        bounds.append((p.theta_min, p.theta_max))
        
    #phiの範囲
    for i in range(p.N):
        bounds.append((p.phi_min, p.phi_max))
        
    #vの範囲
    for i in range(p.N):
        bounds.append((p.v_min, p.v_max))
        
    return bounds




def jac_of_constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    trajectory_matrix = x.reshape(p.M, p.N)
    x, y, theta, phi, v = trajectory_matrix[0, :], trajectory_matrix[1, :], trajectory_matrix[2, :], trajectory_matrix[3, :], trajectory_matrix[4, :]
    
    jac_cons = np.zeros((p.M, p.N))
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x':
            jac_cons[0, i] = -1
            jac_cons[0, i + 1] = 1
            jac_cons[2, i] = v[i] * np.sin(theta[i]) * p.dt
            jac_cons[4, i] = -np.cos(theta[i]) * p.dt
        
        elif args[1][0] == 'y':
            jac_cons[1, i] = -1
            jac_cons[1, i + 1] = 1
            jac_cons[2, i] = -v[i] * np.cos(theta[i]) * p.dt
            jac_cons[4, i] = -np.sin(theta[i]) * p.dt
            
        elif args[1][0] == 'theta':
            jac_cons[2, i] = -1
            jac_cons[2, i+1] = 1
            jac_cons[3, i] = (-v[i] * p.dt) / (p.L * (np.cos(phi[i]) ** 2))
            jac_cons[4, i] = (-np.tan(phi[i]) * p.dt) / p.L
            
        else:
            return 'Error'
        
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
    
        return jac_cons
    
    elif args[0] == 'avoid_obstacle':
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            jac_cons[0, i] = 10 * ((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 9
            jac_cons[1, i] = 10 * ((2*0.8/obs_rectangle[k][3]) ** 10) * (y[i] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 9
            
            #ベクトルに直す
            jac_cons = jac_cons.flatten()
        
            return jac_cons
            
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            jac_cons[0, i] = 2 * (x[i] - obs_circle[k][0])
            jac_cons[1, i] = 2 * (y[i] - obs_circle[k][1])
            
            #ベクトルに直す
            jac_cons = jac_cons.flatten()
        
            return jac_cons
    
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x':
            if ini_ter == 'ini':
                jac_cons[0, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[0, -1] = 1
                
        elif variable == 'y':
            if ini_ter == 'ini':
                jac_cons[1, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[1, -1] = 1  
                
        elif variable == 'theta':
            if ini_ter == 'ini':
                jac_cons[2, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[2, -1] = 1  
                
        elif variable == 'phi':
            if ini_ter == 'ini':
                jac_cons[3, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[3, -1] = 1
                
        elif variable == 'v':
            if ini_ter == 'ini':
                jac_cons[4, 0] = 1
            
            elif ini_ter == 'ter':
                jac_cons[4, -1] = 1    
        
        #ベクトルに直す
        jac_cons = jac_cons.flatten()
    
        return jac_cons
    
    
def constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    trajectory_matrix = x.reshape(p.M, p.N)
    x, y, theta, phi, v = trajectory_matrix[0, :], trajectory_matrix[1, :], trajectory_matrix[2, :], trajectory_matrix[3, :], trajectory_matrix[4, :]
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x':
            value = x[i+1] - (x[i] + v[i] * np.cos(theta[i]) * p.dt)
        
        elif args[1][0] == 'y':
            value = y[i+1] - (y[i] + v[i] * np.sin(theta[i]) * p.dt)
            
        elif args[1][0] == 'theta':
            value = theta[i+1] - (theta[i] + v[i] * np.tan(phi[i]) * p.dt / p.L)
             
        else:
            return 'Error'

        return value

    elif args[0] == 'avoid_obstacle':
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            
            value = (((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 10 + ((2*0.8/obs_rectangle[k][3]) ** 10) * (y[i] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 10) - 1
            
            return value
        
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            value = ((x[i] - obs_circle[k][0]) ** 2 + (y[i] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + p.robot_size) ** 2

            return value 
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x':
            if ini_ter == 'ini':
                value = x[0] - p.initial_x
            
            elif ini_ter == 'ter':
                value = x[-1] - p.terminal_x
                
        elif variable == 'y':
            if ini_ter == 'ini':
                value = y[0] - p.initial_y
            
            elif ini_ter == 'ter':
                value = y[-1] - p.terminal_y  
                
        elif variable == 'theta':
            if ini_ter == 'ini':
                value = theta[0] - p.initial_theta
            
            elif ini_ter == 'ter':
                value = theta[-1] - p.terminal_theta 
                
        elif variable == 'phi':
            if ini_ter == 'ini':
                value = phi[0] - p.initial_phi
            
            elif ini_ter == 'ter':
                value = phi[-1] - p.terminal_phi
                
        elif variable == 'v':
            if ini_ter == 'ini':
                value = v[0] - p.initial_v
            
            elif ini_ter == 'ter':
                value = v[-1] - p.terminal_v    
    
    
        return value
    
    
def generate_cons_with_jac():
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    cons = ()
    
    #障害物回避のための不等式制約を追加する
    #矩形
    for k in range(len(obs_rectangle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['rectangle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
    #円形
    for k in range(len(obs_circle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['circle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
            
    
    #運動学モデルの制約からなる等式制約を追加する
    #x
    for i in range(p.N-1):
        args = ['model', ['x', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y
    for i in range(p.N-1):
        args = ['model', ['y', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    #theta
    for i in range(p.N-1):
        args = ['model', ['theta', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        
        
        
        
    #境界値条件の等式制約を追加
    if p.set_cons['initial_x'] == False:
        pass
    else:
        args = ['boundary', ['x', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x終端条件
    if p.set_cons['terminal_x'] == False:
        pass
    else:
        args = ['boundary', ['x', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)

    #y初期条件
    if p.set_cons['initial_y'] == False:
        pass
    else:
        args = ['boundary', ['y', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y終端条件
    if p.set_cons['terminal_y'] == False:
        pass
    else:
        args = ['boundary', ['y', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta初期条件
    if p.set_cons['initial_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta終端条件
    if p.set_cons['terminal_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #phi初期条件
    if p.set_cons['initial_phi'] == False:
        pass
    else:
        args = ['boundary', ['phi', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #phi終端条件
    if p.set_cons['terminal_phi'] == False:
        pass
    else:
        args = ['boundary', ['phi', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v初期条件
    if p.set_cons['initial_v'] == False:
        pass
    else:
        args = ['boundary', ['v', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v終端条件
    if p.set_cons['terminal_v'] == False:
        pass
    else:
        args = ['boundary', ['v', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        
    return cons