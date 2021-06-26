#encoding=utf8
import numpy as np
from gridword import GridwordEnv

env=GridwordEnv()

def value_iteration(env,theta=0.0001,discount_factor=1.0):
    def  one_step_lookahead(state,v):
        A=np.zeros(env.nA)
        for a in range(env.nA):
            # 定义四种操作
            for prob,next_state,reward,done in env.P[state][a]:
                # 通过bellman方程计算四种操作的结果
                A[a]+=prob*(reward+discount_factor*v[next_state])
        return A
    # 初始化16个状态
    v=np.zeros(env.nS)
    
    while True:
        delta=0
        for s in range(env.nS):
            A=one_step_lookahead(s,v)
            best_action_value=np.max(A)
            delta=max(delta,np.abs(best_action_value-v[s]))
            v[s]=best_action_value
        if delta<theta:
            break
    policy=np.zeros((env.nS,env.nA))
    for s in range(env.nS):
        A=one_step_lookahead(s,v)
        best_action_value=np.max(A)
        policy[s,best_action_value]=1.0
    return policy,v

policy,v=value_iteration(env)
    
    
