if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import common.gridworld as gw

from collections import defaultdict

def eval_onestep(pi: defaultdict, V : defaultdict, env : gw, gamma=0.9) :
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
            
        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)

            new_V += action_prob * (r + gamma * V[next_state])
        
        V[state] = new_V
    
    return V

def policy_eval(pi: defaultdict, V : defaultdict, env : gw, gamma : float, threshold=0.001) :
    while True :
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold :
            break
    return V

env = gw.GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # Policy
V = defaultdict(lambda: 0)  # value function

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)