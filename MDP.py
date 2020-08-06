from collections import namedtuple

GAMMA = 0.9
THETA = 0.00001

action = namedtuple("action",field_names=["Next_state" ,"Prob","Reward"])

class MDP:
    def __init__(self,actions,state_values,gamma=0.99,theta=THETA):
        self.actions = actions
        self.state_values = state_values
        self.gamma = gamma
        self.theta = theta

#Value iteration
def value_iteration(mdp:MDP):
    state_values = mdp.state_values
    actions = mdp.actions
    gamma = mdp.gamma
    theta = mdp.theta
    t = 0
    exit = False
    while not exit:
        temp_v = state_values.copy()
        for state in state_values:
            #looping through actions
            values_for_actions = []
            for action in actions[state]:
                #looping through probabilities of each action
                bellman_backup = 0
                for a in action:
                    bellman_backup += a.Prob*(a.Reward+temp_v[a.Next_state]*gamma)
                values_for_actions.append(bellman_backup)
            state_values[state]=max(values_for_actions)

        bellman_error = max([abs(state_values[k] - temp_v[k]) for k in state_values])
        if bellman_error < theta:
            exit = True
        t+=1
    per_line =5
    for idx, (k,v) in enumerate( state_values.items() ):
        print(f"{k} : {v} ,",end =" " if (idx+1)%per_line else "\n" )
    print(f"at {t} iterations")

    return state_values

