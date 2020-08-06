from collections import namedtuple

GAMMA = 0.9
THETA = 0.00001
states =["S","U","T"]
action = namedtuple("action",field_names=["Next_state" ,"Prob","R"])
# action is S->[[(Next_state,Probability,Reward)]]
actions = {"S":[[action("U",0.5,0),action("T",0.5,0)],[action("T",1,1)]],
           "U":[[action("S",0.6,2),action("U",0.4,2)]],
           "T":[[action("U",1,10)]]}
state_values = {"S":100,"U":100,"T":100}

#Value iteration

t = 0
exit = False
while not exit:
    temp_v = state_values.copy()
    for idx,state in enumerate(states):
        #looping through actions
        values_for_actions = []
        for action in actions[state]:
            #looping through probabilities
            bellman_backup = 0
            for a in action:
                bellman_backup += a.Prob*(a.R+temp_v[a.Next_state]*GAMMA)
            values_for_actions.append(bellman_backup)
        state_values[state]=max(values_for_actions)

    bellman_error = max([abs(state_values[k] - temp_v[k]) for k in state_values])
    if bellman_error < THETA:
        exit = True
    t+=1

print(state_values,f"at {t} iterations")


