from MDP import *

## Mdp defined in lab 1
# actions is S->[[(Next_state,Probability,Reward)]]
actions = {"S":[[action("U", 0.5, 0), action("T", 0.5, 0)], [action("T", 1, 1)]],
           "U":[[action("S",0.6,2),action("U",0.4,2)]],
           "T":[[action("U",1,10)]]}
state_values = {"S":100,"U":100,"T":100}

lab_mdp = MDP(actions,state_values,gamma=GAMMA)
if __name__ == "__main__":
    value_iteration(lab_mdp)