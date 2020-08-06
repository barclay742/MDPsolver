from MDP import *
import matplotlib.pyplot as plt


#Full specifications in Barto Sutton Chapter 4 under Gamblers Problem

Prob_head = 0.5

## Gambler Mdp
states = [i for i in range(1,100)]
actions = {}
actions[0]=[[]]

#Action functions
for state in states:
    action_list = []
    for i in range(0,min(state,101-state)):
        if state+i == 100:
            action_list.append([action(state-i,1-Prob_head,0),action(state+i,Prob_head,1)])
        action_list.append([action(state - i, 1 - Prob_head, 0), action(state + i, Prob_head, 0)])
    actions[state]=action_list
actions[100]=[[]]

state_values = {}
state_values[0]=0
state_values[100]=1

for state in states:
    state_values[state]=1

gambler_Mdp = MDP(actions,state_values,gamma=0.99,theta=0.000000000001)
new_values = value_iteration(gambler_Mdp)

#Plotting
x=[]
y=[]
for k,v in new_values.items():
    x.append(k)
    y.append(v)
plt.scatter(x,y)
plt.show()
