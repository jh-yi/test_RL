

#------------------Initialization MDP: S, A, P, R, gamma, pi, values-------------------------

states = [i for i in range(16)]
actions = ['up','down','left','right']
ds_actions = {'up':-4,'down':4,'left':-1,'right':1 }
#P = 1
gamma = 1
#pi: random policy, which means each actinon with probabilities 0.25

values = [0 for _ in range(16)]

def rewardOf(s):
    return 0 if s in [0,15] else -1

#-----------------get Successors states und update Value according to current state-------------------
def getSuccessors(s):
    successors = []
    if s in [0,15]:return successors
    for a in actions:
        next_state = nextState(s,a)
        successors.append(next_state)
    return successors
def nextState(s,a):
    next_state = s
    if (s<4 and a =='up') or (s%4==0 and a =='left') or (s>11 and a =='down') or ((s+1)%4==0 and a =='right'):
        pass
    else:
        next_state = s + ds_actions[a]
    return next_state
def updateValue(s):
    new_value = 0
    #caculate newValue with Bellman Expectation Equation
    for next_state in getSuccessors(s):
        new_value += 0.25 * (rewardOf(s)+ gamma * values[next_state])
    return new_value

#------------------------perform one Iteration at a time-------------------------------------------
def performOneIteration():
    global values
    new_Values = [0 for _ in range(16)]
    for s in states:
        new_Values[s] = updateValue(s)
    values = new_Values
    printValue(values)

def printValue(v):
    for i in range(16):
        print('{0:6.2f}'.format(v[i]),end = ' ')
        if (i+1)%4==0:
            print("")
    print("")
#-------------------------main()--------------------------------------------------
def main():
    max_Iteration = 160
    cur_Iteration = 0
    while cur_Iteration <= max_Iteration:
        print('Iterateion No.{0}'.format(cur_Iteration))
        performOneIteration()
        cur_Iteration +=1

if __name__ == '__main__':
    main()
