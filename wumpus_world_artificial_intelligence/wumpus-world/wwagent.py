"""
Modified from wwagent.py written by Greg Scott

Modified to only do random motions so that this can be the base
for building various kinds of agent that work with the wwsim.py
wumpus world simulation -----  dml Fordham 2019

# FACING KEY:
#    0 = up
#    1 = right
#    2 = down
#    3 = left

# Actions
# 'move' 'grab' 'shoot' 'left' right'

"""

from random import randint

# constants
directions={'up':0,'right':1,'down':2,'left':3}
x_axis=[0,1,0,-1]
y_axis=[-1,0,1,0]
dir=['up','right','down','left']
move_on=10 # threshhold to keep track of how many times the agent visited an area before they decide to move on


# useful functions


# check if (i,j) is inside the map
def inside_map(i,j,mapsize):
    if 0<=i and i< mapsize and 0<=j and j<mapsize :
        return 1
    else:
        return 0


# visit coefficient of a square
def visitcoefficient(x,y,max,visited_map):
        coef=0
        counter=0
        for i in range (4):
            a=x+x_axis[i]
            b=y+y_axis[i]
            if inside_map(a,b,max) :
                coef+=visited_map[a][b]
                counter+=1
        return(coef/counter)
    
# This is the class that represents an agent

class WWAgent:

    def __init__(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (0, 3) # top is (0,0)
        self.directions=['up','right','down','left']
        self.facing = 'right'
        self.arrow = 1
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)]
        self.dangermap = [[ 4 for i in range(self.max) ] for j in range(self.max)] # represents agent knowledge 
        print("New agent created")
        self.visitedmap=[[ 0 for i in range(self.max) ] for j in range(self.max)] # tracks visits to a square
        self.next_direction = 'up' # next direction should be kept in the player class
        self.last_action ='left' # useful


    # Add the latest percepts to list of percepts received so far
    # This function is called by the wumpus simulation and will
    # update the sensory data. The sensor data is placed into a
    # map structured KB for later use


    def update(self, percept):
        self.percepts=percept
        #[stench, breeze, glitter, bump, scream]
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            self.map[ self.position[0]][self.position[1]]=self.percepts
        # puts the percept at the spot in the map where sensed

    # Since there is no percept for location, the agent has to predict
    # what location it is in based on the direction it was facing
    # when it moved


    def calculateNextPosition(self,action):
        if self.facing=='up':
            self.position = (self.position[0],max(0,self.position[1]-1))
        elif self.facing =='down':
            self.position = (self.position[0],min(self.max-1,self.position[1]+1))
        elif self.facing =='right':
            self.position = (min(self.max-1,self.position[0]+1),self.position[1])
        elif self.facing =='left':
            self.position = (max(0,self.position[0]-1),self.position[1])

        return self.position

    # function to estimate danger to make the safest move
    def danger_estimate(self):

    # [stench, breeze, glitter, bump, scream]
    # safe:0
    # pit:1
    # wumpus:2
    # pit+wumpus:3
    # add 4 if unsure
    # maybe pit:5
    # maybe wumpus:6
    # maybe pit+wumpus:7  next_direction=self.da_way(self.dangermap); #next direction
        
        
        if self.last_action == 'shoot' and self.percepts[4]=='none':
            xtemp=self.position[0]+x_axis[directions[self.facing]] # coordinates of opposite squares
            ytemp=self.position[1]+y_axis[directions[self.facing]]
            if inside_map(xtemp,ytemp,self.max):
                self.dangermap[xtemp][ytemp]=0

        if self.percepts[4]=='scream':
            for i in range(self.max):
                for j in range(self.max):
                    if self.dangermap[i][j]in [6,7]:
                        self.dangermap[i][j]-=6
                    elif self.dangermap[i][j]in [2,3]:
                        self.dangermap[i][j]-=2

        if self.percepts[1]=='none' and self.percepts[0]=='none':
            for i in range(4):
                if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) :
                    self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]=0
                    print('safe')

        if self.visitedmap[self.position[0]][self.position[1]]==0 : # if never visited the square update danger map
            if self.percepts[1]== 'breeze' :
                for i in range (4):
                    if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) and self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=0 :
                        if self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]] in [1,3,5,7] :
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]=1
                        elif self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==6:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]=0
                        else:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]+=1

            elif self.percepts[1]== 'none':
                for i in range (4):
                    if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) and self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=0 :
                        if self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]] in [1,3,5,7] :
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]-=1
                        elif self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==5:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==0

            if self.percepts[0]== 'stench':
                for i in range (4):
                    if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) and self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=0 :
                        if self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]] in [6,7] :
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]-=4
                        elif self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=2:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]+=2
                        elif self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==5:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]=0

            elif self.percepts[0]== 'none':
                for i in range (4):
                    if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) and self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=0 :
                        if self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]] in [5,7]:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]-=2
                        elif self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==6:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==0
            elif 1 :
                for i in range (4):
                    if inside_map(self.position[0]+x_axis[i],self.position[1]+y_axis[i],self.max) and self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]!=0 :
                        if self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]==4:
                            self.dangermap[self.position[0]+x_axis[i]][self.position[1]+y_axis[i]]=0



        # finally, the current position of the agent is safe - so it will be set to 0
        # this value cannot be changed later
        # this will help the agent find their way back safely
        self.dangermap[self.position[0]][self.position[1]]=0

        self.visitedmap[self.position[0]][self.position[1]]+=1
        return self.dangermap



    def calculateNextDirection(self,action):
        # update danger map
        if self.facing=='up':
            if action=='left':
                self.facing = 'left'
            else:
                self.facing = 'right'

        elif self.facing=='down':
            if action=='left':
                self.facing = 'right'
            else:
                self.facing = 'left'
        elif self.facing=='right':
            if action=='left':
                self.facing = 'up'
            else:
                self.facing = 'down'
        elif self.facing=='left':
            if action=='left':
                self.facing = 'down'
            else:
                self.facing = 'up'


    # most important function used to choose the next square
    def da_way(self,danger_map):
        if visitcoefficient(self.position[0],self.position[1],self.max,self.visitedmap) > 5:
            T=[0,4,5]
        else:
            T=[0,4]
        t=5
        min=1000
        nb_vi=1000
        k=0
        for i in range (4):
            x=self.position[0]+x_axis[i]
            y=self.position[1]+y_axis[i]

            if inside_map(x,y,self.max) and self.dangermap[x][y]==0 and self.visitedmap[x][y]< move_on and min>self.visitedmap[x][y]:
                min=self.visitedmap[x][y]
                t=i
            elif inside_map(x,y,self.max) and visitcoefficient(x,y,self.max,self.visitedmap)<nb_vi and self.dangermap[x][y] in T:
                k=i
                nb_vi=visitcoefficient(x,y,self.max,self.visitedmap)

        if t!=5:
            return(dir[t])
        else:
            return(dir[k])

    
    # this is the function that will pick the next action of
    # the agent. This is the main function that needs to be
    # modified when you design your new intelligent agent
    # right now it is just a random choice agent


    def action(self):
        
        # update danger map
        self.danger_estimate()
        
        # test for controlled exit at end of successful gui episode
        if self.stopTheAgent:
            print("Agent has won this episode.")
            return 'exit' # will cause the episide to end

        # reflect action -- get the gold!
        if 'glitter' in self.percepts:
            print("Agent will grab the gold!")
            self.stopTheAgent=True
            return 'grab'

        # check if there is a wumpus nearby
        x=self.position[0]+x_axis[directions[self.facing]] # coordinates of the opposite square
        y=self.position[1]+y_axis[directions[self.facing]]
        print(self.facing)
        print(x,y)
        if inside_map(x,y,self.max) and self.dangermap[x][y] in [2,3] and self.last_action!='shoot':
            action = 'shoot'
            print('SHOOOOOOT')
            for i in range(self.max):
                for j in range(self.max):
                	if self.dangermap[i][j] in [6,7] :
                		self.dangermap[i][j]-=6

        elif self.facing == self.next_direction:
                self.danger_estimate()
                action = 'move'
                # predict the effect of this
                self.calculateNextPosition(action)
                self.next_direction = self.da_way(self.dangermap) # choose a next direction after a move
        else:
            # adjust_direction
            if (directions[self.facing]-directions[self.next_direction]+3)%4 == 0:
                action = 'left'
            else:
                action ='right'
            self.calculateNextDirection(action)

        print ("agent:",action, "-->",self.position[1],
                self.position[0], self.facing)
        print('dangermap:',self.dangermap)
        # print(self.visitedmap)
        # print([[ visitcoefficient(i,j,self.max,self.visitedmap) for i in range(self.max) ] for j in range(self.max)])

        self.last_action=action
        return action

