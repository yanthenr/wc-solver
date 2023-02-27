
import cv2 as cv
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import os
import seaborn as sns
from queue import PriorityQueue
import time

class Search():
    def __init__(self, field, mode = "dfs") -> None:
        self.itemcount = 0
        self.StartSearch(field, mode)

    def CheckNeighbours(self, field, visited, c, x, y):
        '''
        Checks recursively if direct neighbours of the coordinate belong to the class c. 
        Returns list of coordinates (nb) which belong to the class c.
        If list only contains 1 item, it does not have any neighbours of the same class.

        field: field of values, int8 array 
        visited: array of visited cells, boolean array
        c: class, int8
        x: x cor, int
        y: y cor, int
        '''
        maxheight = field.shape[0] - 1
        maxwidth = field.shape[1] - 1

        # BASE CASE:
        # if current node does not belong to specified class return empty list
        if field[x,y] != c:
            return []

        # RECURSION:
        # visit neighbours if they aren't already visited and add current coordinates
        else:
            visited[x,y] = True
            nb = [(x,y)]
            #print("x, y:",x,y) 
            # north
            NORTH = (x-1, y)
            if NORTH[0] >= 0 and visited[NORTH[0],NORTH[1]] == False:
                nb += self.CheckNeighbours(field, visited, c, NORTH[0], NORTH[1])
            
            # east
            EAST = (x, y+1)
            if EAST[1] <= maxwidth and visited[EAST[0],EAST[1]] == False:
                nb += self.CheckNeighbours(field, visited, c, EAST[0], EAST[1])
            
            # south
            SOUTH = (x+1, y)
            if SOUTH[0] <= maxheight and visited[SOUTH[0],SOUTH[1]] == False:
                nb += self.CheckNeighbours(field, visited, c, SOUTH[0], SOUTH[1])
            
            # west
            WEST = (x, y-1)
            if WEST[1] >= 0 and visited[WEST[0],WEST[1]] == False:
                nb += self.CheckNeighbours(field, visited, c, WEST[0], WEST[1])
        return nb

    def UpdateField(self, parent, coordinates):
        '''
        Updates field.
        The cells in the coordinates are "removed": value set to 0.
        Remaining cells 1. "fall down" and 2. columns "shift" from left to right when there is an empty column as a right neighbour.
        Returns numpy array.  
        '''
        field = np.array(parent, copy=True) 
        maxheight = field.shape[0] -1
        maxwidth = field.shape[1] -1

        # Set value of coordinates to 0 (empty):
        for (y,x) in coordinates:
            field[y,x] = 0

        # 1. Fall down:
        xcors = set(list(zip(*coordinates))[1])
        for x in xcors:
            if sum(field[:,x]) != 0:
                valueList = []
                for y in range(maxheight,-1,-1):
                    if field[y,x] != 0:
                        valueList.append(field[y,x])
                amtzeros = field.shape[0] - len(valueList)
                valueList = (amtzeros*[0])+valueList[::-1] # reverse valuelist.
                field[:,x] = valueList

        # 2. Shift empty columns to right:
        empty = [0]*(field.shape[0])
        notEmpty = []
        
        for x in range(maxwidth,-1,-1):
            if field[field.shape[0]-1,x] != 0:
                notEmpty.append(x)

        for x, xne in zip(range(maxwidth,maxwidth-len(notEmpty),-1),notEmpty):
            field[:,x] = field[:,xne]
        
        for x in range(maxwidth-len(notEmpty),-1,-1):
            field[:,x] = empty
            
        return field

    def DFS(self, field, sum, moves, states):
        '''
        Solution to game is recursively searched using DFS. 
        Returns bool whether a solution is found, and list of list of coordinates in order that should be clicked to reach solution. 
        '''
        # BASE CASE:
        # Game is solved
        if sum == 0:
            # Return list of list of coordinates you have to click in order to solve the game
            return True, moves, states
        
        # RECURSION:
        # Iterate through field and generate new transformed field, removing one island of cells
        height = field.shape[0]
        width = field.shape[1]
        visited = np.full((height, width), False) # initialise 'empty' array

        for y in range(0,width):
            for x in range(0,height):
                if field[x,y] > 0:
                    nb = self.CheckNeighbours(field,visited,field[x,y],x,y)
                    if len(nb) > 1: # Cell has island of >= 2 cells: "poppable"
                        newfield = self.UpdateField(field,nb)
                        #newsum = sum - (field[x,y]*len(nb))
                        #newmoves = moves + [nb]
                        #newstates = states + [newfield]
                        foundsolution, allmoves, allstates = self.DFS(newfield, sum - (field[x,y]*len(nb)), moves + [nb], states + [newfield]) 
                        if foundsolution:
                            return True, allmoves, allstates
        
        # When there are no moves left on the field but solution has not been found: 
        return False, [], []
    
    def GenerateChildrenAstar(self, field, moves, states, q, visitedStates):
        '''
        Returns all possible children of current node (field) and puts them in q when field configuration has not been in q yet
        '''
        height = field.shape[0]
        width = field.shape[1]
        visited = np.full((height, width), False) # initialise 'empty' array

        for y in range(0,width):
            for x in range(0,height):
                if field[x,y] > 0:
                    nb = self.CheckNeighbours(field,visited,field[x,y],x,y)
                    if len(nb) > 1: # Cell has island of >= 2 cells

                        # Generate new child field with "popped bubbles":
                        newfield = self.UpdateField(field,nb)

                        # Check if new field (tuple form) has not been visited:
                        if tuple(map(tuple, newfield)) not in visitedStates:
                            
                            # Add to priority queue
                            newmoves = moves + [nb]
                            newstates = states + [newfield]
                            count = int(np.count_nonzero(newfield > 0))
                            q.put((count, self.itemcount, newfield, newmoves, newstates))
                            self.itemcount += 1

                            # Add field configuration to visited states:
                            visitedStates[tuple(map(tuple, newfield))] = True

        return q, visitedStates


    def Astar(self, field, moves, states, q):# iterative
        '''
        Solution to game is iteratively searched using A* Search. 
        Returns bool whether a solution is found, and list of list of coordinates in order that should be clicked to reach solution. 
        '''

        # Enter starting state in priority queue:
        moves = []
        states = [field]
        q.put((np.count_nonzero(field > 0), self.itemcount, field, moves, states))

        # Keep track of visited states - field has to be converted to tuple in order to be hashable
        visitedStates = {tuple(map(tuple, field)): True} 

        while not q.empty():
            print(q.qsize())
            # Get first element of priority queue:
            (currentcount, _, currentfield, currentmoves, currentstates) = q.get()

            # If field is empty: game is solved:
            if currentcount == 0:
                return True, currentmoves, currentstates

            # Generate children of current node and add to queue:
            q, visitedStates = self.GenerateChildrenAstar(currentfield, currentmoves, currentstates, q, visitedStates)
        
        return False, [], []

    def PrintMoves(self, field, moves, states):
        grid = np.zeros((field.shape[0],field.shape[1]))
        
        for i, (move, state) in enumerate(zip(moves,states)):
            grid = np.zeros((field.shape[0],field.shape[1]))

            for (x,y) in move:
                grid[x,y] = 1

            if states == []: # edit !=
                print("State "+str(i+1)+":")
                print(state)

            print("Move "+str(i+1)+":")
            print(grid)

    def StartSearch(self, field, mode):
        print("Starting search...")
        if mode == "dfs":
            foundsolution, moves, states = self.DFS(field, np.sum(field), [], [field])
        if mode == "astar":
            q = PriorityQueue()
            foundsolution, moves, states = self.Astar(field, [], [field], q)

        if foundsolution:
            self.PrintMoves(field, moves, states)
        else:
            print("Solution not found.")



class GetField():
    def __init__(self):
        self.templates = ["1.png", "2.png", "3.png", "4.png"] # 1: shoe, 2: shirt, 3: bag, 4: jeans
        self.thresholds = {"1": 0.99, "2": 0.99, "3": 0.99, "4": 0.99}
        self.fieldHeight = 12
        self.fieldWidth = 11 
        self.imageDimCalculated = False
        self.imageHeight = 0
        self.imageWidth = 0
        self.values = self.ConvertToArray()

    def TemplateMatching(self, imgpath, templatepath):
        locations = []
        img_rgb = cv.imread(imgpath)

        if not self.imageDimCalculated:
            self.imageWidth = img_rgb.shape[1]
            self.imageHeight = img_rgb.shape[0]

        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread(templatepath,0)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        threshold = self.thresholds[templatepath[0]]
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            locations.append((templatepath[0], pt[0], pt[1]))
        cv.imwrite("res"+templatepath[0]+".png",img_rgb)
        return locations

    def GetMostRecentImage(self, dirpath = "/Users/Y/Desktop/", valid_extensions = ["png"]):
        '''
        Gets the newest image (screenshot) from specified directory.
        '''
        # get filepaths of all files and dirs in the given dir
        valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
        # filter out directories, no-extension, and wrong extension files
        valid_files = [f for f in valid_files if '.' in f and \
            f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]
        if not valid_files:
            raise ValueError("No valid images in %s" % dirpath)
        img = max(valid_files, key=os.path.getmtime)
        print("Getting img from "+str(img))
        return img

    def ConvertToArray(self):
        field = np.zeros((self.fieldHeight, self.fieldWidth))
        for template in self.templates: 
            locations = self.TemplateMatching(self.GetMostRecentImage(), template)
            for (c, x, y) in locations:
                xfield = int(x // (self.imageWidth / self.fieldWidth))
                yfield = int(y // (self.imageHeight / self.fieldHeight))
                field[yfield,xfield] = c
        print("Field:")
        print(field)
        return field

slay = np.array([
    [1,1,1,1,1,1,1,1,1],
    [2,1,1,2,2,2,2,2,2],
    [1,3,3,3,3,3,3,3,3],
    [1,2,2,2,2,2,3,3,3],
    [4,4,4,4,4,1,1,1,1],
    [1,2,2,2,4,4,4,4,4]
])


def Solve(mode = "dfs"):
    startTime = time.perf_counter()
    field = GetField()
    Search(field.values, mode)
    endTime = time.perf_counter()
    print(f"Time elapsed: {endTime-startTime:0.4f} seconds")

# Run:
Solve("astar")