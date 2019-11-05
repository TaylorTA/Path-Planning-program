__author__ = 'Jacky Baltes <jacky@cs.umanitoba.ca>'

import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
from pathplanning import PathPlanningProblem, Rectangle
from Astar import Astar

class CellDecomposition:

    def __init__(self, domain, minimumSize):
        self.domain = domain
        self.minimumSize = minimumSize
        self.root = [Rectangle(0.0, 0.0, domain.width, domain.height), 'unknown', []]

    def findAllFreeNodes(self, freeNode, node=None):
        if (node == None):
            node = self.root
        if (node[1] == 'free'):
            freeNode.append(node)
        for c in node[2]:
            self.findAllFreeNodes(freeNode, c)

    def Draw(self, ax, freeNode, node=None):
        if (node == None):
            node = self.root
        r = plt.Rectangle((node[0].x, node[0].y), node[0].width, node[0].height, fill=False, facecolor=None, alpha=0.5)
        if (node[1] == 'mixed'):
            color = '#5080ff'

            if (node[2] == []):
                #freeNode.append(node)
                r.set_fill(True)
                r.set_facecolor(color)

        elif (node[1] == 'free'):
            freeNode.append(node)
            color = '#ffff00'
            r.set_fill(True)
            r.set_facecolor(color)

        elif (node[1] == 'obstacle'):
            color = '#5050ff'
            r.set_fill(True)
            r.set_facecolor(color)
        else:
            print("Error: don't know how to draw cell of type", node[1])
        # print('Draw node', node)
        ax.add_patch(r)
        for c in node[2]:
            self.Draw(ax, freeNode, c)

    def CountCells(self, node=None):
        if (node is None):
            node = self.root
        sum = 0
        if (node[2] != []):
            sum = 0
            for c in node[2]:
                sum = sum + self.CountCells(c)
        else:
            sum = 1
        return sum


class QuadTreeDecomposition(CellDecomposition):
    def __init__(self, domain, minimumSize):
        super().__init__(domain, minimumSize)
        self.root = self.Decompose(self.root)

    def Decompose(self, node):
        cell = 'free'
        r = node[0]
        rx = r.x
        ry = r.y
        rwidth = r.width
        rheight = r.height

        for o in self.domain.obstacles:
            if (o.CalculateOverlap(r) >= rwidth * rheight):
                cell = 'obstacle'
                break
            elif (o.CalculateOverlap(r) > 0.0):
                cell = 'mixed'
                break
        if (cell == 'mixed'):
            if (rwidth / 2.0 > self.minimumSize) and (rheight / 2.0 > self.minimumSize):
                childt1 = [Rectangle(rx, ry, rwidth / 2.0, rheight / 2.0), 'unknown', []]
                qchild1 = self.Decompose(childt1)
                childt2 = [Rectangle(rx + rwidth / 2.0, ry, rwidth / 2.0, rheight / 2.0), 'unknown', []]
                qchild2 = self.Decompose(childt2)
                childt3 = [Rectangle(rx, ry + rheight / 2.0, rwidth / 2.0, rheight / 2.0), 'unknown', []]
                qchild3 = self.Decompose(childt3)
                childt4 = [Rectangle(rx + rwidth / 2.0, ry + rheight / 2.0, rwidth / 2.0, rheight / 2.0), 'unknown', []]
                qchild4 = self.Decompose(childt4)
                children = [qchild1, qchild2, qchild3, qchild4]
                node[2] = children
            else:
                cell = 'obstacle'
        node[1] = cell
        return node


class BinarySpacePartitioning(CellDecomposition):
    def __init__(self, domain, minimumSize):
        super().__init__(domain, minimumSize)
        self.root = self.Decompose(self.root)

    def Entropy(self, p):
        e = 0.0
        if ((p > 0) and (p < 1.0)):
            e = -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
        return e

    def CalcEntropy(self, rect):
        area = rect.width * rect.height
        a = 0.0
        for o in self.domain.obstacles:
            a = a + rect.CalculateOverlap(o)
        p = a / area
        return self.Entropy(p)

    def Decompose(self, node):
        cell = 'free'
        r = node[0]
        rx = r.x
        ry = r.y
        rwidth = r.width
        rheight = r.height
        area = rwidth * rheight

        for o in self.domain.obstacles:
            if (o.CalculateOverlap(r) >= rwidth * rheight):
                cell = 'obstacle'
                break
            elif (o.CalculateOverlap(r) > 0.0):
                cell = 'mixed'
                break

        if (cell == 'mixed'):
            entropy = self.CalcEntropy(r)
            igH = 0.0
            hSplitTop = None
            hSplitBottom = None
            vSplitLeft = None
            vSplitRight = None
            if (r.height / 2.0 > self.minimumSize):
                hSplitTop = Rectangle(rx, ry + rheight / 2.0, rwidth, rheight / 2.0)
                entHSplitTop = self.CalcEntropy(hSplitTop)
                hSplitBottom = Rectangle(rx, ry, rwidth, rheight / 2.0)
                entHSplitBottom = self.CalcEntropy(hSplitBottom)

                igH = entropy - (r.width * r.height / 2.0) / area * entHSplitTop \
                      - (r.width * r.height / 2.0) / area * entHSplitBottom
            igV = 0.0
            if (r.width / 2.0 > self.minimumSize):
                vSplitLeft = Rectangle(rx, ry, rwidth / 2.0, rheight)
                entVSplitLeft = self.CalcEntropy(vSplitLeft)
                vSplitRight = Rectangle(rx + rwidth / 2.0, ry, rwidth / 2.0, rheight)
                entVSplitRight = self.CalcEntropy(vSplitRight)
                igV = entropy - (r.width / 2.0 * r.height) / area * entVSplitLeft \
                      - (r.width / 2.0 * r.height) / area * entVSplitRight
            children = []
            if (igH > igV):
                if (igH > 0.0):
                    if (hSplitTop is not None) and (hSplitBottom is not None):
                        childTop = [hSplitTop, 'unknown', []]
                        childBottom = [hSplitBottom, 'unknown', []]
                        children = [childTop, childBottom]
            else:
                if (igV > 0.0):
                    if (vSplitLeft is not None) and (vSplitRight is not None):
                        childLeft = [vSplitLeft, 'unknown', []]
                        childRight = [vSplitRight, 'unknown', []]
                        children = [childLeft, childRight]
            for c in children:
                self.Decompose(c)
            node[2] = children
        node[1] = cell
        return node


def findAroud(rectangle, searchSpace, resultSet, ax):
    i=0
    rectangleX = rectangle.x
    rectangleY = rectangle.y
    rectangleHeight = rectangle.height
    rectangleWidth = rectangle.width
    Nodes=copy.deepcopy(searchSpace)
    while(i<len(searchSpace)):
        nodeX = searchSpace[i][0].x
        nodeY = searchSpace[i][0].y
        nodeHeight = searchSpace[i][0].height
        nodeWidth = searchSpace[i][0].width
       
        if(rectangleY + rectangleHeight==nodeY and not(nodeX+nodeWidth<=rectangleX) and not(nodeX>=rectangleX+rectangleWidth)):
            searchSpace[i][0].hValue = rectangle.hValue + 1
            resultSet.append(searchSpace[i][0])
            searchSpace.remove(searchSpace[i])
            i-=1
        
        elif(nodeX==rectangleX+rectangleWidth and not(nodeY>=rectangleY+rectangleHeight) and not(nodeY+nodeHeight<=rectangleY)):
            searchSpace[i][0].hValue = rectangle.hValue + 1
            resultSet.append(searchSpace[i][0])
            searchSpace.remove(searchSpace[i])
            i-=1
        
        elif(nodeY+nodeHeight==rectangleY and not(nodeX+nodeWidth<=rectangleX) and not (nodeX>=rectangleX+rectangleWidth)):
            searchSpace[i][0].hValue = rectangle.hValue + 1
            resultSet.append(searchSpace[i][0])
            searchSpace.remove(searchSpace[i])
            i-=1
        
        elif(nodeX+nodeWidth==rectangleX and not (nodeY>=rectangleY+rectangleHeight) and not (nodeY+nodeHeight<=rectangleY)):
            searchSpace[i][0].hValue = rectangle.hValue + 1
            resultSet.append(searchSpace[i][0])
            searchSpace.remove(searchSpace[i])
            i-=1
        
        i+=1

def findNode(x, y, searchSpace):
    for node in searchSpace:
        nodeHeight = node[0].height
        nodeWidth = node[0].width
        nodeX = node[0].x
        nodeY = node[0].y
        if (nodeX <= x <= nodeX + nodeWidth and nodeY <= y <= nodeY + nodeHeight):
            searchSpace.remove(node)
            return node

def ExploreDomain( domain, initial,blockNO,goals ):
    pos = np.array(initial)
    log=pos
    end = np.array(goals)[0]
    dd = 20/blockNO
    endlog=end
    while((abs(pos-end)>dd).any()):
        diff = (end - pos)
        delta = 0
        while (True):
            cut=random.uniform(delta/math.pi*180,-delta/math.pi*180)
            theta = np.arctan2(diff[1], diff[0])
            if(cut>0):
                cut+=90
            else:
                cut-=90
            if(delta!=0):
                theta+=cut
            newpos = pos + dd * np.array([dd * math.cos(theta), dd * math.sin(theta)])
            r = Rectangle(newpos[0], newpos[1], dd, dd)
            if ( newpos[0] >= 0.0 ) and ( newpos[0] < domain.width ) and ( newpos[1] >= 0.0 ) and ( newpos[1] < domain.height ):
                if ( not domain.CheckOverlap( r ) ):
                    pos = newpos
                    break
            if(delta<math.pi/2):
                delta+=0.1*math.pi/2
        delta = 0
        while (True):
            cut = random.uniform(delta / math.pi * 180, -delta / math.pi * 180)
            theta = np.arctan2(-diff[1], -diff[0])
            if (cut > 0):
                cut += 90
            else:
                cut -= 90
            if (delta != 0):
                theta += cut
            newend = end + dd * np.array([dd * math.cos(theta), dd * math.sin(theta)])
            r = Rectangle(newend[0], newend[1], dd, dd)
            if (newend[0] >= 0.0) and (newend[0] < domain.width) and (newend[1] >= 0.0) and (
                    newend[1] < domain.height):
                if (not domain.CheckOverlap(r)):
                    end = newend
                    break
            if (delta < math.pi / 2):
                delta += 0.1 * math.pi / 2

        endlog=np.vstack((end,endlog))
        log=np.vstack((log,pos))
    return np.vstack((log,endlog))

def main(argv=None):
    freeNode = []
    if (argv == None):
        argv = sys.argv[1:]

    width = 100.0
    height = 100.0
    blocknumber = 60
    maxSize = 10
    pp = PathPlanningProblem(width, height, blocknumber, maxSize, maxSize)
    # pp.obstacles = [ Obstacle(0.0, 0.0, pp.width, pp.height / 2.2, '#555555' ) ]
    initial, goals = pp.CreateProblemInstance()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)

    for o in pp.obstacles:
        ax.add_patch(copy.copy(o.patch))
    ip = plt.Rectangle((initial[0], initial[1]), 1.0, 1.0, facecolor='#ff0000')
    ax.add_patch(ip)

    for g in goals:
        g = plt.Rectangle((g[0], g[1]), 1.0, 1.0, facecolor='#00ff00')
        ax.add_patch(g)

    qtd = QuadTreeDecomposition(pp, 1.0)
    qtd.Draw(ax, freeNode)
    n = qtd.CountCells()
    ax.set_title('Quadtree Decomposition\n{0} cells'.format(n))

    path = ExploreDomain(pp, initial, blocknumber, goals)
    path_length = 0
    astar = Astar()
    for i in range(int(path.size/2) - 1):
        path_length += astar.getdistance(path[i][0],path[i][1],path[i+1][0],path[i+1][1])

    plt.plot(path[:, 0], path[:, 1], 'r-')
    print(path_length)

    goalX = goals[0][0]
    goalY = goals[0][1]
    goalNode = findNode(goalX, goalY, freeNode)

    initialX = initial[0]
    initialY = initial[1]

    resultSet = []
    resultSet.append(goalNode[0])
    while (len(resultSet) != 0):
        rectangle = resultSet.pop(0)
        findAroud(rectangle, freeNode, resultSet, ax)

    availableNodes = []
    qtd.findAllFreeNodes(availableNodes)

    #run A* algorithm
    initialNode = findNode(initialX, initialY, availableNodes)
    astar = Astar()
    pathlength = astar.Astarprocessing(initialNode,goalNode,availableNodes,ax,goalX,goalY,initialX,initialY)
    print(pathlength)
    plt.show()
    print("end")




if (__name__ == '__main__'):
    main()
