import numpy as np
import random
from random import randint

# 定义搜索入口
class SearchEntry():
    def __init__(self, x, y, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        self.f_cost = f_cost
        self.g_cost = g_cost  # 计算从起点到当前结点的代价(G值)
        self.pre_entry = pre_entry  # 当前结点的父结点位置

    # 获取结点位置
    def getPos(self):
        return (self.x, self.y)

# 建立搜索地图
class Map():

    # 建立地图
    def __init__(self, row, col):
        self.row = row  # 行数
        self.col = col  # 列数
        self.map = np.zeros([self.row, self.col])  # 创建地图
        self.path = []  # 储存路径
        self.Ispath = True  # 初始假设可以找到路径
        self.Block_Num = 0

    # 随机设置block_num个障碍
    def createBlock(self, block_num):
        self.Block_Num = block_num
        for i in range(block_num):
            x, y = (randint(0, self.row - 1), randint(0, self.col - 1))  # 随机生成一个障碍(x,y)
            while True:
                if self.map[x][y] == 0:  # 如果当前不是障碍或者起始点
                    self.map[x][y] = 1
                    break
                else:
                    x, y = (randint(0, self.row - 1), randint(0, self.col - 1))
        return self.Block_Num

    # 生成起点和终点
    def generatePos(self, X, Y):
        x = X
        y = Y
        while self.map[x][y] == 1:  # 当随机生成的起点是障碍时，重新生成
            print('当前位置为障碍，请重新输入')
            x, y = eval(input())
        return (x, y)  # 返回x行y列

    # 随机生成起点和终点
    def random_generatePos(self):
        x, y = (randint(0, self.row - 1), randint(0, self.col - 1))
        while self.map[x][y] == 1:  # 当随机生成的起点是障碍时，重新生成
            x, y = (randint(0, self.row - 1), randint(0, self.col - 1))
        return (x, y)  # 返回x行y列

    # 打印路径
    def showpath(self):
        if self.Ispath == True:
            self.path.reverse()
            s = "搜索路径: "
            for i in self.path[:-1]:
                s += '(' + str(i.x) + ',' + str(i.y) + ') --> '
            s += '(' + str(self.path[-1].x) + ',' + str(self.path[-1].y) + ')'
            print(s)


    # 显示
    def showMap(self):
        print("+" * (5 * self.col + 2))

        for row in self.map:  # 获取地图中的每行数据
            # print(row)
            s = '+'
            for entry in row:  # 获取每行中的每个数据
                s += ' ' + str(entry) + ' '
            s += '+'
            print(s)

        print("+" * (5 * self.col + 2))

# 搜索算法
def AStarSearch(map, source, dest):

    # location为当前结点,offset是移动方向
    # 获取当前结点的某个相邻位置
    def getNewPosition(map, location, offset):
        x, y = (location.x + offset[0], location.y + offset[1])
        if x < 0 or x >= map.row or y < 0 or y >= map.col or map.map[x][y] == 1:
            return None
        return (x, y)  # 返回x行y列

    # pos是某个相邻结点的位置,offsets是移动方向集,poslist为所有可用相邻结点集的位置
    # 获取当前结点的所有相邻位置
    def getPositions(map, location):
        # 四方向移动
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        # 八方向移动
        # offsets = [(-1,0), (0, -1), (1, 0), (0, 1), (-1,-1), (1, -1), (-1, 1), (1, 1)]
        poslist = []  # 储存当前结点的相邻结点位置
        for offset in offsets:
            pos = getNewPosition(map, location, offset)
            if pos is not None:
                poslist.append(pos)
        return poslist

    # 移动代价:斜移动为14,否则为10
    def getMoveCost(location, pos):
        if location.x != pos[0] and location.y != pos[1]:
            return 14
        else:
            return 10

    # 获取启发函数h的距离
    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0])*10 + abs(dest.y - pos[1])*10  # 使用曼哈顿距离

    # 检查相邻结点是否在表中(用“键”判断）
    def isInList(list, pos):
        if pos in list:
            return list[pos]
        return None

    # 将相邻结点添加到open list中并计算代价
    def addAdjacentPositions(map, location, dest, openlist, closedlist):
        poslist = getPositions(map, location)  # 获取所有可用相邻结点
        for pos in poslist:
            if isInList(closedlist, pos) is None:  # pos不在close list中
                findEntry = isInList(openlist, pos)  # 判断pos是否在open list中
                h_cost = calHeuristic(pos, dest)  # 计算代价H
                g_cost = location.g_cost + getMoveCost(location, pos)  # 计算代价G
                if findEntry is None:
                    # pos不在open list中,添加对应pos键的值SearchEntry
                    openlist[pos] = SearchEntry(pos[0], pos[1], g_cost, g_cost + h_cost, location)
                elif findEntry.g_cost > g_cost:
                    # pos在open list中,且新的代价比原来代价低,则更新父结点和代价F
                    findEntry.g_cost = g_cost
                    findEntry.f_cost = g_cost + h_cost
                    findEntry.pre_entry = location

    # 在open list中寻找具有最小代价的结点，如果open list为空则查找失败
    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None:
                fast = entry
            elif fast.f_cost > entry.f_cost:
                fast = entry
        return fast

    openlist = {}
    closedlist = {}
    location = SearchEntry(source[0], source[1], 0.0)  # 当前搜索起点，初始化为搜索源点
    dest = SearchEntry(dest[0], dest[1], 0.0)  # 搜索终点
    openlist[source] = location  # 将搜索起点添加到open list中
    while True:
        location = getFastPosition(openlist)  # 获取open list中的最小代价点
        if location is None:
            print("can't find valid path")
            map.Ispath = False
            break

        if location.x == dest.x and location.y == dest.y:
            break

        closedlist[location.getPos()] = location  # 将该结点加入close list
        openlist.pop(location.getPos())  # 移出open list
        addAdjacentPositions(map, location, dest, openlist, closedlist)  # 继续下一个结点寻找

    # 标记路径
    while location is not None:
        map.map[location.x][location.y] = 2
        map.path.append(location)
        location = location.pre_entry

if __name__ == '__main__':
    ROW = 5
    COL = 5
    BLOCK_NUM = 5

    map = Map(ROW, COL)
    map.createBlock(BLOCK_NUM)
    map.showMap()

    #print("输入搜索起点:")
    #m, n = eval(input())
    #source = map.generatePos(m, n)
    #print("输入搜索终点:")
    #c, d = eval(input())
    #dest = map.generatePos(c, d)

    source = map.random_generatePos()  # 生成起点
    dest = map.random_generatePos()  # 生成终点

    print("搜索起点:", source)
    print("搜索终点:", dest)
    AStarSearch(map, source, dest)
    map.showMap()
    map.showpath()
    #print(len(map.path))


