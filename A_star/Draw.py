import sys
import time
import numpy as np

from A_star import Astar
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
#from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox


class Window(QtWidgets.QWidget):
    # 初始化
    def __init__(self):
        super(Window, self).__init__()

        # 设置主窗口大小,位置和标题
        self.move(0, 0)
        self.setFixedSize(1500, 1000)
        self.setWindowTitle("A*寻路")

        self.map = Astar.Map(20, 20)    # 调用A_star.py中的Map类,初始化为map对象
        self.row = self.map.row
        self.col = self.map.col

        self.Is_find = True  # 假设找到路径
        self.path = None
        self.source = None
        self.dest = None

        # 根据地图确定方格大小,浮点型
        nums = [1000 / self.row, 1000 / self.col]
        nums.sort()
        self.rect = nums[0]

        # 设置弹窗
        self.message_box = QMessageBox(self)

        # 默认初始化障碍图片标签
        self.obstacle_num = self.map.createBlock(int(self.row * self.col * 0.2))  # 初始设置障碍数目,比例约为20%
        self.obstacle = []
        self.obstacle_init()

        # 添加人工设置的障碍
        self.set_blocks_num = int(self.row * self.col * 0.1)  # 人为设置障碍数目，最多为地图大小的10%
        self.set_blocks = []
        for i in range(0, self.set_blocks_num):
            self.set_blocks.append(QLabel(self))
        self.cur_blocks_num = 0  # 统计已经设置的障碍数目

        # 设置计时器进行动态绘图
        self.Timer = QTimer(self)
        self.Timer.timeout.connect(self.draw_path)
        self.Timer.start(300)

        # 设置选项框:设置障碍，设置起点，设置终点
        self.set_obstacle_button = QRadioButton(self)
        self.set_obstacle_button.setText('设置障碍')
        self.set_obstacle_button.setGeometry(1200, 600, 100, 50)
        self.set_obstacle_button.toggled.connect(lambda: self.choose(self.set_obstacle_button))

        self.set_source_button = QRadioButton(self)
        self.set_source_button.setText('设置起点')
        self.set_source_button.setGeometry(1200, 650, 100, 50)
        self.set_source_button.toggled.connect(lambda: self.choose(self.set_source_button))

        self.set_dest_button = QRadioButton(self)
        self.set_dest_button.setText('设置终点')
        self.set_dest_button.setGeometry(1200, 700, 100, 50)
        self.set_dest_button.toggled.connect(lambda: self.choose(self.set_dest_button))

        # 设置搜索开始按钮
        self.search_begin_button = QPushButton(self)
        self.search_begin_button.setText('开始搜索')
        self.search_begin_button.setGeometry(1175, 800, 150, 50)
        self.search_begin_button.clicked.connect(self.solve)

        # 添加程序说明标签
        self.notice1 = QLabel(self)
        self.notice1.setGeometry(1200, 0, 100, 100)
        self.notice1.setText('程序说明：')
        self.notice1.setFont(QFont("Roman times", 15, QFont.Bold))

        self.notice2 = QLabel(self)
        self.notice2.setGeometry(1050, 100, 350, 100)
        self.notice2.setText('1.蓝色方块表示通道，黄色方块表示起点， 红色方块表示终点，绿色方块表示最优路径，砖块表示障碍。起点和终点设置之后不可取消，但可以更改。')
        self.notice2.setWordWrap(True)   #　根据内容自动换行

        self.notice3 = QLabel(self)
        self.notice3.setGeometry(1050, 200, 350, 100)
        self.notice3.setText('2.程序会自动随机生成20%的障碍，该部分障碍人为无法操作。此外，可人为生成10%的障碍，且一旦生成即不可撤销与更改。')
        self.notice3.setWordWrap(True)

        self.notice4 = QLabel(self)
        self.notice4.setGeometry(1050, 300, 350, 100)
        self.notice4.setText('3.按Esc键可退出程序。')
        self.notice4.setWordWrap(True)

    # 初始化障碍
    def obstacle_init(self):
        pixmap = QPixmap('obstacle.png')
        for i in range(self.map.row):  # 行数
            for j in range(self.map.col):  # 列数
                if self.map.map[i][j] == 1:
                    self.obstacle.append(QLabel(self))
                    self.obstacle[-1].move(j*self.rect, i*self.rect)
                    self.obstacle[-1].resize(self.rect, self.rect)
                    self.obstacle[-1].setPixmap(pixmap)
                    self.obstacle[-1].setScaledContents(True)

    # 画地图网格
    #def paint_map(self):
        #pen = QPainter()
        #pen.begin(self)
        #for i in range(self.row + 1):
            #pen.drawLine(0, i * self.rect, self.col * self.rect, i * self.rect)  # 画横线
        #for j in range(self.col + 1):
            #pen.drawLine(j * self.rect, 0, j * self.rect, self.row * self.rect)  # 画竖线
        #pen.end()

    # 绘制
    def paint_all(self):
        qp = QPainter()
        qp.begin(self)
        # 绘制路径,绿色
        #if self.Is_find == True:
            #for node in self.path[1:(len(self.path) - 1)]:
                #time.sleep(2)
                #qp.setBrush(QColor(125, 255, 125))
                #qp.drawRect(node.y * self.rect, node.x * self.rect, self.rect, self.rect)

        for i in range(self.map.row):  # 行数
            for j in range(self.map.col):  # 列数
                # 可行区域,蓝色
                if self.map.map[i][j] == 0:
                    qp.setBrush(QColor(110, 250, 255))
                    qp.drawRect(j * self.rect, i * self.rect, self.rect, self.rect)

                # 障碍,锗色
                #if self.map.map[i][j] == 1:
                    #qp.setBrush(QColor(96, 57, 18))
                    #qp.drawRect(j * self.rect, i * self.rect, self.rect, self.rect)

                # 起点和终点
                if self.map.map[i][j] == 3:   # 起点,黄色
                    qp.setBrush(QColor(255, 255, 125))
                    qp.drawRect(j * self.rect, i * self.rect, self.rect, self.rect)
                if self.map.map[i][j] == 4:   # 终点,红色
                    qp.setBrush(QColor(225, 0, 0))
                    qp.drawRect(j * self.rect, i * self.rect, self.rect, self.rect)

                # 绘制搜索路径,绿色
                if self.map.map[i][j] == 2:
                    #if self.map.map[self.source] == 2:
                        #self.map.map[self.source] = 3
                    #if self.map.map[self.dest] == 2:
                        #self.map.map[self.dest] = 4
                    qp.setBrush(QColor(125, 255, 125))
                    qp.drawRect(j * self.rect, i * self.rect, self.rect, self.rect)
        qp.end()
        self.update()

    # 选项框的槽函数
    def choose(self, btn):
        return btn.isChecked()
        #if btn.text() == '设置障碍':
            #if btn.isChecked() == True:
                #print(btn.text() + "is selected")

        #if btn.text() == '设置起点':
            #if btn.isChecked() == True:
                #print(btn.text() + "is selected")
                # 随机生成起点
                #if self.source != None:
                    #self.map.map[self.source] = 0
                #self.source = self.map.random_generatePos()
                #if self.source == self.dest:
                    #self.source = self.map.random_generatePos()
                #self.map.map[self.source] = 3

        #if btn.text() == '设置终点':
            #if btn.isChecked() == True:
                #print(btn.text() + "is selected")
                # 随机生成终点
                #if self.dest != None:
                    #self.map.map[self.dest] = 0
                #self.dest = self.map.random_generatePos()
                #if self.dest == self.source:
                    #self.dest = self.map.random_generatePos()
                #self.map.map[self.dest] = 4

    # 寻找路径
    def solve(self):
        # 存在开始结点和目标结点
        if self.source != None and self.dest != None:
            # 禁用按钮
            self.set_obstacle_button.setEnabled(False)
            self.set_source_button.setEnabled(False)
            self.set_dest_button.setEnabled(False)
            self.search_begin_button.setEnabled(False)

            Astar.AStarSearch(self.map, self.source, self.dest)
            self.Is_find = self.map.Ispath
            if self.Is_find == False:
                self.message_box.about(self, '提示', '无法搜索到可行路径!')
            if self.Is_find == True:
                self.path = self.map.path
                self.path.reverse()
                self.map.map[self.path[0].x][self.path[0].y] = 3
                self.map.map[self.path[-1].x][self.path[-1].y] = 4
                self.path.pop(0)  # 将起点去除
                self.path.pop(-1)  # 将终点去除

                for i in self.path:
                    self.map.map[i.x][i.y] = 0

                #for i in range(self.map.row):  # 行数
                    #for j in range(self.map.col):  # 列数
                        #if self.map.map[i][j] == 2:
                            #self.map.map[i][j] = 0
        else:
            self.message_box.about(self, '提示', '无开始结点或结束结点!')

    # 动态设置路径
    def draw_path(self):
        if self.path != None:
            # print('执行了调用')
            self.map.map[self.path[0].x][self.path[0].y] = 2
            # 如果path列表中只剩最后一个元素,结束计时并终止函数
            if len(self.path) == 1:
                self.Timer.stop()
                self.path.pop(0)
                self.message_box.about(self, '提示', '搜索到可行路径!')
                return
            self.path.pop(0)
            self.repaint()

    # 鼠标点击设置起点,终点和障碍
    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:  # 左键按下
            # 设置障碍
            if self.set_obstacle_button.isChecked() == True:
                # 最多手动设置10%的障碍
                if self.cur_blocks_num < self.set_blocks_num:
                    # 获取鼠标点击点相对主窗口的位置
                    #print('获取鼠标点击点的位置:')
                    x = event.windowPos().x()
                    y = event.windowPos().y()
                    #print((x, y))
                    if x < 1000 and y < 1000:
                        a = int(x // self.rect)  # 对应网格列
                        b = int(y // self.rect)  # 对应网格行
                        #print('转化为地图坐标:', (b, a))
                        if self.map.map[b][a] == 0 :
                            self.set_blocks[self.cur_blocks_num].setGeometry(a * self.rect, b * self.rect, self.rect,
                                                                             self.rect)
                            pixmap = QPixmap('obstacle.png')
                            self.set_blocks[self.cur_blocks_num].setPixmap(pixmap)
                            self.set_blocks[self.cur_blocks_num].setScaledContents(True)
                            self.map.map[b][a] = 1
                            self.cur_blocks_num = self.cur_blocks_num + 1

            # 设置起点
            if self.set_source_button.isChecked() == True:
                #print('获取鼠标点击点的位置:')
                x = event.windowPos().x()
                y = event.windowPos().y()
                #print('转化为地图坐标:', (x, y))
                if x < 1000 and y < 1000:
                    a = int(x // self.rect)
                    b = int(y // self.rect)
                    #print('起点坐标', (b, a))  # 对应网格坐标
                    if self.map.map[b][a] == 0:
                        if self.source != None:
                            self.map.map[self.source] = 0
                        self.source = (b, a)
                        self.map.map[self.source] = 3


            # 设置终点
            if self.set_dest_button.isChecked() == True:
                #print('获取鼠标点击点的位置:')
                x = event.windowPos().x()
                y = event.windowPos().y()
                #print('转化为地图坐标:', (x, y))
                if x < 1000 and y < 1000:
                    a = int(x // self.rect)
                    b = int(y // self.rect)
                    #print('终点坐标:', (b, a))  # 对应网格坐标
                    if self.map.map[b][a] == 0:
                        if self.dest != None:
                            self.map.map[self.dest] = 0
                        self.dest = (b, a)
                        self.map.map[self.dest] = 4


    # 按ESC键退出程序
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:  # 按住键是Esc时
            self.close()  # 关闭程序

    # 绘图函数
    def paintEvent(self, QpaintEvent):
        #self.paint_map()
        self.paint_all()
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Window()

    w.show()
    sys.exit(app.exec_())

