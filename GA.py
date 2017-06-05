# -*- coding: UTF-8 -*-

"""
问题：

从某物流中心用多台配送车辆向多个客户送货,每个客户的位置和货物需求量一定,每台配送车辆的载重量一定,其一次配送的最大行驶距离一定,要求合理安排车辆配送路线,使目标函数得到优化,并满足以下条件:

(1) 每条配送路径上各客户的需求量之和不超过配送车辆的载重量;

(2) 每条配送路径的长度不超过配送车辆一次配送的最大行驶距离;

(3) 每个客户的需求必须满足,且只能由一台配送车辆送货。

以配送总里程最短为目标函数
"""

"""
一个实例：

某物流中心有2 台配送车辆,其载重量均为8t ,车辆每次配送的最大行驶距离为50km ,配送中心(其编号为0) 与8 个客户之间及8 个客户相互之间的距离dij 、8 个客户的货物需求量qj (i 、j = 1 ,2 , ⋯,8) 均见表1 。要求合理安排车辆配送路线,使配送总里程最短。
采用以下参数:群体规模取20 ,进化代数取25 ,交叉概率取0.9 ,变异概率取0.09 ,变异时基因换位次数取5 , 对不可行路径的惩罚权重取100km ,实施爬山操作时爬山次数取20 。对实例随机求解10 次。

"""

import random

#遗传算法
class GeneticAlgorithm:

    #-----------初始数据定义---------------------
    #定义一个9 * 9的二维数组表示配送中心(编号为0)与8个客户之间，以及8个客户相互之间的距离d[i][j]
    d = [[0, 4, 6, 7.5, 9, 20, 10, 16, 8],              #配送中心（编号为0）到8个客户送货点的距离
         [4, 0, 6.5, 4, 10, 5, 7.5, 11, 10],            #第1个客户到配送中心和其他8个客户送货点的距离
         [6, 6.5, 0, 7.5, 10, 10, 7.5, 7.5, 7.5],       #第2个客户到配送中心和其他8个客户送货点的距离
         [7.5, 4, 7.5, 0, 10, 5, 9, 9, 15],
         [9, 10, 10, 10, 0, 10, 7.5, 7.5, 10 ],
         [20, 5, 10, 5, 10, 0, 7, 9, 7.5],
         [10, 7.5, 7.5, 9, 7.5, 7, 0, 7, 10],
         [16, 11, 7.5, 9, 7.5, 9, 7, 0, 10],
         [8, 10, 7.5, 15, 10, 7.5, 10, 10, 0]];

    # 8个客户分布需要的货物的需求量，第0位为配送中心自己
    q = [0, 1, 2, 1, 2, 1, 4, 2, 2];

    #定义一些遗传算法需要的参数
    JCL = 0.9   #遗传时的交叉率
    BYL = 0.09  #遗传时的变异率
    JYHW = 5    #变异时的基因换位次数
    PSCS = 20   #爬山算法时的迭代次数

    def __init__(self, rows, times, mans, cars, tons, distance, PW):
        self.rows = rows                            #排列个数
        self.times = times                          #迭代次数
        self.mans = mans                            #客户数量
        self.cars = cars                            #车辆总数
        self.tons = tons                            #车辆载重
        self.distance = distance                    #车辆一次行驶的最大距离
        self.PW = PW                                #当生成一个不可行路线时的惩罚因子

    #-------------遗传函数开始执行---------------------
    def run(self):

        print "开始迭代"

        #路线数组
        lines = [[0 for i in range(self.mans)] for i in range(self.rows)]

        #适应度
        fit = [0 for i in range(self.rows)]

        # print "初始输入获取rows个随机排列，并且计算适应度"
        #初始输入获取rows个随机排列，并且计算适应度
        j = 0
        for i in range(0, self.rows):
            j = 0
            while j < self.mans:
                num = int(random.uniform(0, self.mans)) + 1
                if self.isHas(lines[i], num) == False:
                    lines[i][j] = num
                    j += 1

            #计算每个线路的适应度
            # print "计算每个线路的适应度 i = %d" % (i)
            fit[i] = self.calFitness(lines[i], False)

        #迭代次数
        t = 0

        while t < self.times:

            #适应度
            newLines = [[0 for i in range(self.mans)] for i in range(self.rows)]
            nextFit = [0 for i in range(self.rows)]
            randomFit = [0 for i in range(self.rows)]
            totalFit = 0
            tmpFit = 0

            # print "计算总的适应度"
            #计算总的适应度
            for i in range(self.rows):
                totalFit += fit[i]

            # print "通过适应度占总适应度的比例生成随机适应度"
            #通过适应度占总适应度的比例生成随机适应度
            for i in range(self.rows):
                randomFit[i] = tmpFit + fit[i] / totalFit
                tmpFit += randomFit[i]

            # print "上一代中的最优直接遗传到下一代"
            #上一代中的最优直接遗传到下一代
            m = fit[0]
            ml = 0

            for i in range(self.rows):
                if m < fit[i]:
                    m = fit[i]
                    ml = i

            for i in range(self.mans):
                newLines[0][i] = lines[ml][i]

            nextFit[0] = fit[ml]

            # print "对最优解使用爬山算法促使其自我进化"
            #对最优解使用爬山算法促使其自我进化
            self.clMountain(newLines[0])

            # print "开始遗传"
            #开始遗传
            nl = 1
            while nl < self.rows:
                #根据概率选取排列
                r = int(self.randomSelect(randomFit))

                #判断是否需要交叉，不能越界
                if random.random() < self.JCL and nl + 1 < self.rows:
                    fline = [0 for x in range(self.mans)]
                    nline = [0 for x in range(self.mans)]

                    #获取交叉排列
                    rn = int(self.randomSelect(randomFit))

                    f = int(random.uniform(0, self.mans))
                    l = int(random.uniform(0, self.mans))

                    min = 0
                    max = 0
                    fpo = 0
                    npo = 0

                    if f < l:
                        min = f
                        max = l
                    else:
                        min = l
                        max = f

                    # print "将截取的段加入新生成的基因"
                    #将截取的段加入新生成的基因
                    """
                    除排在第一位的最优个体外,另N - 1 个个体要按交叉概率Pc 进行配对交叉重组。
                    采用类OX法实施交叉操作,现举例说明其操作方法: 
                    ①随机在父代个体中选择一个交配区域,如两父代个体及交配区域选定为:A = 47| 8563| 921 ,B = 83| 4691|257 ;
                    ②将B 的交配区域加到A 的前面,A 的交配区域加到B 的前面,得:A’= 4691| 478563921 ,B’=8563| 834691257 ;
                    ③在A’、B’中自交配区域后依次删除与交配区相同的自然数,得到最终的两个体为:A”= 469178532 ,B”= 856349127 
                    
                    """
                    while min < max:
                        fline[fpo] = lines[rn][min]
                        nline[npo] = lines[r][min]

                        min += 1
                        fpo += 1
                        npo += 1

                    for i in range(self.mans):
                        if self.isHas(fline, lines[r][i]) == False:
                            fline[fpo] = lines[r][i]
                            fpo += 1

                        if self.isHas(nline, lines[rn][i]) == False:
                            nline[npo] = lines[rn][i]
                            npo += 1

                    #基因变异
                    self.change(fline)
                    self.change(nline)

                    # print "交叉并且变异后的结果加入下一代"
                    #交叉并且变异后的结果加入下一代
                    for i in range(self.mans):
                        newLines[nl][i] = fline[i]
                        newLines[nl + 1][i] = nline[i]

                    nextFit[nl] = self.calFitness(fline, False)
                    nextFit[nl + 1] = self.calFitness(nline, False)

                    nl += 2
                else:
                    # print "不需要交叉的，直接变异，然后遗传到下一代"
                    #不需要交叉的，直接变异，然后遗传到下一代

                    line = [0 for i in range(self.mans)]
                    i = 0
                    while i < self.mans:
                        line[i] = lines[r][i]
                        i += 1

                    #基因变异
                    self.change(line)

                    #加入下一代
                    i = 0
                    while i < self.mans:
                        newLines[nl][i] = line[i]
                        i += 1

                    nextFit[nl] = self.calFitness(line, False)
                    nl += 1

            # print "新的一代覆盖上一代 当前是第 %d 代" %(t)
            #新的一代覆盖上一代
            for i in range(self.rows):
                for h in range(self.mans):
                    lines[i][h] = newLines[i][h]

                fit[i] = nextFit[i]

            t += 1

        #上代中最优的为适应函数最小的
        m = fit[0]
        ml = 0

        for i in range(self.rows):
            if m < fit[i]:
                m = fit[i]
                ml = i

        print "迭代完成"
        #输出结果:
        self.calFitness(lines[ml], True)

        print "最优权值为: %f" %(m)
        print "最优结果为:"

        for i in range(self.mans):
            print "%d," %(lines[ml][i]),

        print "    "
        print "    "
        print "    "


    #-----------------遗传函数执行完成--------------------

    #-----------------各种辅助计算函数--------------------
    #线路中是否包含当前的客户
    def isHas(self, line, num):
        for i in range(0, self.mans):
            if line[i] == num:
                return True
        return False


    #计算适应度,适应度计算的规则为每条配送路径要满足题设条件，并且目标函数即 车辆行驶的总里程越小，适应度越高
    def calFitness(self, line, isShow):

        carTon = 0  #当前车辆的载重
        carDis = 0  #当前车辆行驶的总距离
        newTon = 0
        newDis = 0
        totalDis = 0

        # ll = []
        r = 0       #表示当前需要车辆数
        # l = 0
        fore = 0    #表示正在运送的客户编号
        M = 0       #表示当前的路径规划所需要的总车辆和总共拥有的车辆之间的差，如果大于0，表示是一个失败的规划，乘以一个很大的惩罚因子用来降低适应度

        #遍历每个客户点
        for i in range(0, self.mans):
            #行驶的距离
            newDis = carDis + self.d[fore][line[i]]

            #当前车辆的载重
            newTon = carTon + self.q[line[i]]

            #如果已经超过最大行驶距离或者超过车辆的最大载重，切换到下一辆车
            if newDis + self.d[line[i]][0] > self.distance or newTon > self.tons:
                #下一辆车
                totalDis += carDis + self.d[fore][0]  #后面加这个d[fore][0]表示需要从当前客户处返程的距离
                r += 1
                fore = 0
                i -= 1  #表示当前这个点的配送还没有完成
                carTon = 0
                carDis = 0
            else:
                carDis = newDis
                carTon = newTon
                fore = line[i]

        #加上最后一辆车的距离和返程的距离
        totalDis += carDis + self.d[fore][0]

        if isShow:
            print "总行驶里程为: %.1fkm" %(totalDis)
        else:
            # print "中间过程尝试规划的总行驶里程为: %.1fkm" %(totalDis)
            pass

        #判断路径是否可用，所使用的车辆数量不能大于总车辆数量
        if r - self.cars + 1 > 0:
            M = r - self.cars + 1

        #目标函数，表示一个路径规划行驶的总距离的倒数越小越好
        result = 1 / (totalDis + M * self.PW)

        return result


    #爬山算法
    def clMountain(self, line):
        oldFit = self.calFitness(line, False)

        i = 0
        while i < self.PSCS:
            f = random.uniform(0, self.mans)
            n = random.uniform(0, self.mans)

            self.doChange(line, f, n)

            newFit = self.calFitness(line, False)

            if newFit < oldFit:
                self.doChange(line, f, n)
            i += 1

    #基因变异
    #变异的意思是当满足变异率的条件下，随机的两个因子发生多次交换，交换次数为变异迭代次数规定的次数
    def change(self, line):
        if random.random() < self.BYL:
            i = 0
            while i < self.JYHW:
                f = random.uniform(0, self.mans)
                n = random.uniform(0, self.mans)

                self.doChange(line, f, n)
                i += 1


    #将线路中的两个因子执行交换
    def doChange(self, line, f, n):

        tmp = line[int(f)]
        line[int(f)] = line[int(n)]
        line[int(n)] = tmp

    #根据概率随机选择的序列
    def randomSelect(self, ranFit):

        ran = random.random()

        for i in range(self.rows):
            if ran < ranFit[i]:
                return i


#-------------入口函数，开始执行-----------------------------

"""
输入参数的的意义依次为

        self.rows = rows                            #排列个数
        self.times = times                          #迭代次数
        self.mans = mans                            #客户数量
        self.cars = cars                            #车辆总数
        self.tons = tons                            #车辆载重
        self.distance = distance                    #车辆一次行驶的最大距离
        self.PW = PW                                #当生成一个不可行路线时的惩罚因子

"""
ga = GeneticAlgorithm(rows=20, times=25, mans=8, cars=2, tons=8, distance=50, PW=100)

for i in range(20):
    print "第 %d 次：" %(i + 1)
    ga.run()
