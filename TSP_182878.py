"""
Author: Vivek Chaudhari
"""

import random
from Individual import *
import sys
import math
import copy
import os


random.seed(12345)

class TSP_GA:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations,no_iter):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.no_iter = no_iter

        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            if sys.argv[5] == '2':
                genes = self.huiristicPopulation()
                individual.setGene(genes)
            individual.computeFitness()
            self.population.append(individual)
        
        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        self.storeResults("\n==================================")
        text = "\nBest initial sol: "+str(self.best.getFitness())        
        print (text)
        self.storeResults(text)
        
    def square(self,num):
       return int(num) * int(num)

    def minus(self,num1, num2):
       return int(num1)-int(num2)

    def calculate(self,x1, y1, x2, y2):
        return int(round(math.sqrt(self.square(self.minus(x1,x2)) + self.square(self.minus(y1,y2)))))

    def find_city(self,city, filter_data, all_data):
        x1 = all_data.get(city)[0]
        y1 = all_data.get(city)[1]
        temp_min = 0
        min_city_index = 0
        for city_index in filter_data :
            x2 = all_data.get(city_index)[0]
            y2 = all_data.get(city_index)[1]
            calc_min = self.calculate(x1,y1,x2,y2)
            if(temp_min == 0):
                temp_min = calc_min
            if(calc_min <= temp_min):
                temp_min = calc_min
                min_city_index = city_index
        return min_city_index
    pass

    def solve(self,filter_data, all_data):
        output_data = []
        firstCity = random.sample(all_data.keys(),1)
        output_data.append(firstCity[0])
        filter_data.pop(firstCity[0])
        all_city_checked = False
        city1_index = firstCity[0]
        while not all_city_checked:
            nearest_city = self.find_city(city1_index, filter_data, all_data)
            city1_index = nearest_city
            output_data.append(nearest_city)
            filter_data.pop(nearest_city)
            if(len(filter_data) == 0):
                all_city_checked = True
        return output_data

    def huiristicPopulation(self):
        filter_data = copy.deepcopy(self.data)
        all_data = copy.deepcopy(filter_data)
        return self.solve(filter_data, all_data)
        pass
    
    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() > self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())
            text = "iteration: "+str(self.iteration)+" best: "+str(self.best.getFitness())+'\n'
            self.storeResults(text)
            #TODO
            text2 = str(self.no_iter)+','+str(self.iteration)+','+str(self.best.getFitness())+'\n'
            self.generateGraph(text2)

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        index = 0
        fitnessArr = {}
        for ind_i in self.population:
            fitnessArr[index] = ind_i.getFitness()
            index+=1
            
        maxFitness = max(fitnessArr.values())

        for k, fitness in fitnessArr.items():
            transFitness = (maxFitness - fitness)+1
            fitnessArr[k] = transFitness

        totalTransFitness = sum(fitnessArr.values())
        for k, transFitness in fitnessArr.items():
            selectionProbability = round(transFitness / totalTransFitness, 8)
            fitnessArr[k] = selectionProbability
            
        p = 1  / self.popSize
        startPoint = random.uniform(0,p)
        matingPool = []
        for i in range(0,self.popSize):
            rulerPoint = startPoint*(i+1)
            popIndex = 0
            selctProb = 0
            for index, probability in fitnessArr.items():
                selctProb = selctProb + probability
                if rulerPoint < selctProb:
                    popIndex = index   
                    break
            matingPool.append(self.population[popIndex])
            
        self.matingPool = matingPool
        pass

    def uniformCrossover(self, parent1, parent2):
        """
        Your Uniform Crossover Implementation
        """
        indexes = random.sample(range(0,self.genSize), round(self.genSize/2))

        child1 = {}
        child2 = {}
        for index in indexes:
            child1[index] = parent1.genes[index]
            child2[index] = parent2.genes[index]

        parent2_remain_items = []
        parent1_remain_items = []
        for index in range(0,self.genSize):
            exist_in_child1 = parent2.genes[index] in child1.values()
            if exist_in_child1 is False:
                parent2_remain_items.append(parent2.genes[index])
                
            exist_in_child2 = parent1.genes[index] in child2.values()
            if exist_in_child2 is False:
                parent1_remain_items.append(parent1.genes[index])
           
        i=0 
        j=0
        for index in range(0, self.genSize):
            if (index in child1) is False:
                child1[index] = parent2_remain_items[i]
                i+=1
           
            if (index in child2) is False:
                child2[index] = parent1_remain_items[j]
                j+=1
        children1 = dict(sorted(child1.items()))

        parent1.setGene(children1.values())
        parent1.computeFitness()
        
        return parent1
        pass
    
    def getKey(self,dataDict,value):
     return [key for key in dataDict.keys() if (dataDict[key] == value)]
 
    def getNextNumber(self, index, child1, child2, parent1):
        missNum = parent1[index]
        missKey = self.getKey(child1, missNum)
        
        found = True
        while found :
            ch2Num = child2.get(missKey[0])
            if ch2Num in child1.values():
                newKey = self.getKey(child1, ch2Num)
                missKey = newKey
            else :    
                found = False
        return ch2Num   
    
    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        parent1 = indA.genes
        parent2 = indB.genes
        indexes = random.sample(range(0,self.genSize), round(self.genSize/2))
        child1 = {}
        child2 = {}
        for index in indexes:
            child1[index] = parent2[index]
            child2[index] = parent1[index]
            
        parent2_remain_index = []
        parent1_remain_index = []
        for index in range(0,self.genSize):
            exist_in_child1 = parent1[index] in child1.values()
            if exist_in_child1 is False:
                if child1.get(index) is None:
                    child1[index] = parent1[index]
            elif child1.get(index) is None :
                parent1_remain_index.append(index)
                
            exist_in_child2 = parent2[index] in child2.values()
            if exist_in_child2 is False:
                if child2.get(index) is None:
                    child2[index] = parent2[index]
            elif child2.get(index) is None :
                parent2_remain_index.append(index)

        for index in parent1_remain_index:
            child1[index] = self.getNextNumber(index, child1, child2, parent1)  
        '''
        # not required because expecting only one child from crossover
        for index in parent2_remain_index:
            child2[index] = getNextNumber(index, child2, child1, parent2)
        '''    
        indA.setGene(dict(sorted(child1.items())).values())    
        indA.computeFitness()
        
        return indA
    
        pass
    def reciprocalExchangeMutation(self, chromosome):
        """
        Reciprocal Exchange Mutation implementation
        """
        if random.random() > self.mutationRate:
            return chromosome
        index = random.sample(range(0,self.genSize), k=2)
        swap1 = chromosome.genes[index[0]]
        swap2 = chromosome.genes[index[1]]
    
        chromosome.genes[index[0]] = swap2
        chromosome.genes[index[1]] = swap1
        
        return chromosome
        pass

    def inversionMutation(self, child):
        """
        Inversion Mutation implementation
        """
        
        if random.random() > self.mutationRate:
            return child
        index = random.sample(range(0,self.genSize), k=2)
        index.sort()
        startIndex = index[0]
        endIndex = index[1]
        rev_operator = 0
        chromosome = child.genes
        for index in range(startIndex, endIndex+1):
            revIndex = endIndex-rev_operator
            temp1 = chromosome[index]
            temp2 = chromosome[revIndex]
            if index > revIndex:
                break
            rev_operator+=1
            chromosome[index] = temp2
            chromosome[revIndex] = temp1
            
        child.setGene(chromosome)
        
        return child
        pass

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)
        return ind

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        if sys.argv[2] == '1' :
            self.matingPool = []
            for ind_i in self.population:
                self.matingPool.append( ind_i.copy() )
                
        elif sys.argv[2] == '2':
            self.stochasticUniversalSampling()
    
    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
        for i in range(0, len(self.population)):

            parent1, parent2 = self.randomSelection()
            
            if sys.argv[3] == '1' :    
                child = self.pmxCrossover(parent1, parent2)
            elif sys.argv[3] == '2' :
                child = self.uniformCrossover(parent1, parent2)
            
            if sys.argv[4] == '1' :          
                mutated = self.inversionMutation(child)
            elif sys.argv[4] == '2' :
                mutated = self.reciprocalExchangeMutation(child)

            mutated.computeFitness()
            self.updateBest(mutated)
            self.population[i] = mutated

    def storeResults(self, text):
        
        inputFile = str(sys.argv[1]).split('.')
        file = open(str(inputFile[0])+'/result_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt','a')
        file.write(text)
        file.close()
        
        #TODO
    def generateGraph(self, text):
        inputFile = str(sys.argv[1]).split('.')
        
        file = open(str(inputFile[0])+'/graph_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt','a')
        file.write(text)
        file.close()
        
    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        text = "Total iterations: "+str(self.iteration)+" Best: "+str(self.best.getFitness())
        print(text)
        self.storeResults(text)

if len(sys.argv) < 4:
    print ("Error - Incorrect input")
    print ("Expecting python TSP_GA.py [instance] ")
    sys.exit(0)


inputFile = str(sys.argv[1]).split('.')
try:
    # Create target Directory
    os.mkdir(inputFile[0])    
    print("Directory " , inputFile[0] ,  " Created ") 
except FileExistsError:
    pass

try:
    os.remove(str(inputFile[0])+'/result_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt')
    #TODO
    os.remove(str(inputFile[0])+'/graph_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt')
except OSError:
    pass
    
for no_iter in range(0,5):
    ga = TSP_GA(sys.argv[1], 100 , 0.1, 500,no_iter)
    ga.search()