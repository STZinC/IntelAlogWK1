from __future__ import print_function
import random
import array

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import fuzzyLogic
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.animation as animation  
# custom-defined evaluation function
def evaluate(x,rf,vmin,vmax,fmin,fmax):
    return fuzzyLogic.F(x,rf,vmin,vmax,fmin,fmax), 

def checkBound(x,low, up):
    if x < low:
        x = low
    if x > up:
        x = up
    return x
# output Log or not
outLog = False
# Problem dimension
NDIM = 13
# Problem solution domain (bound of each x)
xlow = [0]*13
xup = [1]*9 + [100]*3 +[1]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", evaluate)

def main():
    # Differential evolution parameters
    # CR cross rate(0-1, usually 0.3) F Learning rate(0-2,usually 0.5) M/U population /NGEN evolution generation
    CR = 0.3
    F = 0.2
    MU = 200
    NGEN = 1000   
    
    pop = toolbox.population(n=MU);
    for ind in pop:
        ind[9] = ind[9]*100
        ind[10] = ind[10]*100
        ind[11] = ind[11]*100
    #Best Record
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "max"
    
    # Evaluate the individuals

    fit = [fuzzyLogic.fitness(x) for x in pop]
    vio = [fuzzyLogic.violation(x) for x in pop]
    fes = [fuzzyLogic.isFeasible(x) for x in pop]
    vmin = min(vio)
    vmax = max(vio)
    fmin = min(fit)
    fmax = max(fit)
    rf = float(sum(fes))/MU    

    for ind in pop:
        ind.fitness.values = evaluate(ind, rf,vmin,vmax,fmin,fmax);

    #History Best individual
    bestInd = pop[0];
    bestFit = 0;
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    
    for g in range(1, NGEN):
        y =  toolbox.clone(pop)
        for k, agent in enumerate(pop):
            a,b,c = toolbox.select(pop)
            
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    tmp = a[i] + F*(b[i]-c[i])
                    if tmp > xlow[i] and tmp < xup[i]:
                        y[k][i] = tmp
                    
                    elif tmp < xlow[i]:
                       y[k][i] = xlow[i]
                    else:
                        y[k][i] = xup[i]
                    

        fit = [fuzzyLogic.fitness(x) for x in y]
        vio = [fuzzyLogic.violation(x) for x in y]
        fes = [fuzzyLogic.isFeasible(x) for x in y]
        vmin = min(vio)
        vmax = max(vio)
        fmin = min(fit)
        fmax = max(fit)
        rf = float(sum(fes))/MU 
        #calc fitness 
        for ind in y:
            ind.fitness.values = evaluate(ind, rf,vmin,vmax,fmin,fmax);
            # print("F:"+str(y.fitness.values))
        for k, agent in enumerate(pop):
            if y[k].fitness > agent.fitness:
                pop[k] = y[k]
        hof.update(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        if(outLog == True):
            print(logbook.stream, fuzzyLogic.fitness(hof[0]),hof[0].fitness.values)
            print("vmin:"+str(vmin)+", vmax:"+str(vmax)+", fmin:"+str(fmin)+", fmax:"+str(fmax)+", rf:"+str(rf))
            print("nf:"+str(sum(fes)))
        
        temp = toolbox.clone(pop)
        temp.sort(reverse = 1)
        
        for one in temp:
            if(fuzzyLogic.isFeasible(one)==1):
                if bestFit > fuzzyLogic.fitness(one):
                    bestInd =  one;
                    bestFit = fuzzyLogic.fitness(one)
                print("Best feasible individual is ", fuzzyLogic.fitness(one))
                break
        print("--------------------------------------------------------------")
        #pop[0] = bestInd
        #use to plot
        yield(g, bestFit)

    print("Best individual is ",bestInd, bestFit)

def run(data):  
    # update the data  
    t,y = data  
    xdata.append(t)  
    ydata.append(y)  
    xmin, xmax = ax.get_xlim()  
  
    if t >= xmax:  
        ax.set_xlim(xmin, 2*xmax)  
        ax.figure.canvas.draw()  
    line.set_data(xdata, ydata)  
  
    return line, 
    
if __name__ == "__main__":
    fig, ax = plt.subplots()  
    line, = ax.plot([], [], lw=2)  
    ax.set_ylim(-20,10)  
    ax.set_xlim(0, 5)  
    ax.grid()  
    xdata, ydata = [], []
    ani = animation.FuncAnimation(fig, run, main, blit=True, interval=.01,  
        repeat=False)  
    plt.show()
