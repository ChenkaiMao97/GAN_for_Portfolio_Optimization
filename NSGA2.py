# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import numpy as np
import matplotlib.pyplot as plt

N = 0
num_stocks = 22
b = 40
f = 20

fake_batches = np.load("portfolio_data/fake_batches.npy")[0]
real_batch = np.load("portfolio_data/real_batches.npy")[0]
scales = np.load("portfolio_data/scales.npy")[0]

print("shapes:", fake_batches.shape, real_batch.shape, scales.shape)
buy_in_prices = (real_batch[:,b-1] + 1)/2 * (scales[:,1]-scales[:,0]) + scales[:,0]
end_prices = (fake_batches[:,:,-1] + 1 )/2 * (scales[:,1]-scales[:,0]) + scales[:,0]
real_end_prices = (real_batch[:,-1] + 1 )/2 * (scales[:,1]-scales[:,0]) + scales[:,0]
print("buy_in_prices: ", buy_in_prices)
#First function to optimize
def function1(x):
    cost = np.sum(buy_in_prices*x)
    mean_end_price = np.mean(end_prices, axis=0)

    if abs(cost) < 1e-3:
        print("cost: ", cost, buy_in_prices, x)

    value = np.sum(x*mean_end_price)
    f1 = (value-cost)/cost
    # print("f1:", f1)
    return f1

#Second function to optimize
def function2(x):
    x = x/np.mean(x)
    cost = np.sum(buy_in_prices*x)

    x = x.reshape(1,x.shape[0])
    value = np.var(np.sum(x*end_prices, axis=1))
    
    if abs(cost) < 1e-3:
        print("cost: ", cost, buy_in_prices, x)

    f2 = -value
    # print("f2: ", f2)
    return f2

def real_return(x):
    cost = np.sum(buy_in_prices*x)
    value = np.sum(real_end_prices*x)

    return (value-cost)/cost

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    r=random.random()
    return mutation((a+b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <0.5:
        solution = min_x+(max_x-min_x)*np.random.rand(num_stocks)
    return solution

#Main program starts here
pop_size = 100
max_gen = 100

#Initialization
min_x=np.zeros(num_stocks)
max_x=np.ones(num_stocks)
solution=[min_x+(max_x-min_x)*np.random.rand(num_stocks) for i in range(0,pop_size)]
gen_no=0
while(gen_no<max_gen):
    print("geneartion: ", gen_no)
    function1_values = [function1(solution[i]/np.sum(solution[i])) for i in range(0,pop_size)]
    function2_values = [function2(solution[i]/np.sum(solution[i])) for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    #print("The best front for Generation number ",gen_no, " is")
    #for valuez in non_dominated_sorted_solution[0]:
    #    print(np.round(solution[valuez],3),end=" ")
    #print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        ab = random.sample(list(range(pop_size)),2)
        a1 = ab[0]
        b1 = ab[1]
        solution2.append(crossover(solution[a1],solution[b1]))
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1


#Lets plot the final front now
plt.figure()
f1_value = [i for i in function1_values]
f2_value = [-j for j in function2_values]
plt.xlabel('risk', fontsize=15)
plt.ylabel('return', fontsize=15)
plt.scatter(f2_value, f1_value)
plt.savefig("optimized.png")

plt.figure()
risk = [-function2(i) for i in solution]
true_return = [real_return(i) for i in solution]
plt.scatter(risk, true_return)
plt.xlabel("risk")
plt.ylabel("real return")
plt.savefig("risk_return.png")




