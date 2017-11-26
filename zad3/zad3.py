
# coding: utf-8

# In[37]:


import numpy as np
import numpy.random as np_rand
import matplotlib.pyplot as plt
import math

# find all empty positions

def find_empty(sud_m):
    empty_l = []
    for x in range(0, len(sud_m)):
        for y in range(0, len(sud_m[x])):
            if sud_m[x, y] == 0:
                empty_l.append((x,y))
                sud_m[x, y] = np_rand.randint(1,10)
    return empty_l

# energy

def get_close_neighbours(x, y, n):
    l = []
    for x2 in range(x-1, x+2):
        for y2 in range(y-1, y+2):
            if (-1 < x < n 
                and -1 < y < n 
                and (x != x2 or y != y2) 
                and (0 <= x2 < n) 
                and (0 <= y2 < n)):
                l.append((x2, y2))
    return l

def update_dict(d, key, nval):
    val = d.get(key)
    if val != None:
        val.append(nval)
        d.update({key: val})
    else:
        d.update({key: [nval]})

def energy(matrix):
    n = len(matrix)
    sum = 0    
    for x in range(0,n): # check rows
        row_dict = {}
        for y in range(0,n):
            update_dict(row_dict, matrix[x,y], y)
        rep_l = list(filter(lambda e: e > 1, list(map(lambda l: len(l), row_dict.values()))))
        for e in rep_l:
            sum += e
    for y in range(0,n): # check columns
        col_dict = {}
        for x in range(0,n):
            update_dict(col_dict, matrix[x,y], x)
        rep_l = list(filter(lambda e: e > 1, list(map(lambda l: len(l), col_dict.values()))))
        for e in rep_l:
            sum += e
    for y in range(1,7,3): # check blocks
        for x in range(1,7,3):
            neibs = get_close_neighbours(x, y, n)
            block_dict = {}
            update_dict(block_dict, matrix[x,y],5)
            for i in range(0,len(neibs)):
                k,m = neibs[i]
                update_dict(block_dict, matrix[k,m],i)
            rep_l = list(filter(lambda e: e > 1, list(map(lambda l: len(l), block_dict.values()))))
            for e in rep_l:
                sum += e
    return sum      

# schedule functions

def lin_schedule(t_0, k, n):
    a = np.linspace(0,t_0,n)
    return -a[k]+t_0

def exp_schedule(t_0, k, n):
    return t_0*(0.85**k)

def quad_schedule(t_0, k, n):
    return t_0/(1+2*k^2)

# swap functions

def arbit_swap(matrix, empty_l, neib_num):
    new_m = matrix.copy()
    n = len(empty_l)
    m_l = []
    if neib_num > n:
        neib_num = n
    for i in range(0,neib_num):
        x, y = empty_l[np_rand.randint(0,len(empty_l))]
        old = new_m[x,y]    
        new = np_rand.randint(1,10)
        while new == old:
            new = np_rand.randint(1,10)
        new_m[x,y] = new
        m_l.append(new_m)
    return m_l
                                  
# probability

def p(e, e_new, t):
    if t < 1:
        t = 1
    return math.exp(-(e_new - e)/t)

# simulated annealing

def sim_an(s, k_max, t_0, t_1, p, schedule, swap, energy, neighbours_num, empty_l):
    for i in range(0, 5):
        for k in range(0, k_max):
            t = schedule(t_0, k, k_max)
            a = swap(s, empty_l, neighbours_num)
            for s_new in a:
                ne = energy(s_new)
                if ne == 0:
                    return s_new
                if p(energy(s), ne, t) >= np_rand.random():
                    s = s_new
            if math.isclose(t_1, t, rel_tol = 0.01):
                break
    return s

def sim_an_extreme(s, k_max, t_0, t_1, p, schedule, swap, energy, neighbours_num, empty_l):
    for i in range(0, 5):
        for k in range(0, k_max):
            t = schedule(t_0, k, k_max)
            a = swap(s, empty_l, neighbours_num)
            for s_new in a:
                ne = energy(s_new)
                if ne == 0:
                    return s_new
                if p(energy(s), ne, t) >= np_rand.random():
                    s = s_new
    while energy(s) != 0:
        a = swap(s, empty_l, neighbours_num)
        for s_new in a:
            ne = energy(s_new)
            if p(energy(s), ne, t) >= np_rand.random():
                s = s_new
    return s


# In[38]:


with open('sudoku.txt') as f:
    read_data = f.readlines()
stream = []
for line in read_data:
    l = line.split()
    stream += l
con_stream = list(map(lambda x: 0 if x == 'x' else int(x), stream))
sud_m = np.array(con_stream).reshape(9,9)

schedule = quad_schedule
swap = arbit_swap

empty_l = find_empty(sud_m)
print(sud_m)
print(energy(sud_m))


# In[39]:


s = sim_an(sud_m, 5000, 5000, 1, p, schedule, swap, energy, 40, empty_l)
print(s)
print(energy(s))

