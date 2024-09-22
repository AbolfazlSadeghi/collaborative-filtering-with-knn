# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:39:30 2015

@author: Yasser
"""

import numpy as np
import scipy.sparse as sps

#%%


def cos_sim(x, y, mode='sparse', normalize=False):
    if mode.lower()=='sparse':
        if not normalize:
            s = x.multiply(y).sum()
        else:
            s = x.multiply(y).sum()/np.sqrt(x.multiply(x).sum()*y.multiply(y).sum())
    else:
        x = np.array(x)
        y = np.array(y)
        if not normalize:
            s = np.sum(x*y)
        else:
            s = np.sum(x*y)/np.sqrt(np.sum(x**2)*np.sum(y**2))
    return s

#%%


def sort_vec(V):
    V_s = []
    for vi in V:
        vi_s = sorted(enumerate(vi),reverse=True, key=lambda x: x[1]) #[(idx1, val1),(idx2,val2), ...]
        V_s.append(vi_s)
    return V_s

#%%

def construct_prefix_vec(Vs, P):
    Vprime = []
    init = [0 for i in Vs[0]]
    for idx, v in enumerate(Vs):
        vprime = init[:]
        for e_ix, e in enumerate(v):
            if e_ix<P[idx][0]:
                vprime[e[0]] = e[1] # reconstruct vprime (prefix of v) using their original indices/dimensions
        Vprime.append(vprime)
    Vprime = sps.csr_matrix(Vprime) # conver into sparse
    return Vprime

#%%


def update_Knn_Graph(Q, Knn, sim_jk, j, k):
    qj = Q[j,:].toarray()[0]
    qj_ = qj[qj>0]
    qj_ = np.sort(qj_)
    if len(qj_)>=Knn:
        if sim_jk>qj_[0]:
            loc_ix = np.argwhere(qj==qj_[0])
            Q[j,loc_ix[0]] = 0 # remove that vector from the Knn set
            Q[j,k] = sim_jk
        else:
            pass
    else:
        Q[j,k] = sim_jk
    return Q
    
#%%
    
def Brute_force_search(Knn, L, P, V, Vs=[], mode='fast', normalize=False):
    if mode.lower()=='fast':
        Vprime = construct_prefix_vec(Vs, P) # Vprime is a sparse matrix of prefix vectors
        simmode = 'sparse'
    else:
        Vprime = V[:] # prefix is same as original
        simmode = 'regular'
    Q = [[] for i in V]
    nvec = len(V)
    Q = sps.csr_matrix(np.zeros((nvec,nvec))) # initialize sparce matrix with all zeros
    for di in range(len(L)):
        L_di = L[di]
        L_di = sorted(L_di)
        # below: measure sim(vi,vk) for vectors in in L_di
        for idx, j in enumerate(L_di): # select vj
            vj = Vprime[j]
            for k in L_di[idx+1:]: # select vk
                if (Q[j,k]==0) & (Q[k,j]==0): # if they have not already been measured
                    vk = Vprime[k]
                    sim_jk = cos_sim(vj, vk, mode=simmode, normalize=normalize)
                    Q = update_Knn_Graph(Q, Knn, sim_jk, j, k)
                    Q = update_Knn_Graph(Q, Knn, sim_jk, k, j)
    return Q, Vprime
        
#%%

def Greedy_filtering(V, mu):
    Vs = sort_vec(V) # sorts elements desacendingly and keep track of their original dimension/index
    L = [[] for i in Vs[0]] # L is of length equal to diemntion of cectors vi. at each dimention (in a row of L), the vectors keep their elements at that dimention. e.g. L=[[0,1],[1,3],[0,2,3]], there are four vectors with 3 dimentions, v0 keeps v0[0]&v0[2] as prefix, v1 keeps v1[0]&v1[1] as prefix, only v2[2] for v2, and v3[1]&v3[2] for v3 prefix  
    P = [-1 for i in Vs] # will be filled with the number of prefixes for each vi
    c = 0 # the counter (index at sorted vectors) to find matches to vi at 0 to c indeces
    R = Vs[:]
    r_idx = range(len(R)) # this is only used to manage indices when removing vectors from R
    while len(R)>0:
        for vi_ix, vi in zip(r_idx,R): # This loop tells us when the prefix index mu is set to "c", which vi's are being match at each dimension
            e_c_v = vi[c]  #(idx, val)
            dim = e_c_v[0] #idx
            L[dim].append(vi_ix) #L[dim].append(vi)
        R_tmp = R[:]
        r_idx_bkup = r_idx[:]
        for idx, vi in zip(r_idx_bkup,R_tmp):
            M = 0 # number of matches
            for j in range(c+1): # This loop now counts the number of matches for each vector when the prefix index mu is set to "c"
                e_j_v = vi[j]  #(idx, val)
                dim = e_j_v[0] #idx
                M = M + len(L[dim]) - 1 # minus one is to remove the vi itself
            if (M>=mu) or (c+1>=len(vi)):
                P[idx] = (c+1,M) # the depth we had to go to find at least mu matched to vi, and the #of matches found for that depth
                ix = r_idx.index(idx)
                R.pop(ix) # remove vi
                r_idx.pop(ix)
        c += 1
    return L, P, Vs

#%%
    

def Example1():
    print '######################## EXAMPLE 1 ########################'
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    #v_prime1 = [0.5,0,0.37,0,0,0,0,0,0,0]
    #v_prime2 = [0.73,0.55,0,0,0,0,0,0,0,0]
    #v_prime3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0]
    #v_prime4 = [0,0,0.3,0.35,0.8,0,0,0,0,0]
    #v_prime5 = [0,0,0.48,0,0.37,0,0,0,0,0]
    
    V = [v1, v2, v3, v4, v5]
    for i in range(len(V)):
        for j in range(i+1,len(V)):
            print 'sim(v\'[%d],v\'[%d]) = %f' %(i+1, j+1, cos_sim(V[i],V[j],mode='regular', normalize=False)) 


    print '#################################################################\n\n'
    return 
    
#%%
    
def Example2():
    print '######################## EXAMPLE 2 ########################'
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    
    V = [v1, v2, v3, v4, v5]

    L, P, Vs = Greedy_filtering(V, mu=2)

    Knn = 2
    mode= 'regular'
    normalize = False
    
    Q, Vprime = Brute_force_search(Knn, L, P, V, Vs=Vs, mode=mode, normalize=normalize)
    
    print '--------------- KNN GRAPH for K=%d ----------------' %(Knn)
    print Q.toarray()
    print '---------------------------------------------------'
    print '#################################################################\n\n'
    return 
    
#%%

def Example3():
    print '######################## EXAMPLE 3 ########################'
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    
    V = [v1, v2, v3, v4, v5]
    
    mu = 2
    L, P, Vs = Greedy_filtering(V, mu)
    
    print('')
    print '--------------- P & L from Algorithm 1 ----------------'
    print 'P(depth,#match) = ', P
    print 'L = ', L
    print('')
    print '#################################################################\n\n'
    return 
    
#%%


def Example4():
    print '######################## EXAMPLE 4 ########################'
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    
    V = [v1, v2, v3, v4, v5]
    
    mu = 2
    
    L, P, Vs = Greedy_filtering(V, mu)

    Knn = 2
    mode= 'fast'
    normalize = False
    
    Q, Vprime = Brute_force_search(Knn, L, P, V, Vs=Vs, mode=mode, normalize=normalize)
    
    print '--------------- KNN GRAPH for K=%d ----------------' %(Knn)
    print Q.toarray()
    print '---------------------------------------------------'
    print '#################################################################\n\n'
    return 
    
    
#%%
#%%########### ----- MAIN ----- ############

    
if __name__ == '__main__':
    
    Example1()
    Example2()
    Example3()
    Example4()
    
    
    
    
    