# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:39:30 2015

@author: Yasser
"""

import pandas as pd
import numpy as np
import scipy.sparse as sps
import time
#%%

def get_data(filename, sep='\t'):
    data = pd.read_csv(filename, sep=sep,engine='python', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    return data

#%%

def TF_IDF(V, normalize=False):
    # Term freq * inverse doc freq. Here each dimention/item is a doc. Term freq emphasizes the value of large elements in the item/vector, and IDF dimishes its value if a large value gets repeated a lot in the set of items by the same user 
    # note: elements with value of zero rate mean they have not been rated hence we do not do TF-IDF on them and keep them zero. 
    V = np.array(V, dtype=float) # I have to make it float so that normalization does not zero everything
    nitem = V.shape[0] # num. of items/dimentions
    U = np.float(V.shape[1]) # num. of users
    for idx in range(nitem):
        if np.max(V[idx,:])>0: # if all zero, then pass
            FU = np.sum(V[idx,:]==V, axis=0) # frequency of each user's value across items for the current dimension/item
            V[idx,:] = (0.5 + 0.5*np.array(V[idx,:])/(1.0*np.max(V[idx,:]))) * (np.log(U/FU)) * (V[idx,:]>0) # TF-IDF value at the current dimension/item
            if normalize:
                Vnorm = np.linalg.norm(V[idx,:]) 
                if Vnorm>0:
                    V[idx,:] = V[idx,:]/np.float(Vnorm) # normalize to have norm=1
                else:
                    pass
            else:
                pass
        else:
            pass
        
    return V
    
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
    
def Brute_force_search(Knn, L, P, V, Vs=[], mode='fast', normalize=False, verbose=False):
    if mode.lower()=='fast':
        Vprime = construct_prefix_vec(Vs, P) # Vprime is a sparse matrix of prefix vectors
        simmode = 'sparse'
    else:
        Vprime = V[:] # prefix is same as original
        simmode = 'regular'
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
        if verbose:
            if di%(len(L)/10)==0:
                print 'Brute_force_search::  %d percent is done...' %(np.round(100*np.float(di)/(1.0*len(L))))
                
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
    # regular greedy filtering
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
    # Fast greedy filtering
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
    
def Example5():
    # regular greedy filtering with TF-IDF preprocessing
    print '######################## EXAMPLE 2 ########################'
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    
    V = [v1, v2, v3, v4, v5]
    
    V = TF_IDF(V) # preprocessing
    
    L, P, Vs = Greedy_filtering(V, mu=2)

    Knn = 2
    mode= 'regular'
    normalize = True # now we need to normalize due to loss of normalization in TF-IDF preprocessing 
    
    Q, Vprime = Brute_force_search(Knn, L, P, V, Vs=Vs, mode=mode, normalize=normalize)
    
    print '--------------- KNN GRAPH for K=%d ----------------' %(Knn)
    print Q.toarray()
    print '---------------------------------------------------'
    print '#################################################################\n\n'
    return 
    
#%%

def get_Knn_idx(Q_knn):
    Q_knn_idx = []
    for q in Q_knn:
        q_s = sorted(enumerate(q),reverse=True, key=lambda x: x[1]) #[(idx1, val1),(idx2,val2), ...]
        Q_knn_idx.append(filter(lambda x: True if x[1]>0 else False, q_s))
        
    return Q_knn_idx
    
#%%
    

def fast_recommendation():
    # RCF+TFIDF+GF Algorithm for the example item vector used in above examples
    print '######################## EXAMPLE 6 ########################'
    
    v1 = [0.5,0,0.37,0.31,0,0,0,0.33,0.23,0]
    v2 = [0.73,0.55,0.1,0,0.37,0,0,0.05,0,0]
    v3 = [0.25,0.4,0,0,0,0.27,0.29,0,0,0.1]
    v4 = [0.25,0.27,0.3,0.35,0.8,0,0,0,0,0]
    v5 = [0,0,0.48,0.32,0.37,0,0.34,0,0,0.2]
    
    V = [v1, v2, v3, v4, v5]
    
    activeuser = 1 # change this to your user of choice to see rating prediction results for that user
    Knn=2
    Kprime=2
    
    if Kprime>Knn:
        raise ValueError('Kprime must be less than or equal to Knn !! \n')

        
    V = TF_IDF(V, normalize=True) # preprocessing. # now we need to normalize due to loss of normalization in TF-IDF preprocessing 
    
    L, P, Vs = Greedy_filtering(V, mu=2)

    Knn = Knn
    mode= 'regular'
    normalize = False 
    
    Q_knn, Vprime = Brute_force_search(Knn, L, P, V, Vs=Vs, mode=mode, normalize=normalize)
    Q_knn = Q_knn.toarray()
    
    Q_knn_idx = get_Knn_idx(Q_knn)

    
    Ir = np.argwhere(V[:,activeuser]>0).flatten() # set of rated items for activeuser
    Iu = np.argwhere(V[:,activeuser]==0).flatten()  # set of unrated items for activeuser

    S = [[] for v in V[:,activeuser]] # set of items whose knn include the jth unrated item
    S_sim = [[] for v in V[:,activeuser]] # Knn similarity values of S
    for j in Iu:
        for i in Ir:
            nn = filter(lambda x: True if x[0]==j else False, Q_knn_idx[i])
            if len(nn)>0:
                S[j].append(i) # retain i
                S_sim[j].append(nn[0][1]) # retain the similarity of <i,j>
        if len(S[j])>Knn:
            S[j] = np.array(S[j])
            arg = np.argsort(S_sim[j])
            S[j] = S[j][arg[-Kprime:][-1::-1]] # retain only the most similar Kprime elements, descendingly
    
    r_avg = np.average(V, weights=np.array(V>0, dtype=int), axis=1) # average of non-zero ratings
    ra = V[:,activeuser] # ratings of the activeuser
    PR = -1*np.ones((Q_knn.shape[0],1)) # prediction matrix for the activeuser's unrated elements in Iu
    for j in Iu:
        n = S[j] # neighbors of item j
        if len(n)<Kprime: # do NOT do prediction if there was less than Kprime neighbors
            PR[j,0] = 0
        else: # do the prediction only if there was Kprime neighbors
            sim_jn = np.zeros(len(n)) # cosine similarity between j and its neighbors n
            for ix, item in enumerate(n):
                sim_jn[ix] = cos_sim(V[j,:], V[item,:], mode='regular', normalize=True)
            PR[j,0] = r_avg[j] + np.sum((ra[n]-r_avg[n])*sim_jn)
            
            
    print 'Prediction ([-1]: already rated, [0]: not enough neighbors, [>0]: predicted rating) \n', PR
    return PR    
    
#%%
 
def get_training_Knn_Graph(V, Knn, test_index, mu=300, verbose=False):
    # create the Knn graph given Matrix V of item-user by the Greedy filtering algorithm
    tm = time.time()
    V = TF_IDF(V, normalize=True) # preprocessing. # now we need to normalize due to loss of normalization in TF-IDF preprocessing 
    if verbose:
        print 'TF-IDF finished after %.1f sec ..! \n' %(time.time() - tm)
    
    mask = np.zeros(V.shape)
    mask[test_index[0],test_index[1]] = 1
    V_ts = V*mask
    V_tr = V*(1-mask)
    
    tm = time.time()
    L, P, Vs = Greedy_filtering(V_tr, mu)
    if verbose:
        print 'Greedy_filtering finished after %.1f sec ..!\n' %(time.time() - tm)

    Knn = Knn
    mode= 'regular'
    normalize = False 
    tm = time.time()
    Q_knn, Vprime = Brute_force_search(Knn, L, P, V=V_tr, Vs=Vs, mode=mode, normalize=normalize, verbose=verbose)
    Q_knn = Q_knn.toarray()
    if verbose:
        print 'Brute_force_search finished after %.1f sec ..! \n' %(time.time())
    
    tm = time.time()
    Q_knn_idx = get_Knn_idx(Q_knn)
    if verbose:
        print 'get_Knn_idx finished after %.1f sec ..! \n' %(time.time() - tm)
    
    return Q_knn, Q_knn_idx, V_tr, V_ts

#%%

def predict_rating(V, Q_knn_idx, Knn, Kprime, activeuser, item_to_predict=[]):
    Iu = np.array(item_to_predict)  # set of unrated items for activeuser
    Ir = np.argwhere(V[:,activeuser]>0).flatten() # set of rated items for activeuser

    S = [[] for v in Iu] # set of items whose knn include the jth unrated item
    S_sim = [[] for v in Iu] # Knn similarity values of S
    for idx, j in enumerate(Iu):
        for i in Ir:
            nn = filter(lambda x: True if x[0]==j else False, Q_knn_idx[i])
            if len(nn)>0:
                S[idx].append(i) # retain i
                S_sim[idx].append(nn[0][1]) # retain the similarity of <i,j>
        if len(S[idx])>Knn:
            S[idx] = np.array(S[idx])
            arg = np.argsort(S_sim[idx])
            S[idx] = S[idx][arg[-Kprime:][-1::-1]] # retain only the most similar Kprime elements, descendingly
    
    weights = np.array(V>0, dtype=int)
    idx = np.sum(weights, axis=1)==0
    weights[idx,:] = 1
    r_avg = np.average(V, weights=weights, axis=1) # average of non-zero ratings
    ra = V[:,activeuser] # ratings of the activeuser
    PR = -1*np.ones(len(Iu)) # prediction matrix for the activeuser's unrated elements in Iu
    for idx, j in enumerate(Iu):
        n = S[idx] # neighbors of item j
        if len(n)<Kprime: # do NOT do prediction if there was less than Kprime neighbors
            PR[idx] = 0
        else: # do the prediction only if there was Kprime neighbors
            sim_jn = np.zeros(len(n)) # cosine similarity between j and its neighbors n
            for ix, item in enumerate(n):
                sim_jn[ix] = cos_sim(V[j,:], V[item,:], mode='regular', normalize=True)
            PR[idx] = r_avg[j] + np.sum((ra[n]-r_avg[n])*sim_jn)
            
    return PR    


#%%########### ----- MAIN ----- ############
#%%    
if __name__ == '__main__':

    Example1()
    Example2()
    Example3()
    Example4()
    Example5()
    fast_recommendation()
    
    # --- Load training and test data
    
    #filename_tr = r'C:\Users\ghanby01\PROJECTS\Recommendation System\MovieLens\100K Dataset\ml-100k\u1.base'
    #data_tr = get_data(filename_tr, sep='\t')
    #filename_ts = r'C:\Users\ghanby01\PROJECTS\Recommendation System\MovieLens\100K Dataset\ml-100k\u1.test'
    #data_ts = get_data(filename_tr, sep='\t')
    
    filename = r'C:\Users\Abolfazl\Downloads\Documents\MovieLens\1M Dataset\ml-1m\ratings.dat'
    data = get_data(filename, sep='::')

    V = sps.csr_matrix((data['rating'], (data['item_id']-1, data['user_id']-1)), shape=(np.max(data['item_id']), np.max(data['user_id']))).toarray()
    
    # --- Preprocessing: Train the Knn Graph. Constructing training and test sets
    Knn = 10
    Kprime = 10
    mu = 300
    test_size = 500
    
    data_ts = data.loc[np.random.randint(len(data), size=test_size),:]
    
    test_index = [list(data_ts['item_id']-1), list(data_ts['user_id']-1)]
    
    Q_knn, Q_knn_idx, V_tr, V_ts = get_training_Knn_Graph(V, Knn, test_index, mu, verbose=True)

    
    # --- Predict ratings for test
    tm = time.time()
    random_select = 1000
    N = 10
    hits = -1*np.ones(test_size) # 0: no hit, 1: hit in prediction
    for ix, i in enumerate(data_ts.index):
        #tm = time.time()
        activeuser = data_ts.at[i,'user_id'] - 1 # minus one to make it index
        unrated_item = data_ts.at[i,'item_id'] - 1 # minus one to change it to index
        all_unrated_items_of_activeuser = np.argwhere(V_tr[:,activeuser]==0).flatten() # all unrated by activeuser
        rnd_idx = np.random.randint(len(all_unrated_items_of_activeuser), size=random_select) # create 1000 random indices
        random_unrated_items = np.array(list(all_unrated_items_of_activeuser))[rnd_idx] # get 1000 unrated items from activeuser
        random_unrated_items = np.delete(random_unrated_items, np.argwhere(random_unrated_items==unrated_item)) # if unrated_item has been caught, remove it from the random set
        item_to_predict = [unrated_item]+list(random_unrated_items) # append the 1000 randomly selected itemss to the intem-to-predict
        PR = predict_rating(V_tr, Q_knn_idx, Knn, Kprime, activeuser, item_to_predict)
        if 0 in np.argsort(PR)[-N:][-1::-1]: # if the 0th item, i.e. the unrated_item, is within the top N prediction scores then it is a hit
            hits[ix] = 1
        else:
            hits[ix] = 0 # no hit
        #print time.time() - tm
        if ix%(len(data_ts)/10)==0:
            print 'Prediction on test data::  %d percent is done...' %(np.round(100*np.float(ix)/(1.0*len(data_ts))))
            
        #if ix>100:
        #    break

    print '------ recall = %f' %(np.sum(hits[:ix])/np.float(len(hits[:ix])))
    print '------ Average Elapsed time per test case = %f sec ' %((time.time() - tm)/(ix+1))


