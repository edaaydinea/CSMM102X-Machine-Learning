from __future__ import division
import csv 
import sys
import numpy as np
np.set_printoptions(precision=6)

def EMGMM(data):
    #Lecture 16, slide 20
    classes = 5 #number of classes
    iterationMax = 10
    length = data.shape[0]
    dim = data.shape[1]
    Sigma_k = np.eye(dim)
    Sigma = np.repeat(Sigma_k[:,:,np.newaxis],classes,axis=2) #initialize Sigma to identity matrix   
    piClass = np.ones(classes)*(1/classes) #initialize with uniform probability distribution
    phi = np.zeros((length,classes))
    phiNorm = np.zeros((length,classes))
#    #initialize the mu with uniform random selection of data points
    indices = np.random.randint(0,length,size=classes)
    mu = data[indices]

    
    for iteration in range(iterationMax):
        #compute expectation step of EM algorithm
        for k in range(classes):
            invSigma_k = np.linalg.inv(Sigma[:,:,k])
            invSqrDetSigma_k = (np.linalg.det(Sigma[:,:,k]))**-0.5
            for index in range(length):
                xi = data[index,:]
                temp1 = (((xi-mu[k]).T).dot(invSigma_k)).dot(xi-mu[k])
                phi[index, k] = piClass[k]*((2*np.pi)**(-dim/2))*invSqrDetSigma_k*np.exp(-0.5*temp1)
            for index in range(length):
                tot = phi[index,:].sum()
                phiNorm[index,:] = phi[index,:]/float(tot)
        
        #compute maximization step of EM algorithm
        nK = np.sum(phiNorm,axis=0)
        piClass = nK/float(length)
        for k in range(classes):
            mu[k] = ((phiNorm[:,k].T).dot(data))/nK[k]
        for k in range(classes):
            temp1 = np.zeros((dim,1))
            temp2 = np.zeros((dim,dim))
            for index in range(length):
                xi = data[index,:]
                temp1[:,0] = xi - mu[k]                
                temp2 = temp2 + phiNorm[index,k]*np.outer(temp1,temp1)
            Sigma[:,:,k] = temp2/float(nK[k]) 
           
        #Write output to file
        path1 = "pi-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in piClass:
                writer.writerow([val])
        #Write output to file
        path1 = "mu-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in mu:
                writer.writerow(val)
        #Write output to file
        for k in range(classes):
            path1 = "Sigma-{:g}-{:g}.csv".format(k+1,iteration+1)
            with open(path1, "w") as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for val in Sigma[:,:,k]:
                    writer.writerow(val)

def kMeans(data):   
    cNum = 5 #number of clusters
    iterationMax = 10
    length = data.shape[0]
    c = np.zeros(length) #cluster assignment vector
    #initialize the mu with uniform random selection of data points
    indices = np.random.randint(0,length,size=cNum)
    mu = data[indices]
 
    for iteration in range(iterationMax):
        #Update cluster assignments ci
        for i, xi in enumerate(data):
            temp1 = np.linalg.norm(mu-xi,2,1)
            c[i] = np.argmin(temp1)
        #Update cluster mu
        n = np.bincount(c.astype(np.int64),None,cNum)      
        for k in range(cNum):
            indices = np.where(c == k)[0]
            mu[k] = (np.sum(data[indices],0))/float(n[k])
        #Write output to file
        path1 = "centroids-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in mu:
                writer.writerow(val)
                
def main():  
    file_X = np.genfromtxt(sys.argv[1], delimiter=',')
    
    kMeans(file_X)
    EMGMM(file_X)
    
if __name__ == "__main__":
    main()
