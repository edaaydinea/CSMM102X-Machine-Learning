import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
n, d = X_train.shape

## Solution for Part 1
def part1(lambda_input, X_train, y_train):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    term1 = lambda_input * np.eye(d) + np.matmul(X_train.T,X_train)
    wRR = np.matmul(np.linalg.inv(term1), np.matmul(X_train.T, y_train))
    return wRR

wRR = part1(lambda_input, X_train, y_train)  # Assuming wRR is returned from the function

np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(lambda_input, X_train, X_test):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    test_set = dict((i,j) for i,j in zip(range(1, (X_test.shape[0]+1)),[row for row in X_test]))
    term1 = lambda_input * np.eye(d) + (1/sigma2_input) * np.matmul(X_train.T,X_train)
    E = np.linalg.inv(term1) #dxd
    X_new = X_train 
    output = []
    
    for i in range(max(10, len(test_set))):
        sigma_max, x_max, idx_max = 0 , np.array([]), -sys.maxsize
        
        for idx, x_o in test_set.items():
            sigma_o = sigma2_input + np.matmul(x_o, np.matmul(E, x_o))
            if sigma_o > sigma_max:
                sigma_max, x_max, idx_max = sigma_o, x_o, idx

        X_new= np.vstack([X_new, x_max])      
        term1_new = lambda_input * np.eye(d) + (1/sigma2_input) * np.matmul(X_new.T, X_new)
        E = np.linalg.inv(term1_new)
        output.append(idx_max)
        del test_set[idx_max]
    return output        

active = part2(lambda_input, X_train, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv" , active, delimiter=",") # write output to file
