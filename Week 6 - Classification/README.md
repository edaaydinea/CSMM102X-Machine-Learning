Week 6 Project : Classification
===============================

### **INSTRUCTIONS**

Assume you are given labeled data <img src="https://render.githubusercontent.com/render/math?math=(x_1,y_1),\dots,(x_N,y_N)"> , where <img src="https://render.githubusercontent.com/render/math?math=x\in\mathbb{R}^d">  and <img src="https://render.githubusercontent.com/render/math?math=y\in\{1,\dots,K\}">. In this assignment, you will implement a *K*-class Bayes classifier. In the specific classifier that you will implement, assume the following generative model: For the *i*-th data point, assume that

<img src="https://render.githubusercontent.com/render/math?math=y_i \stackrel{iid}{\sim} Discrete(\pi),\quad x_i|y_i \sim Normal(\mu_{y_i},\Sigma_{y_i}),\quad i = 1,\dots, N.">

For this model, you will need to derive the maximum likelihood updates for the class prior probability vector <img src="https://render.githubusercontent.com/render/math?math=\widehat{\pi}"> and the class-specific Gaussian parameters <img src="https://render.githubusercontent.com/render/math?math=(\widehat{\mu}_k,\widehat{\Sigma}_k)">  for each class <img src="https://render.githubusercontent.com/render/math?math=k=1,\dots,K">, where  indicates "maximum likelihood estimate". While you will not turn in these derivations, you will need to implement them in your code, as well as the prediction for a new point <img src="https://render.githubusercontent.com/render/math?math=y_0"> given <img src="https://render.githubusercontent.com/render/math?math=x_0"> and these estimates:

<img src="https://render.githubusercontent.com/render/math?math=Prob(y_0=y|x_0,\widehat{\pi},(\widehat{\mu}_1,\widehat{\Sigma}_1),\dots,(\widehat{\mu}_K,\widehat{\Sigma}_K))">

More details about the inputs we provide and the expected outputs are given below.

***Sample starter code to read the inputs and write the outputs:  [Download hw2_classification.py](https://courses.edx.org/assets/courseware/v1/9bdea8f7c66a1ab0deea07cda0f1f68f/asset-v1:ColumbiaX+CSMM.102x+1T2021+type@asset+block/hw2_classification.py)***

### **WHAT YOU NEED TO SUBMIT**

You can use either **Python (3.6.4)** or Octave coding languages to complete this assignment. Octave is a free version of Matlab. Your Matlab code should be able to directly run in Octave, but you should not assume that advanced built-in functions will be available to you in Octave. Unfortunately we will not be supporting other languages in this course.


Depending on which language you use, we will execute your program using one of the following two commands.


Either

    $ python3 hw2_classification.py X_train.csv y_train.csv X_test.csv

Or

    $ octave -q hw2_classification.m X_train.csv y_train.csv X_test.csv


You must name your file as indicated above for your chosen language. If both files are present, we will only run your Python code. We will create and input the csv data files to your code.


The csv files that we will input into your code are formatted as follows:.

1.  **X_train.csv:** A comma separated file containing the covariates. Each *row* corresponds to a single vector . <img src="https://render.githubusercontent.com/render/math?math=x_i">
2.  **y_train.csv:** A file containing the classes. Each row has a single number and the *i*-th row of this file combined with the *i*-th row of "X_train.csv" constitutes the labeled pair <img src="https://render.githubusercontent.com/render/math?math=(x_i,y_i)">. There are 10 classes having index values 0,1,2,...,9.
3.  **X_test.csv:** This file follows exactly the same format as "X_train.csv".No class file is given for the testing data.

### **WHAT YOUR PROGRAM OUTPUTS**

When executed, you should have your code write the output to the file listed below. It is required that you follow the formatting instructions given below.

probs_test.csv: This is a comma separated file containing the posterior probabilities of the label of each row in "X_test.csv". Since there are 10 classes, the *i*-th row of this file should contain 10 numbers, where the *j*-th number is the probability that the *i*-th testing point belongs to class *j-1* (since classes are indexed 0 to 9 here).


**Note on Correctness**

Please note that for both of these problems, there is one and only one correct solution. Therefore, we will grade your output based on how close your results are to the correct answer. We strongly suggest that you test out your code on your own computer before submitting. The UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/) has a good selection of datasets for classification.
