import pandas as pd 
import numpy as np 
from scipy.linalg import svd 

'''defining the signals we study '''
sig1 = lambda x: np.sin(x)
sig2 = lambda x: np.cos(x)
sig3 = lambda x: np.sin(2*x)
sig4 = lambda x: np.cos(2*x)


df = pd.read_csv(r'./data.csv') #importing signal data 

class dataframes:
    ''' defining a class to decompose the dataframe appropiately'''
    def __init__(self, data):
        '''initiating values'''
        self.data = data

        self.shape,_ = data.shape 
    
    def input_data(self):
        '''returns the input data in x'''
        xvalues = self.data['Input']
        return xvalues
    
    def output_data(self):
        '''returns the output data in y'''
        yvalues = self.data['Output']
        return yvalues        
  
    def makearray(self, varlen):
        '''constructs the array needed for solving the least squares method'''
        array = np.zeros((self.shape, varlen))
        return  array

    def fillarray(self, *args):
        '''fills the matrix with appropiate values'''
        xvals = self.input_data() #imports the x values 

        vl = len(args) #dim of the function space

        skel = self.makearray(vl) #matrix to be filled 


        k=0
        for i in xvals:
            m=0
            for j in args:
                skel[k,m] = j(i) #combining the inputs with functions in the function space 
                m+=1
            k+=1
        
        return skel 

class solution:
    '''A class to solve the least square problem'''
    def __init__(self,matrix, yvalues):
        self.matrix = matrix 
        self.yvalues = yvalues 
    
    def sing(self):
        '''Uses SVD to compute a pseudo-inverse '''
        matr = self.matrix #matrix decleration
        u,sigma, vt = svd(matr)#SVD
        v = vt.T #defines the v 
        m,n = u.shape[1], vt[0] #dimensions of the \Sigma matrix 
        sigmat = np.zeros((m,n)) #declares the \Sigma matrix
        for i in range(m):
            '''fills the diagonal'''
            sigmat[i,i] = sigma[i] 
        
        return u, sigmat, v #returns u, \Sigma, v 

    
    def least_square(self):
        '''least square method  '''
        u, s, vt = self.sing() #SVD 

        yvals = np.array(self.yvalues()) #obtain the values 

        xvec = np.matmul(vt.T@s.T@u.T, yvals) #computes the output vector

        return xvec 







classitem = dataframes(df) #defines the class item 

a = classitem.fillarray(sig1, sig2, sig3, sig4) #testing 

#b = classitem.least_square(sig1, sig2, sig3, sig4)


