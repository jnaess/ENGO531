from numpy import transpose as t
from numpy import matrix as mat, matmul as mm

from numpy import linalg as lin
from numpy.linalg import inv
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
from Level import Delta


class Network(LS):
    """
    Build to run the least squares adjustment and set up the overall network
    """
    def __init__(self, models = [Delta()]):
        """
        Desc:
        Input:
            models: list of models that have been initialized with 
                data. Must contain the save number of columns in their a 
                matrix (predefined by LS())
        Output:
        """
        LS.__init__(self)
        self.models = models
        self.initialize_variables()
        
        
    def initialize_variables(self):
        """
        Desc:
            initializes major variables (combining matrices and stuff)
            
                    [E,N,H]
        ASCM2      -9393.784, 5660608.432, 1110.234
        ASCM1: -9904.726, 5659098.669, 1100.076 [-9903.780, 5659098.252, 1099.627]
        SA10: -9292.095, 5660358.617, 1136.040
        UCAL: -9378.481, 5660425.221, 1135.642
        BASE: -9245.156, 5660419.921, 1111.683
        C1: -9484.624, 5660402.217, 1111.207
        C2: -9492.719, 5660518.107, 1112.521
        C3: -9428.997, 5660514.752, 1114.125
        N: E, N, 1135.959
        S: E, N, 1135.887
        
        
        Input:
        Output:
           self.u
        """
                              
        self.u = self.models[0].u
        
        #set up observation matrix
        temp = []
        for obs in self.models:
            temp.append(obs.obs)
        self.obs = np.vstack(temp)
        
        #set up errors matrix
        temp = []
        for obs in self.models:
            temp.append(obs.errs)
        self.errs = np.vstack(temp)
        
        #set up number of observations variable
        self.n = len(self.errs)
        
        #set up design matrix
        self.design()
        
        #set up covariance (no additional formatting needed)
        self.covariance()
        
        #set up apriori
        self.apriori = 1
        
        #set up weight matrix
        self.P = inv(self.Cl)
        
        #l_0
        self.obs_0()

        #set up N matrix
        
        self.n_mat()
        
        self.misc =  self.l_0 - self.obs
        
        self.u = mm(t(self.A),mm(self.P,self.misc))

        self.S_hat = -mm(inv(self.N),self.u)
        
        self.x_hat = self.x_0 + self.S_hat
        
        self.v_hat = mm(self.A,self.S_hat) + self.misc 
        
        self.a_pri_hat = mm(t(self.v_hat),mm(self.P,self.v_hat))[0,0]/(self.n - len(self.x_0))
        
        self.cx_hat = self.a_pri_hat*inv(self.N)
        
        
        
        #set up first Cx
        #self.cx_mat()
        
        #set up first misclosure
        #self.w_mat()
        
        #correction vector
        #self.correction()
        
    def covariance(self):
        """
        Desc:
            Initialized covariance matrix based on observation standard deviations
        Input:
        Output:
           self.Cl
        """
        self.Cl = mat(np.zeros((self.n, self.n)))
        
        for i in range(0,self.n):
            self.Cl[i,i] = self.errs[i]*self.errs[i]
            
    def design(self):
        """
        Desc:
            Set up overall design matrix
        Input:
        Output:
           self.A
        """
        self.A = mat(np.zeros((self.n, self.u)))
        
        temp = []
        for obs in self.models:
            temp.append(obs.A)
        self.A = np.vstack(temp)
        
        
        
        #___________needs updating to merge all matrices______
        #self.A = self.models[0].A
        
    def n_mat(self):
        """
        """
        self.N = mm(t(self.A),mm(self.P,self.A))
        
    def cx_mat(self):
        """
        """
        self.Cx = inv(self.N)
        
    def w_mat(self):
        """

        """
        #adds constants and unknowns together and solves for values
        #temp = np.vstack((self.c, self.x_0))
        #self.w = mm(self.A,temp) - self.obs
        self.w = mm(self.A,self.x_0) - self.obs
           
           #____________for non linear this will need to change______
    def u_mat(self):
        """
        """
        self.v = t(self.A,mm(self.P,self.w))
        
    def correction(self):
        """
        """
        self.S = -mm(inv(self.N),mm(t(self.A),mm(self.P,self.w)))
        
    def obs_0(self):
        """
        Assembles l_obs from each matrix
        """
        
        self.l_0 = mat(np.zeros((self.n, 1)))
        
        temp = []
        for obs in self.models:
            temp.append(obs.l_0)
        self.l_0 = np.vstack(temp)
        
        