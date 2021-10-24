from numpy import transpose as t
from numpy import matrix as mat, matmul as mm

from numpy import linalg as lin
from numpy.linalg import inv
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
from Level import Delta
from PostAdjustmentTester import PostAdjustmentTester


class Network(LS, PostAdjustmentTester):
    """
    Build to run the least squares adjustment and set up the overall network
    """
    def __init__(self, models, net_type = "Photo"):
        """
        Desc:
        Input:
            models: list of models that have been initialized with 
                data. Must contain the same number of columns in their a 
                matrix (predefined by LS())
        Output:
        """       
        LS.__init__(self)
        PostAdjustmentTester.__init__(self)
        
        #for picking things
        self.net_type = net_type
        
        self.models = models
        
        if self.net_type == "Photo":
            #_________________setup first round of stuff______________________
            self.initialize_variables()

            #_________________begin LSA______________________
            #self.photo_LSA()

            #_________________format matrices for outputting statistics______________________
            #self.photo_mats()
        
        #_________________output statistics______________________
        #self.final_matrices()
        
    def initialize_variables(self):
        """
        Desc:
            initializes major variables (combining matrices and stuff)        
        
        Input:
        Output:
           self.u
        """
        self.ue = self.models[0].ue
        self.uo = self.models[0].uo
        self.u = self.models[0].ue + self.models[0].uo
        
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
        
        if self.net_type == "Photo":
            #set up control weight errors matrix
            temp = []
            for obs in self.models:
                temp.append(obs.errs_o)
            self.errs_o = np.vstack(temp)
            
            #set up control weight errors matrix
            temp = []
            for obs in self.models:
                temp.append(obs.errs_e)
            self.errs_e = np.vstack(temp)
        
        #set up number of observations variable
        self.n = len(self.errs)
        
        #set up design matrix
        self.design()
        
        #set up covariance (no additional formatting needed)
        self.covariance()
        
        #set up apriori
        self.apriori = 1
        
        #set up weight matrix
        self.P = self.apriori**2 * inv(self.Cl)
        
        if self.net_type == "Photo":
            #then a Po will also need to be made
            self.Po = mat(np.zeros((self.uo, self.uo)))
        
            for i in range(0,self.uo):
                if self.errs_o[i] != 0:
                    self.Po[i,i] = 1/self.errs_o[i]**2
                    
            #then a Po will also need to be made
            self.Pe = mat(np.zeros((self.ue, self.ue)))
        
            for i in range(0,self.ue):
                if self.errs_e[i] != 0:
                    self.Pe[i,i] = 1/self.errs_e[i]**2
        
                
    def final_matrices(self):
        """
        Desc:
            Once the LSA is completed then this generates all desired matrices for analysis
        Input:
        Output:
            self.r_hat: residuals
            self.l_hat: adjusted observations
            self.a_post: a-posteriori variance factor
            self.uvf: unit variance factor
            self.Cx (also Cs): 
            self.Cl:
            self.Cr:
        """
        
        self.r_hat = mm(self.A,self.S_hat) + self.w_0
        self.l_hat = self.obs + self.r_hat
        self.a_post = m.sqrt(mm(t(self.r_hat),mm(self.P,self.r_hat)/(self.n-self.u))[0,0])
        self.uvf = self.a_post**2 / self.apriori**2
        
        self.Cx = self.a_post**2 * inv(mm(t(self.A),mm(self.P,self.A)))
        if self.net_type == "Photo":
            self.Cx = inv(self.N)
        #self.plot_mat(self.Cx, "Covariance Matrix of Unknowns")
        
        self.Cl = mm(self.A,mm(self.Cx,t(self.A)))
        #self.plot_mat(self.Cl, "Covariance Matrix of Measurements")
        
        self.Cr = self.a_post**2*inv(self.P)-self.Cl
        #self.plot_mat(self.Cr, "Covariance Matrix of Residuals")
        
        if self.net_type == "Photo":
            pass
        
    def nonlinear_LSA(self):
        """
        Desc:
            Iterates a nonlinear LSA, checking whether criterea was met. Once it was met then it constructs the final matrices for analysis
        Input:
        Output:
        
        """
        self.not_met = True
        
        i = 0
        
        self.w_0 = mat(np.zeros((self.n, 1)))
        self.S_hat = mat(np.zeros((self.n, 1)))
        self.x_hat = mat(np.zeros((self.n, 1)))
        
        while self.not_met:
            i = i + 1
            #print("LSA iteration: " + str(i))
            #print("x_0: ")
            #print(LS.x_0)
            
            #l_0
            self.obs_0()
            
            #update l_0 and A
            self.update_values()

            #misclosure
            
            self.w_0 =  self.l_0 - self.obs

            #S_hat
            
            self.S_hat = -mm(inv(mm(t(self.A),mm(self.P,self.A))),mm(t(self.A),mm(self.P,self.w_0)))

            #print("l_0: ")
            #print(self.l_0)
            
            #x_hat
            self.x_hat = LS.x_0 + self.S_hat
           
                
            #update x_0
            LS.x_0 = self.x_hat
          
            #print("S_hat:")
            #print(self.S_hat)
            #print("x_hat: ")
            #print(self.x_hat)
            #print("A: ")
            #print(self.A)
            
            
            
            
            
            
            self.convergence(i)
        
        #print("LSA passed in: " + str(i) + " iterations")
        #self.final_matrices()
        
    def photo_LSA(self):
        """
        Desc:
            Iterates a nonlinear LSA, checking whether criterea was met. Once it was met then it constructs the final matrices for analysis
        Input:
        Output:
        
        """
        self.not_met = True
        
        i = 0
        
        
        self.w_0 = mat(np.zeros((self.n, 1)))
        self.S_hat = mat(np.zeros((self.n, 1)))
        self.x_hat = mat(np.zeros((self.n, 1)))
        
        while self.not_met and i < 5:
            i = i + 1
            
            #l_0
            self.obs_0()
            
            #update l_0 and A
            self.update_values()

            #misclosure
            self.w_0 =  self.l_0 - self.obs
            
            self.set_N()
            self.set_U()
            
            #S_hat
            self.S_hat = -mm(inv(self.N),self.U)
            
            #x_hat
            self.x_hat = LS.x_0 + self.S_hat
           
            #update x_0
            LS.x_0 = self.x_hat
          
            self.convergence(i)
        
        print("LSA passed in: " + str(i) + " iterations")
        #self.final_matrices()
        
        #not 100% sure but probably
        self.A = self.N
    
    def error_ellipses(self):
        """
        Desc:
            generates the error ellipses, minor, major, bearing_major
            **must already have self.Cx generates**
            **assumes Xa, Ya, Xb, Yb, Xc, Yc etc in the Cx diagonal**
        Input:
        Output:
        """
        self.u
        ellipses = []
        for i in range(0,self.Cx.shape[0],2):
            q11 = self.Cx[i,i]
            q12 = self.Cx[i,i+1]
            q21 = self.Cx[i+1,i]
            q22 = self.Cx[i+1,i+1]
            ellipses.append(self.ellipse(q11, q12, q21, q22))
            
        return ellipses
            
    def ellipse(self, q11, q12, q21, q22):
        """
        Desc:
            Calculates the error ellipse, returns back a dataframe of the values
        Input:
            q11, 
            q12, 
            q21, 
            q22
        Output:
            {
            "minor": float,
            "major": float,
            "major_orientation": radians
            }
        """
        minor = m.sqrt(abs((q11 + q22 - m.sqrt((q11-q22)**2+4*(q12**2)))/2))
        major = m.sqrt(abs((q11 + q22 + m.sqrt((q11-q22)**2+4*(q12**2)))/2))
        
        major_orientation = m.atan(q12/(major**2-q22))
        
        return {
            "minor": minor,
            "major": major,
            "major_orientation": major_orientation
            }
        
        
            
        
    def convergence(self,i):
        """
        Desc:
            Checks based on this criterea, if convergence is met then sets self.not_met to False
        Input:
            i: number of iterations (for simple # of ter break)
        Output:
            self.not_met --> False if the criterea is met
        """
        #max 10 iterations
        if i > 3:
            self.not_met = False
            
        #minimum self.S_hat to be under .001m
        
        not_under = False
        for key in self.S_hat:
            if abs(key[0,0]) > .0001:
                #this means the criterea was not met for atleast one of the unknowns
                not_under = True

        if not not_under:
            #then all things were under .0001m in change and therefore the criterea was met
            self.not_met = False
        
        
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
            self.Cl[i,i] = self.errs[i]**2
            
   
    def photo_mats(self):
        """
        Desc:
            Sets up matrices needed for statistics
        Input:
        Output:
           self.A
           self.S
        """
        self.A = np.concatenate((self.Ae,self.Ao), axis = 1)
        self.u_list = []
        for i in self.models[0].u_list_ae:
            self.u_list.append("{}Xcj".format(i))
            self.u_list.append("{}Ycj".format(i))
            self.u_list.append("{}Zcj".format(i))
            self.u_list.append("{}w".format(i))
            self.u_list.append("{}o".format(i))
            self.u_list.append("{}k".format(i))
            
        for i in self.models[0].u_list_ao:
            self.u_list.append("{}Xi".format(i))
            self.u_list.append("{}Yi".format(i))
            self.u_list.append("{}Zi".format(i))

        
    def design(self):
        """
        Desc:
            Set up overall design matrix
        Input:
        Output:
           self.A
        """
        #self.A = mat(np.zeros((self.n, self.u)))
                    
        #temp = []
        #for model in self.models:
        #    temp.append(model.A)
        #self.A = np.vstack(temp)
        self.Ae = self.models[0].Ae
        self.Ao = self.models[0].Ao
    
    
    def set_N(self):
        """
        Desc:
            Sets up the N matrix with the four quadrants
        Input:
            self.Ae
            self.Ao
            self.P
        Output:
            self.Nee
            self.Neo
            self.Noo
            self.N
        """
        self.Nee = mm(t(self.Ae),mm(self.P,self.Ae))+self.Pe
        self.Neo = mm(t(self.Ae),mm(self.P,self.Ao))
        self.Noo = mm(t(self.Ao),mm(self.P,self.Ao))+self.Po
        
        a = np.concatenate((self.Nee,self.Neo), axis = 1)
        b = np.concatenate((t(self.Neo),self.Noo), axis = 1)
        
        self.N = np.concatenate((a,b), axis = 0)
        
    def set_U(self):
        """
        Desc:
            Sets up the U matrix with the two halves
        Input:
            self.Ae
            self.Ao
            self.P
            self.w_0
        Output:
            self.Ue
            self.Uo
            self.U
        """
        #self
        self.w_0_e = LS.x_0_ae - self.x_0[0:self.ue,0]
        self.w_0_o = LS.x_0_ao - self.x_0[self.ue:,0]
        
        self.Ue = mm(t(self.Ae),mm(self.P,self.w_0))+mm(self.Pe,self.w_0_e)
        self.Uo = mm(t(self.Ao),mm(self.P,self.w_0))+mm(self.Po,self.w_0_o)
        #print(self.Uo.shape)
        self.U = np.concatenate((self.Ue,self.Uo), axis = 0)
               
        
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
        self.w = mm(self.A,LS.x_0) - self.obs
           
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
        Desc:
            Assembles l_obs from each matrix
        Input:
        Output:
            self.l_0 constructed
        """
        
        self.l_0 = mat(np.zeros((self.n, 1)))
                    
        temp = []
        for obs in self.models:
            temp.append(obs.l_0)
        self.l_0 = np.vstack(temp)
        
    def update_values(self):
        """
        Desc:
            Updates x_0 and design and l_0
        Input:
            Uses most recent x_hat value
        Output:
            none:
        """
        #update models
        for model in self.models:
            #model.x_0 = self.x_0
            
            model.obs_0()
            
            #update design matrix
            model.set_design()
           
        #update within network
        self.design()
        self.obs_0()