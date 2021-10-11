from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from Bundle import Bundle
from Design_e import Design_e as ae

class Design_o(Bundle):
    """
    Desc:
        Generates and facilitates the manipulation of Ae
    """
    
    def __init__(self):
        """
        Desc:
        Input:
        Output:
        """
        Bundle.__init__(self)
        
        self.Ae = ae()
        
        #from LS class to find unknown columns
        self.set_col_list_ao()
        
        self.set_design()
                
    def set_obs(self):
        """
        Desc:
           uses self.pho to take the x and y and set up the observations and converts them to RHC with a bundle functions
           
           also sets errs
        Input:
            self.pho
        Output:
            self.obs: l matrix (never changes)   
            self.errs
        """
        self.obs = mat(np.zeros((self.n, 1)))
        
        #data input as ***meters***
        self.errs = mat(np.zeros((self.n, 1)))
        
        #get desired numbers in a list
        y = self.pho['y'].to_list()
        x = self.pho['x'].to_list()
        check = self.pho['knowns'].to_list()
        
        j = 0
        for i in range(0, self.n, 2):
            #set up x_ij and y_ij info
            self.rhc(x[j],y[j])
            
            #x pixel
            self.obs[i,0] = self.x_ij
            
            #y pixel
            self.obs[i+1,0] = self.y_ij
            
            #assign errors
            if check[j] == "u":
                #then tie point and larger std
                self.errs[i,0] = .01
                self.errs[i+1,0] = .01
            else:
                self.errs[i,0] = .0001
                self.errs[i+1,0] = .0001
            
            #increment index in y and x lsits
            j = j+1
            
    def set_errors(self):
        """
        Desc:
            sets control point to .01mm and current tie points to 10mm
        Input:
        Output:
            self.errs
        """
        self.errs = mat(self.df[self.d_error]).transpose()
        
    def set_design(self):
        """
        Desc:
            Initializes the design matrix
        Output:
        Input:
        """
        #set it up as just zeros
        self.A = mat(np.zeros((self.n, self.uo)))
        
        #0, 2, 4, etc. are X pixels
        #1, 3, 5, etc. are Y pixels
        #__print("n: "+str(self.n))
        for i in range(0, self.n, 2):
            #increments every two because one row is for X, one row is for Y

            #each time we should go through one observation
            #indexes every 2
            #this is the observation
            #get image id from photo obs
            obs = self.pho.iloc[int(i/2)]
            
            #get image row from ext EOP's
            #j = int(obs["image_id"])
            j = self.find_col_ao(obs["point_id"])
            
            j_2 = self.Ae.find_col_ae(obs["image_id"])            

            #then evens (X partial)
            self.A[i,j] = -self.Ae.A[i,j_2]
                #Y
            self.A[i,j + 1] = -self.Ae.A[i,j_2+1]
                #Z
            self.A[i,j + 2] = -self.Ae.A[i,j_2+2]
            
            #then odds (Y partial)
                #X
            self.A[i+1,j] = -self.Ae.A[i+1,j_2]
                #Y
            self.A[i+1,j + 1] = -self.Ae.A[i+1,j_2+1]
                #Z
            self.A[i+1,j + 2] = -self.Ae.A[i+1,j_2+2]