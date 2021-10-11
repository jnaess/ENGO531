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
        
        self.set_obs()
        
        self.set_X_0()
        
        self.obs_0()
                
    def set_obs(self):
        """
        Desc:
           uses self.pho to take the x and y and set up the observations and converts them to RHC with a bundle functions
           
           sets control point to .01mm and current tie points to 10mm
        Input:
            self.pho
        Output:
            self.obs: l matrix (never changes)   
            self.errs
        """
        self.obs = mat(np.zeros((self.n, 1)))
        
        #data input as ***mm***
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
                self.errs[i,0] = 10
                self.errs[i+1,0] = 10
            else:
                self.errs[i,0] = .01
                self.errs[i+1,0] = .01
            
            #increment index in y and x lsits
            j = j+1
            
    def set_X_0(self):
        """
        Desc:
            Sets up X_0 from the dataframe values
        Input:
        Output:
            
        """
        #assumes images already sorted in ascending order
        #assumes camera also sorted
        x_0_ae = []
        for index, row in self.ext.iterrows():
            x_0_ae.append(row["Xc"])
            x_0_ae.append(row["Yc"])
            x_0_ae.append(row["Zc"])
            x_0_ae.append(m.radians(row["w"]))
            x_0_ae.append(m.radians(row["o"]))
            x_0_ae.append(m.radians(row["k"]))
        
        x_0_ao = []
        for index, row in self.obj.iterrows():
            x_0_ao.append(row["X"])
            x_0_ao.append(row["Y"])
            x_0_ao.append(row["Z"])
            
        self.x_0 = t(mat(x_0_ae+x_0_ao))
        
    def obs_0(self):
        """
        desc:
            Sets up self.l_0 (extimated observations)
            Used for finding the current misclosure
            Assumes only one camera for IOP's from self.int
        input:
            self.x_0
        output:
            self.l_0
        """
        self.xp = self.int["xp"][0]
        self.yp = self.int["yp"][0]
        self.c = self.int["c"][0]
        
        #set it up as just zeros
        self.l_0 = mat(np.zeros((self.n, 1)))
        
        for i in range(0, self.n, 2):
            obs = self.pho.iloc[int(i/2)]
            
            #row for ae parameters
            j = self.Ae.find_col_ae(obs["image_id"])
            #row for ue parameters
            j_2 = self.ue + self.find_col_ao(obs["point_id"])
            
            self.X_cj = self.x_0[j]
            self.Y_cj = self.x_0[j+1]
            self.Z_cj = self.x_0[j+2]
            self.w = self.x_0[j+3]
            self.o = self.x_0[j+4]
            self.k = self.x_0[j+5]

            #xp, yp, c values should be updated here if multiple cameras were used
            
            self.X_i = self.x_0[j_2]
            self.Y_i = self.x_0[j_2+1]
            self.Z_i = self.x_0[j_2+2]

            v = self.V()
            w = self.W()
            u = self.U()
            m_temp = self.M()
            
            x = self.xp-self.c*u/w
            y = self.yp - self.c*v/w
            self.rhc(x,y)
            
            #setup xij
            self.l_0[i,0] = self.x_ij
            
            #set up yij
            self.l_0[i+1,0] = self.y_ij
        
        
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