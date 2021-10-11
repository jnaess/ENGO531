from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from Bundle import Bundle
from Design_e import Design_e as ae
from LeastSquares import LS

class Design_o(Bundle, LS):
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
        LS.__init__(self)
        
        
        
        #from LS class to find unknown columns
        self.set_col_list_ao()
        
        self.set_X_0()
        
        self.set_obs()
        
        self.Ae = ae()
        
        self.set_design()
        
        
        
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
            self.x_0
            and
            LS.x_0
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
        LS.x_0 = self.x_0
        
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
        self.xp = self.pix_to_m*self.int["xp"][0]
        self.yp = self.pix_to_m*self.int["yp"][0]
        self.c = self.pix_to_m*self.int["c"][0]
        
        #set it up as just zeros
        self.l_0 = mat(np.zeros((self.n, 1)))
        
        for i in range(0, self.n, 2):
            obs = self.pho.iloc[int(i/2)]
            
            #row for ae parameters
            j = self.Ae.find_col_ae(obs["image_id"])
            #row for ue parameters
            j_2 = self.ue + self.find_col_ao(obs["point_id"])
            
            self.X_cj = LS.x_0[j]
            self.Y_cj = LS.x_0[j+1]
            self.Z_cj = LS.x_0[j+2]
            self.w = LS.x_0[j+3]
            self.o = LS.x_0[j+4]
            self.k = LS.x_0[j+5]

            #xp, yp, c values should be updated here if multiple cameras were used
            
            self.X_i = LS.x_0[j_2]
            self.Y_i = LS.x_0[j_2+1]
            self.Z_i = LS.x_0[j_2+2]

            v = self.V()
            w = self.W()
            u = self.U()
            m_temp = self.M()
            
            x = self.xp - self.c*u/w
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
        self.update_Ae()
        
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
            
    def update_Ae(self):
        """
        Desc:
            Initializes the design matrix
        Output:
        Input:
        """    
        self.xp = self.pix_to_m*self.int["xp"][0]
        self.yp = self.pix_to_m*self.int["yp"][0]
        self.c = self.pix_to_m*self.int["c"][0]
        
        #set it up as just zeros
        self.Ae.A = mat(np.zeros((self.n, self.ue)))
        
        #0, 2, 4, etc. are X pixels
        #1, 3, 5, etc. are Y pixels
        #__print("n: "+str(self.n))
        for i in range(0, self.n, 2):
            obs = self.pho.iloc[int(i/2)]
            
            #row for ae parameters
            j = self.Ae.find_col_ae(obs["image_id"])
            #row for ue parameters
            j_2 = self.ue + self.find_col_ao(obs["point_id"])
            
            self.X_cj = LS.x_0[j]
            self.Y_cj = LS.x_0[j+1]
            self.Z_cj = LS.x_0[j+2]
            self.w = LS.x_0[j+3]
            self.o = LS.x_0[j+4]
            self.k = LS.x_0[j+5]

            #xp, yp, c values should be updated here if multiple cameras were used
            
            self.X_i = LS.x_0[j_2]
            self.Y_i = LS.x_0[j_2+1]
            self.Z_i = LS.x_0[j_2+2]

            v = self.V()
            w = self.W()
            u = self.U()
            m_temp = self.M()


                #then evens (X partial)
                #X
            self.Ae.A[i,j] = -(self.c/w**2)*(m_temp[2,0]*u-m_temp[0,0]*w)
                #Y
            self.Ae.A[i,j + 1] = -self.c/w**2*(m_temp[2,1]*u-m_temp[0,1]*w)
                #Z
            self.Ae.A[i,j + 2] = -self.c/w**2*(m_temp[2,2]*u-m_temp[0,2]*w)
                #w
            self.Ae.A[i,j + 3] = -self.c/w**2*((self.Y_i - self.Y_cj)*(u*m_temp[2,2]-w*m_temp[0,2])
                                                       -(self.Z_i - self.Z_cj)*(u*m_temp[2,1]-w*m_temp[0,1]))
                #o
            self.Ae.A[i,j + 4] = self.c/w**2*((self.X_i - self.X_cj)*(-w*m.sin(self.o)*m.cos(self.k)-u*m.cos(self.o))
                                                      +(self.Y_i - self.Y_cj)*(w*m.sin(self.w)*m.cos(self.o)*m.cos(self.k)-u*m.sin(self.w)*m.sin(self.o))
                                                      +(self.Z_i - self.Z_cj)*(-w*m.cos(self.w)*m.cos(self.o)*m.cos(self.k)+u*m.cos(self.w)*m.sin(self.o)))
                #k
            self.Ae.A[i,j + 5] = -self.c*v/w
            
                #then odds (Y partial)
                #X
            self.Ae.A[i+1,j] = -self.c/w**2*(m_temp[2,0]*v-m_temp[1,0]*w)
                #Y
            self.Ae.A[i+1,j + 1] = -self.c/w**2*(m_temp[2,1]*v-m_temp[1,1]*w)
                #Z
            self.Ae.A[i+1,j + 2] = -self.c/w**2*(m_temp[2,2]*v-m_temp[0,2]*w)
                #w
            self.Ae.A[i+1,j + 3] = -self.c/w**2*((self.Y_i - self.Y_cj)*(v*m_temp[2,2]-w*m_temp[1,2])
                                                        -(self.Z_i - self.Z_cj)*(v*m_temp[2,1]-w*m_temp[1,1]))
                #o
            self.Ae.A[i+1,j + 4] = self.c/w**2*((self.X_i - self.X_cj)*(w*m.sin(self.o)*m.sin(self.k)-v*m.cos(self.o))
                                                      +(self.Y_i - self.Y_cj)*(-w*m.sin(self.w)*m.cos(self.o)*m.sin(self.k)-v*m.sin(self.w)*m.sin(self.o))
                                                      +(self.Z_i - self.Z_cj)*(w*m.cos(self.w)*m.cos(self.o)*m.sin(self.k)+v*m.cos(self.w)*m.sin(self.o)))
                #k
            self.Ae.A[i+1,j + 5] = -self.c*u/w