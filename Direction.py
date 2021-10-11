from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
        
class Horizontal_Direction(LS):
    """
    """
    def __init__(self, df_name = "Deformation.txt", dimension_word = "HA", dimension_symbol = "HA"):
        """
        Desc:
            takes in the dataframe of bearing from to VD, HD, HA, and respective errors
        Input:
            df_name
            dimension_word = "Height", can be switched to "Easting" or "Northing"
            dimension_symbol = "H", can be switched to "E", "N"
        Output:
            self.obs type: matrix: observation matrix
        """
        LS.__init__(self)
        
        self.d_word = dimension_word+" (Meter)"
        self.d_error = dimension_word+" Error (Meter)"
        self.d_symbol = dimension_symbol
        
        self.df = pd.read_csv(df_name, sep = '\t')
        self.set_obs()
        self.obs_0()
        self.set_errors()
        self.set_design()
        
    def set_obs(self):
        """
        Desc:
            sets up the observation matrix from the heights
        Input:
        Output:
            self.obs
            self.n, number of observations
        """
        #switch from 
        self.obs = mat(self.df[self.d_word]).transpose()
        self.n = len(self.df[self.d_word])
        
    def set_design(self):
        """
        Desc:
            initializes the design matrix with 0's, 1's and -1's
        Input:
        Output:
            self.A, type matrix
        """
        #set it up as just zeros
        self.A = mat(np.zeros((self.n, self.u)))
        
        #get the from and tos ready to be accessed
        angles = self.df["HA (Meter)"].to_list()
        bearings = self.df["Bearing"].to_list()
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        w = 0
        #set placeholder
        i = 0
        
        while(i < self.n):
            #update wi if a new set of measurements is used
            if i == 0 or froms[i] != froms[i-1] or bearings[i] != bearings[i-1]:
                #then calculate a new wi
                from_e = self.find_col("E", froms[i])
                from_n = self.find_col("N", froms[i])
                to_e = self.find_col("E", tos[i])
                to_n = self.find_col("N", tos[i])
                self.wi = m.atan((to_e-from_e)/(to_n-from_n))
                w = w + 1
                     
                           
            #constants for this line
            d_n = to_n-from_n
            d_e = to_e-from_e
            lij = m.sqrt(d_e**2+d_n**2)
            
            #update rij
            from_e = self.find_col("E", froms[i])
            from_n = self.find_col("N", froms[i])
            to_e = self.find_col("E", tos[i])
            to_n = self.find_col("N", tos[i])
            rij = m.atan(d_e/d_n) - self.wi
            
            # (S_xi)
            col = self.find_col("E", froms[i])
            #self.A[i,col] = rij - (d_n / lij**2)
            self.A[i,col] = - (d_n / lij**2)
            
            # (S_yi)
            col = self.find_col("N", froms[i])
            #self.A[i,col] = rij + (d_e / lij**2)
            self.A[i,col] =  (d_e / lij**2)
            
            # (S_xj)
            col = self.find_col("E", tos[i])
            #self.A[i,col] = rij +(d_n / lij**2)
            self.A[i,col] = (d_n / lij**2)
            
            # (S_xj)
            col = self.find_col("N", tos[i])
            #self.A[i,col] = rij - (d_e / lij**2)
            self.A[i,col] = -(d_e / lij**2)
            
            #last column is wi
            col = self.find_col("solo", str(w))
            self.A[i,-1] = - 1          
            
            i = i + 1
                    
    def set_errors(self):
        """
        Desc:
            sets up the errors in an n,1 matrix
        Input:
        Output:
            self.errs
        """
        self.errs = mat(self.df[self.d_error]).transpose()
        
    def omega(self, fro_e, fro_n, to_e, to_n):
        """
        Calculates omega from
        """
        fro = np.array([fro_e, fro_n])
        to = np.array([to_e, to_n])

        north = np.array([0,1])

        r_v = to - fro

        self.wi = np.dot(r_v / np.linalg.norm(r_v), north / np.linalg.norm(north))
        
    def obs_0(self):
        """
        Generates the approximate coordinates
        """
        #get the from and tos ready to be accessed
        angles = self.df["HA (Meter)"].to_list()
        bearings = self.df["Bearing"].to_list()
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        self.l_0 = []
        w = 0
        #set placeholder
        i = 0
        
        while(i < self.n):
            #update wi if a new set of measurements is used
            if i == 0 or froms[i] != froms[i-1] or bearings[i] != bearings[i-1]:
                #then calculate a new wi
                from_e = self.find_col("E", froms[i])
                from_n = self.find_col("N", froms[i])
                to_e = self.find_col("E", tos[i])
                to_n = self.find_col("N", tos[i])
                self.omega(from_e, from_n, to_e, to_n)
                w = w + 1
                     
                           
            #constants for this line
            d_n = to_n-from_n
            d_e = to_e-from_e
            lij = m.sqrt(d_e**2+d_n**2)
            
            #update rij
            from_e = self.find_col("E", froms[i])
            from_n = self.find_col("N", froms[i])
            to_e = self.find_col("E", tos[i])
            to_n = self.find_col("N", tos[i])
            rij = m.atan(d_e/d_n) - self.wi
                   
            self.l_0.append(rij)
            i = i + 1
            
        self.l_0 = t(mat(np.array(self.l_0)))
        
    def update_obs_0(self, x_0):
        """
        Desc:
            Updates the self.df to have better estimated values for unknowns
        Input:
        Output:
        """
        
        
        