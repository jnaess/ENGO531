from numpy import transpose as t
from numpy import matrix as mat, matmul as mm
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
        
class Delta(LS):
    """
    """
    def __init__(self, df_name = "Leveling.txt", dimension_word = "Height", dimension_symbol = "H"):
        """
        Desc:
            takes in the dataframe of from, to, heigh, height error
        Input:
            df_name
            dimension_word = "Height ", can be switched to "Easting" or "Northing"
            dimension_symbol = "H", can be switched to "E", "N"
        Output:
            self.obs type: matrix: observation matrix
        """
        LS.__init__(self)
        #test which known to use
        self.test = dimension_word
        
        self.d_word = dimension_word+" (Meter)"
        self.d_error = dimension_word+" Error (Meter)"
        self.d_symbol = dimension_symbol
        
        self.df = pd.read_csv(df_name, sep = '\t')
        self.set_obs()
        self.set_errors()
        self.set_design()
        self.obs_0()
        
    def set_obs(self):
        """
        Desc:
            sets up the observation matrix from the heights
        Input:
        Output:
            self.obs
            self.n, number of observations
        """
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
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        
        from_num = 0
        to_num = 0
        
        #set placeholder
        i = 0
        
        while(i < self.n):
            #the extra code here was incase approx values were needed to additionally populate the design matrices
            from_const = False
            to_const = False
            #find columns to place values
            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])                
            if from_col == -1:
                from_col = self.find_col(self.d_symbol, froms[i], li = "datums")
                #then it is a datum
                from_num = self.c[from_col,0]
                from_const = True
            else:
                from_num = self.x_0[from_col,0]
            
            to_col = self.find_col(self.d_symbol, tos[i])
            if to_col == -1:
                to_col = self.find_col(self.d_symbol, tos[i], li = "datums")
                #then it is a datum
                to_num = self.c[to_col,0]
                to_const = True
            else:
                to_num = self.x_0[to_col,0]
                
            #there is not yet an x_0 for each functional model, instead only in the overall network
            delta = to_num - from_num
            
            if not from_const:
                #self.A[i,from_col] = delta - 1
                self.A[i,from_col] = - 1
            if not to_const:
                #self.A[i,to_col] = delta + 1
                self.A[i,to_col] = 1
            
            
            
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
        
    def obs_0(self):
        """
        """
                #set it up as just zeros
        self.l_0 = mat(np.zeros((self.n, 1)))
        
        #get the from and tos ready to be accessed
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        
        from_num = 0
        to_num = 0
        
        #set placeholder
        i = 0
        
        while(i < self.n):

            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])
            
            if from_col == -1:
                if self.test == "Height" or self.test == "VD":
                    from_num = self.c[0,0]
                if self.test == "Easting":
                    from_num = self.c[1,0]
                if self.test == "Northing":
                    from_num = self.c[2,0]
            else:
                from_num = self.x_0[from_col,0]
            
            to_col = self.find_col(self.d_symbol, tos[i])
            if to_col == -1:
                if self.test == "Height" or self.test == "VD":
                    to_num = self.c[0,0]
                if self.test == "Easting":
                    to_num = self.c[1,0]
                if self.test == "Northing":
                    to_num = self.c[2,0]
            else:
                to_num = self.x_0[to_col,0]
                
            delta = to_num - from_num
            
            self.l_0 [i,0] = delta

            i = i + 1
        