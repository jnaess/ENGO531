from numpy import transpose as t
from numpy import matrix as mat, matmul as mm
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
        
class Distance(LS):
    """
    """
    def __init__(self, df_name = "distances.txt"):
        """
        Desc:
            reads in the distance observations and preps that point of the LSA
        Input:
            df_name
            dimension_word = 
            dimension_symbol = "H", can be switched to "E", "N"
        Output:
            self.obs type: matrix: observation matrix
        """
        LS.__init__(self)
        
        self.df_name = df_name
        
        self.read_distance()
        self.set_obs()
        self.set_errors()
        
        self.set_design()
        self.obs_0()
        
    def read_distance(self):
        """
        Desc:
            reads in the distance stuff for a 2D
        Input:
        Output:
            self.d_word for the observations
            self.d_error for the stddev column
            self.df of all info [From To Distance[m] StDev[m]]
        """
        self.d_word = "Distance[m]"
        self.d_error = "StDev[m]"
        self.d_symbol = "E"
        self.df = pd.read_csv(self.df_name, sep = ' ')
        
        
    def set_obs(self):
        """
        Desc:
            sets up the observation matrix from the distance observations.
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
        
        #making these variables always accessible so that no errors arrise (don't worry they aren't assigned if a partial derivative is meant for it)
        e_from = 0
        n_from = 0
        
        e_to = 0
        n_to = 0
        
        #set placeholder
        i = 0
        
        while(i < self.n):
            #this is for figuring out if its a known or unknown_____________________
            #the extra code here was incase approx values were needed to additionally populate the design matrices
            from_const = False
            to_const = False
            #find columns to place values
            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])                
            if from_col == -1:
                #honestly this is only if we were to do something with the partial of the datums but 
                #we don't so ignore this and the next similar if statement, other than setting the True value is important
                from_col = self.find_col(self.d_symbol, froms[i], li = "datums")
                #then it is a datum
                #sets easting and northing values from the froms
                e_from = self.c[from_col,0]
                n_from = self.c[from_col+1,0]
                from_const = True
            else:
                #sets easting and northing values from the froms
                e_from = LS.x_0[from_col,0]
                n_from = LS.x_0[from_col+1,0]
            
            to_col = self.find_col(self.d_symbol, tos[i])
            if to_col == -1:
                to_col = self.find_col(self.d_symbol, tos[i], li = "datums")
                #then it is a datum
                #to_num = self.c[to_col,0]
                e_to = self.c[to_col,0]
                n_to = self.c[to_col+1,0]
                to_const = True
            else:
                #set the easting and northing values of the 
                e_to = LS.x_0[to_col,0]
                n_to = LS.x_0[to_col+1,0]
                                                             
            #__________________________________________________________________________
            
            if self.debugging:
                print("E_from: "+str(e_from)+" N_from:"+str(n_from))
                print("E_to: "+str(e_to)+" N_to: "+str(n_to))
            
            
            dist = m.sqrt((e_from-e_to)**2+(n_from-n_to)**2)
            
            #this is where the values are assigned______________________________________
            #it is important that files are formatted as X and then Y so that we can find the X column 
            #and autopopulate the Y column next to it
            if not from_const:
                self.A[i,from_col] = (e_from - e_to)/dist
                self.A[i,from_col+1] = (n_from - n_to)/dist
                
            if not to_const:
                #self.A[i,to_col] = delta + 1
                self.A[i,to_col] = -(e_from - e_to)/dist
                self.A[i,to_col+1] = -(n_from - n_to)/dist
            #_____________________________________________________________________________
            
            
            
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
        desc:
            Sets up self.l_0 (extimated observations)
            Used for finding the current misclosure
        """
                #set it up as just zeros
        self.l_0 = mat(np.zeros((self.n, 1)))
        
        #get the from and tos ready to be accessed
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        
        #making these variables always accessible so that no errors arrise (don't worry they aren't assigned if a partial derivative is meant for it)
        e_from = 0
        n_from = 0
        
        e_to = 0
        n_to = 0
        
        #set placeholder
        i = 0
        
        while(i < self.n):
            #this is for figuring out if its a known or unknown_____________________
            #the extra code here was incase approx values were needed to additionally populate the design matrices
            from_const = False
            to_const = False
            #find columns to place values
            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])                
            if from_col == -1:
                #honestly this is only if we were to do something with the partial of the datums but 
                #we don't so ignore this and the next similar if statement, other than setting the True value is important
                from_col = self.find_col(self.d_symbol, froms[i], li = "datums")
                #then it is a datum
                #sets easting and northing values from the froms
                e_from = self.c[from_col,0]
                n_from = self.c[from_col+1,0]
                from_const = True
            else:
                #sets easting and northing values from the froms
                e_from = LS.x_0[from_col,0]
                n_from = LS.x_0[from_col+1,0]
            
            to_col = self.find_col(self.d_symbol, tos[i])
            if to_col == -1:
                to_col = self.find_col(self.d_symbol, tos[i], li = "datums")
                #then it is a datum
                #to_num = self.c[to_col,0]
                e_to = self.c[to_col,0]
                n_to = self.c[to_col+1,0]
                to_const = True
            else:
                #set the easting and northing values of the 
                e_to = LS.x_0[to_col,0]
                n_to = LS.x_0[to_col+1,0]
 #__________________________________________________________________________    
            
            dist = m.sqrt((e_from-e_to)**2+(n_from-n_to)**2)
            
            #this is where the values areassigned______________________________________                
                #self.A[i,to_col] = delta + 1
            self.l_0[i,0] = dist
            #_____________________________________________________________________________

            i = i + 1