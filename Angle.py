from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
        
class Angle(LS):
    """
    """
    def __init__(self, df_name = "angles.txt"):
        """
        Desc:
            reads in the anglular model data, expects format [From At To Degrees Minutes Seconds StDev[sec]]
        Input:
            df_name
            dimension_word = "Height", can be switched to "Easting" or "Northing"
            dimension_symbol = "H", can be switched to "E", "N"
        Output:
            self.obs type: matrix: observation matrix
        """
        LS.__init__(self)
        
        self.df_name = df_name
        self.read_angle()
        self.set_obs()
        self.set_errors()
        self.set_design()

        self.obs_0()
        
    def read_angle(self):
        """
        Desc:
            reads in the distance stuff for a 2D
        Input:
        Output:
            self.d_word for the observations radians
            self.d_error for the stddev column
            self.df of all info [From To Distance[m] StDev[m]]
        """
        self.df = pd.read_csv(self.df_name, sep = ' ')
        
        #Switch DMS to Radians____________________
        self.df = pd.read_csv('angles.txt', sep = ' ')
        deg = np.array(self.df["Degrees"].to_list())
        mins = np.array(self.df["Minutes"].to_list())
        sec = np.array(self.df["Seconds"].to_list())

        degree = deg + mins/60 + sec/3600
        radians = np.radians(degree)
        #drop DMS
        self.df = self.df.drop(columns = ["Degrees", "Minutes", "Seconds"])
        #add degrees
        self.df["Radians"] = radians
        #_________________________________________
        
        self.d_word = "Radians"
        
        #Switch second error to radian error _______________
        self.df["StDev[rad]"] = np.radians(np.array(self.df["StDev[sec]"])/3600)
        #self.df = self.df.drop(columns = ["StDev[sec]"])
        #_____________________________________________________
        
        #now formatted [From At To Radians StDev[rad]]
        self.d_error = "StDev[sec]"
        self.d_symbol = "E"
        
        
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
        angles = self.df[self.d_word].to_list()
        ats = self.df["At"].to_list()
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        
        #print(ats)
        #print(froms)
        #print(tos)
        w = 0
        #set placeholder
        i = 0
        
        e_from = 0
        n_from = 0
        
        e_to = 0
        n_to = 0
        
        e_at = 0
        n_at = 0
        
        while(i < self.n):                     
            #this is for figuring out if its a known or unknown_____________________
            #the extra code here was incase approx values were needed to additionally populate the design matrices
            from_const = False
            to_const = False
            at_const = False
            #find columns to place values
            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])                
            if from_col == -1:
                #honestly this is only if we were to do something with the partial of the datums but 
                #we don't so ignore this and the next similar if statement, other than setting the True value is important
                from_col = self.find_col(self.d_symbol, froms[i], li = "datums")
                #then it is a datum
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
                e_to = self.c[to_col,0]
                n_to = self.c[to_col+1,0]
                to_const = True
            else:
                #set the easting and northing values of the 
                e_to = LS.x_0[to_col,0]
                n_to = LS.x_0[to_col+1,0]
                
            at_col = self.find_col(self.d_symbol, ats[i])
            if at_col == -1:
                at_col = self.find_col(self.d_symbol, ats[i], li = "datums")
                #then it is a datum
                e_at = self.c[at_col,0]
                n_at = self.c[at_col+1,0]
                at_const = True
            else:
                #set the easting and northing values of the 
                e_at = LS.x_0[at_col,0]
                n_at = LS.x_0[at_col+1,0]
                                                             
            #__________________________________________________________________________               
            #constants for this line    
            #print("======================")
            
            #print(e_from)
            #print(n_from)
            #print(e_at)
            #print(n_at)
            #print(e_to)
            #print(n_to)           
            #print("======================")
            
            #this is where the values are assigned______________________________________
            #it is important that files are formatted as X and then Y so that we can find the X column 
            #and autopopulate the Y column next to it
            if not at_const:
                #self.A[i,to_col] = delta + 1
                
                #(1)
                self.A[i,at_col] = (((n_to - n_at)/((e_to-e_at)**2+(n_to-n_at)**2))-((n_from-n_at)/((e_from-e_at)**2+(n_from-n_at)**2)))
                
                #(2)
                self.A[i,at_col+1] = ((-(e_to - e_at)/((e_to-e_at)**2+(n_to-n_at)**2))+((e_from-e_at)/((e_from-e_at)**2+(n_from-n_at)**2)))
                
            if not from_const:
                
                #(3)
                self.A[i,from_col] = ((n_from-n_at)/((e_from-e_at)**2+(n_from-n_at)**2))
                
                #(4)
                self.A[i,from_col+1] = (-(e_from-e_at)/((e_from-e_at)**2+(n_from-n_at)**2))         
                
            if not to_const:
                #self.A[i,to_col] = delta + 1
                
                #(5)
                self.A[i,to_col] = (-(n_to-n_at)/((e_to-e_at)**2+(n_to-n_at)**2))
                
                #(6)
                self.A[i,to_col+1] = ((e_to-e_at)/((e_to-e_at)**2+(n_to-n_at)**2))
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
        #converts from arc seconds to radians
        self.errs = np.radians(mat(self.df[self.d_error]).transpose()/3600)
        
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
        desc:
            Sets up self.l_0 (extimated observations)
            Used for finding the current misclosure
        """
        self.l_0 = mat(np.zeros((self.n, 1)))        
        
        #get the from and tos ready to be accessed
        angles = self.df[self.d_word].to_list()
        ats = self.df["At"].to_list()
        froms = self.df["From"].to_list()
        tos = self.df["To"].to_list()
        
        #print(ats)
        #print(froms)
        #print(tos)
        w = 0
        #set placeholder
        i = 0
        
        e_from = 0
        n_from = 0
        
        e_to = 0
        n_to = 0
        
        e_at = 0
        n_at = 0
        
        while(i < self.n):                     
            #this is for figuring out if its a known or unknown_____________________
            #the extra code here was incase approx values were needed to additionally populate the design matrices
            from_const = False
            to_const = False
            at_const = False
            #find columns to place values
            #picks the dimension symbol to search for
            from_col = self.find_col(self.d_symbol, froms[i])                
            if from_col == -1:
                #honestly this is only if we were to do something with the partial of the datums but 
                #we don't so ignore this and the next similar if statement, other than setting the True value is important
                from_col = self.find_col(self.d_symbol, froms[i], li = "datums")
                #then it is a datum
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
                e_to = self.c[to_col,0]
                n_to = self.c[to_col+1,0]
                to_const = True
            else:
                #set the easting and northing values of the 
                e_to = LS.x_0[to_col,0]
                n_to = LS.x_0[to_col+1,0]
                
            at_col = self.find_col(self.d_symbol, ats[i])
            if at_col == -1:
                at_col = self.find_col(self.d_symbol, ats[i], li = "datums")
                #then it is a datum
                e_at = self.c[at_col,0]
                n_at = self.c[at_col+1,0]
                at_const = True
            else:
                #set the easting and northing values of the 
                e_at = LS.x_0[at_col,0]
                n_at = LS.x_0[at_col+1,0]
                                                             
            #__________________________________________________________________________               
            #constants for this line    
            #print("======================")
            
            #print(e_from)
            #print(n_from)
            #print(e_at)
            #print(n_at)
            #print(e_to)
            #print(n_to)           
            #print("======================")
            
            #this is where the values are assigned______________________________________
            #it is important that files are formatted as X and then Y so that we can find the X column 
            #and autopopulate the Y column next to it
                        #this is where the values are generated______________________________________
            #it is important that files are formatted as X and then Y so that we can find the X column 
            #and autopopulate the Y column next to it
            
            
            ang = m.atan((n_to-n_at)/(e_to-e_at)) - m.atan((n_from-n_at)/(e_from-e_at))
            
            if ang < 0:
                ang = ang + m.pi
            #this is where the values are assigned______________________________________                
                #self.A[i,to_col] = delta + 1
            self.l_0[i,0] = ang
            #_____________________________________________________________________________
            i = i + 1
        