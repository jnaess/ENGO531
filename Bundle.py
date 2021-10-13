from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
from FileReader import File_Reader

class Bundle(LS, File_Reader):
    """
    Desc:
        Contains the LS for all LSA info
        Contains the Bundle for all Bundle Adjustment specific specs
    """
    
    def __init__(self):
        """
        Desc:
        Input:
        Output:
        """
        LS.__init__(self)
        File_Reader.__init__(self)
        
        self.initialize_variables()
        
    def initialize_variables(self):
        """
        Desc:
            initializes import dimensions as taken in from the File_Reader
        Input:
        Output:
            self.ue
            self.uo
            self.n
        """
        #pixel spacing (mm)
        self.pix_to_m = 3.45e-3
        
        #pixel spacing (mm)
        self.delta_x = 3.45e-6*1000
        self.delta_y = 3.45e-6*1000

        #normal principal distance (mm) 
        self.n_p_d = 7

        #number of pixels for total columns
        self.Np = 3000

        #number of rows of pixels
        self.Mp = 4000
        
        self.set_ue()
        self.set_uo()
        self.set_n()
        
    def set_ue(self):
        """
        Desc:
            finds m from # of images and then makes ue = 6 * m
        Input:
            self.ext
        Output:
            self.ue
        """
        m = len(self.ext.index)
        
        self.ue = 6 * m
        
    def set_uo(self):
        """
        Desc:
            finds p from # of points (currently just tie) and then makes uo = 2 * p
        Input:
            maybe self.con??
            self.tie
        Output:
            self.uo
        """
        #control stuff added
        q = len(self.obj.index)
        
        self.uo = 3 * q
        
    def set_n(self):
        """
        Desc:
            finds n from total number of pixel observations
        Input:
            self.pho
        Output:
            self.n
        """
        p = len(self.pho.index)
        
        self.n = 2 * p
            
    def rhc(self, n_ij = 2015.203, m_ij = 1566.904):
        """
        Desc:
            converts from LHC to RHC
            Must be formatted to assign or return the x, y coordinates as desired
        Input:
            n_ij (number of columns for that pixel)
            m_ij (number of rows for that pixel)
            
        Out:
            self.x_ij
            self.y_ij
        """
        #self.x_ij = (n_ij-((self.Np/2)-.5))*self.delta_x
        #self.y_ij = (((self.Mp/2)-.5)-m_ij)*self.delta_y
        
        self.x_ij = (n_ij-((self.Np/2)-.5))*self.delta_x
        self.y_ij = (((self.Mp/2)-.5)-m_ij)*self.delta_y
    
    def M(self):
        """
        Desc:
            Generates the M rotation matrix (3x3)
            converts from LHC to RHC
            Must be formatted to assign or return the x, y coordinates as desired
        Input:
            w, in radians
            k, in radians
            o, in radians
        Out:
            none atm
        """
        o = self.o
        k = self.k
        w = self.w
        
        temp = mat(np.zeros((3,3)))
        #row zero
        temp[0,0] = m.cos(o)*m.cos(k)
        temp[0,1] = m.cos(w)*m.sin(k)+m.sin(w)*m.sin(o)*m.cos(k)
        temp[0,2] = m.sin(w)*m.sin(k)-m.cos(w)*m.sin(o)*m.cos(k)

        #row one
        temp[1,0] = -m.cos(o)*m.sin(k)
        temp[1,1] = m.cos(w)*m.cos(k)-m.sin(w)*m.sin(o)*m.sin(k)
        temp[1,2] = m.sin(w)*m.cos(k)+m.cos(w)*m.sin(o)*m.sin(k)

        #row two
        temp[2,0] = m.sin(o)
        temp[2,1] = -m.sin(w)*m.cos(o)
        temp[2,2] = m.cos(w)*m.cos(o)

        #testing using matrix multiplication instead
        #temp = mm(self.R3(k), mm(self.R2(o), self.R1(w)))
        
        #for future reference
        #w = m.atan(-temp[2,1]/temp[2,2])
        #o = m.asin(temp[2,0])
        #k = m.atan(-temp[2,1]/temp[0,0])

        return temp
    
    def U(self):
        """
        Desc:
        *******test values are for angles ATM************
            uses the angle values and input XYZ values to output U
        Input:
            w, in radians
            k, in radians
            o, in radians
            X_i, 
            self.X_cj, 
            Y_i, 
            self.Y_cj, 
            self.Z_i, 
            self.Z_cj
        Out:
            none atm
        """
        U = self.M()[0,0]*(self.X_i-self.X_cj)+self.M()[0,1]*(self.Y_i-self.Y_cj)+self.M()[0,2]*(self.Z_i-self.Z_cj)

        return U

    def W(self):
        """
        Desc:
        *******test values are for angles ATM************
            uses the angle values and input XYZ values to output W
        Input:
            w, in radians
            k, in radians
            o, in radians
            self.X_i, 
            self.X_cj, 
            self.Y_i, 
            self.Y_cj, 
            self.Z_i, 
            self.Z_cj
        Out:
            none atm
        """
        W = self.M()[2,0]*(self.X_i-self.X_cj)+self.M()[2,1]*(self.Y_i-self.Y_cj)+self.M()[2,2]*(self.Z_i-self.Z_cj)
        
        return W

    def V(self):
        """
        Desc:
        *******test values are for angles ATM************
            uses the angle values and input XYZ values to output W
        Input:
            w, in radians
            k, in radians
            o, in radians
            self.X_i, 
            self.X_cj, 
            self.Y_i, 
            self.Y_cj, 
            self.Z_i, 
            self.Z_cj
        Out:
            none atm
        """
        V = self.M()[1,0]*(self.X_i-self.X_cj)+self.M()[1,1]*(self.Y_i-self.Y_cj)+self.M()[1,2]*(self.Z_i-self.Z_cj)

        return V
             
    def R1(self, o):
        """
        Desc:
            Returns R1 matrix
        Input:
            radians o
        Output:
            R1 (3x3)
        """
        temp = mat(np.zeros((3,3)))
        #row zero
        temp[0,0] = 1
        temp[0,1] = 0
        temp[0,2] = 0

        #row one
        temp[1,0] = 0
        temp[1,1] = m.cos(o)
        temp[1,2] = m.sin(o)

        #row two
        temp[2,0] = 0
        temp[2,1] = -m.sin(o)
        temp[2,2] = m.cos(o)
        
        return temp
        
    def R2(self, o):
        """
        Desc:
            Returns R2 matrix
        Input:
            radians o
        Output:
            R2 (3x3)
        """
        temp = mat(np.zeros((3,3)))
        #row zero
        temp[0,0] = m.cos(o)
        temp[0,1] = 0
        temp[0,2] = -m.sin(o)

        #row one
        temp[1,0] = 0
        temp[1,1] = 1
        temp[1,2] = 0

        #row two
        temp[2,0] = m.sin(o)
        temp[2,1] = 0
        temp[2,2] = m.cos(o)
        
        return temp
    
    def R3(self, o):
        """
        Desc:
            Returns R3 matrix
        Input:
            radians o
        Output:
            R3 (3x3)
        """
        temp = mat(np.zeros((3,3)))
        #row zero
        temp[0,0] = -m.cos(o)
        temp[0,1] = m.sin(o)
        temp[0,2] = 0

        #row one
        temp[1,0] = -m.sin(o)
        temp[1,1] = m.cos(o)
        temp[1,2] = 0

        #row two
        temp[2,0] = 0
        temp[2,1] = 0
        temp[2,2] = 1
        
        return temp