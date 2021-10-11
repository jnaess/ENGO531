from numpy import transpose as t
from numpy import matrix as mat, matmul as mm
import math as m
import numpy as np
import pandas as pd
from Tools import Tools

class LS(Tools):
    """
        Holds the universal values needed to integrate the different LS adjustments into one
    """
    x_0 = []
    def __init__(self, file_name = "coords.txt", debugging = False):
        """
        Desc:
            reads in the list of knowns and unknowns and assigns their values. Will construct design matrix, etc. based off of these
        Input:
            file_name where the knowns and unknowns are defined
            debugging, T/F. If true then more printing of stuff happens
        Output:
            sets up u_list (predefined in here)
            sets up number of unknowns (self.u)
            
        """
        #brings in the tool files for use
        Tools.__init__(self)
        
        self.debugging = debugging
        self.file_name = file_name
        self.read_2D()

    def read_2D(self):
        """
        Desc:
            reads in the 2D set of points and assigns values
            expects format of [name easting northing known/unknown]
            more specifically: [Point X[m] Y[m] Known[n]/Unknown[u]]
        Input:
            self.file_name
        Output:
            self.u_list (string list of unknown)
            self.x_0 (initial guesses of unknowns)
            self.c (constant values of knowns)
            self.datums (string list of knowns)
            self.u # of unknowns
        """
        df = pd.read_csv(self.file_name, sep = ' ')
        #currently only formatted for 2D

        self.u_list = []
        LS.x_0 = []
        self.c = []
        #pretty sure datums aren't actually used
        self.datums = []

        #assign values
        for index, row in df.iterrows():
            #check if known or unknown
            if row[3] == "u":
                #unknown name
                self.u_list.append(row[0]+"_E")
                self.u_list.append(row[0]+"_N")

                #add unknown values in order of x, y
                LS.x_0.append(row[1])
                LS.x_0.append(row[2])
            else: #then they are "n" --> knowns
                #known name
                self.datums.append(row[0]+"_E")
                self.datums.append(row[0]+"_N")

                #add known values in order of x, y
                self.c.append(row[1])
                self.c.append(row[2])
                
        LS.x_0 = t(mat(LS.x_0))
        self.c = t(mat(self.c))
        self.u = len(self.u_list)
        
    def find_col(self, dimension, point_name, li = "u"):
        """
        Desc:
            returns the column index of the desired points
            expects 'n' for known and 'u' for unknown
            **all values must be in caps**
        Input:
            u_list, list of strings of "pointname_dimension"
            dimension, string either "N", "E", "H"
        Output:
            integer value of the column to place the value 
            in the desired design matrix     
        """
        if li == "u" :
            li = self.u_list
        else:
            li = self.datums
            
        index = 0
        for key in li:
            #split the key into point name and dimension
            temp_name = key.split('_')[0]
            temp_dimension = key.split('_')[1]
            if (point_name == temp_name and dimension == temp_dimension):
                return index
            else:
                index = index + 1
        
        #debugging stuff
        if self.debugging:
            print(point_name + " Could not be found")
        return -1
    
    def set_col_list_ae(self):
        """
        Desc:
            
            Initializes the order of image_id's for the Ae matrix so that numbers are positioned correctly
        Input:
        Output:
            self.u_list_ae for Ae
        """
        #assumes images already sorted in ascending order
        self.u_list_ae = self.pho['image_id'].unique()
        
    def find_col_ae(self, image_id, li = "u"):
        """
        Desc:
            returns the column index of the desired points
            expects 'n' for known and 'u' for unknown
            **all values must be in caps**
        Input:
            u_list_ae, list of strings of "pointname_dimension"
            image_id: string of the image id index to return
        Output:
            integer value of the column to place the value in the desired design matrix multiplied by 6   
        """
        
        if li == "u" :
            li = self.u_list_ae
        else:
            li = self.datums
            
        index = 0
        for key in li:
            if image_id == key:
                return index*6
            else:
                index = index + 1
        
        #debugging stuff
        if self.debugging:
            print(point_name + " Could not be found")
        return -1