from numpy import matrix as mat, matmul as mm
from numpy import transpose as t
import math as m
import numpy as np
import pandas as pd
from Bundle import Bundle

class Design_e(Bundle):
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
        
        #from LS class to find unknown columns
        self.set_col_list_ae()
        
        self.set_design()
                
        
    def set_design(self):
        """
        Desc:
            Initializes the design matrix
        Output:
        Input:
        """
        #set it up as just zeros
        self.A = mat(np.zeros((self.n, self.ue)))
        
        #0, 2, 4, etc. are X pixels
        #1, 3, 5, etc. are Y pixels
        #__print("n: "+str(self.n))
        for i in range(0, self.n, 2):
            #increments every two because one row is for X, one row is for Y
            
            #print(i)
            #print(i/2)
            #each time we should go through one observation
            #indexes every 2
            #this is the observation
            #get image id from photo obs
            obs = self.pho.iloc[int(i/2)]
            
            #get image row from ext EOP's
            #j = int(obs["image_id"])
            j = self.find_col_ae(obs["image_id"])
            #print(j)
                 
            #from EOP (ext)
            #pretty sure should always be out best estimate
            #print("eop index: " + str(obs["image_id"]))
            
            eop = self.ext.loc[(self.ext["image_id"] == obs["image_id"]) | (self.ext["image_id"] == str(obs["image_id"]))]
            #print(type(eop))
            #print(eop)
            
            #assign values
            """
            Values connected to Bundle?
                yes
            """
            self.w = m.radians(eop['w'].to_list()[0])
            self.o = m.radians(eop["o"].to_list()[0])
            self.k = m.radians(eop["k"].to_list()[0])
            self.X_cj = eop["Xc"].to_list()[0]
            self.Y_cj = eop["Yc"].to_list()[0]
            self.Z_cj =eop["Zc"].to_list()[0]
            
            #test print values
            #print("self.w "+ str(self.w))
            #print("self.o "+ str(self.o))
            #print("self.k "+ str(self.k))
            #print("self.X_cj "+ str(self.X_cj))
            #print("self.Y_cj "+ str(self.Y_cj))
            #print("self.Z_cj "+ str(self.Z_cj))
            
            #from IOP (int)
            #print("iop index: " + str(eop["camera_id"]))
            #print(self.int)
            #print(self.int["camera_id"])
            #print(eop["camera_id"])
            #print(self.int["camera_id"])
            #print(str(eop["camera_id"]))
            #iop = self.int.loc[(self.int["camera_id"] == eop["camera_id"])]# | (self.int["camera_id"] == str(eop["camera_id"]))]
            #print(iop)
            #***********We only have one camera and therefore no search is needed. For multiple cameras this will need to be updated*************
            iop = self.int
            """
            Values connected to Bundle?
                no, just inside () here stuff
            """
            self.xp = 3.45e-3*iop["xp"].to_list()[0]
            self.yp = 3.45e-3*iop["yp"].to_list()[0]
            self.c = 3.45e-3*iop["c"].to_list()[0]
            #print("self.xp "+ str(self.xp))
            #print("self.yp "+ str(self.yp))
            #print("self.c "+ str(self.c))
            
            #from object points (either tie or control)
            #pretty sure should always be our best estimate
            #get point coords from obs
            #print("point id: "+str(obs["point_id"]))
            obj_pt = self.obj.loc[self.obj['point_id'] == str(obs["point_id"])]
            
            """
            Values connected to Bundle?
                yes
            """
            self.X_i = obj_pt["X"].to_list()[0]
            self.Y_i = obj_pt["Y"].to_list()[0]
            self.Z_i = obj_pt["Z"].to_list()[0]
            #print("self.X_i "+ str(self.X_i))
            #print("self.Y "+ str(self.Y_i))
            #print("self.Z "+ str(self.Z_i))
            
            #calculate important values
            v = self.V()
            w = self.W()
            u = self.U()
            m_temp = self.M()
            
            
            
            #print("v "+str(v))
            #print("w " +str(w))
            #print("u "+str(u))
            #print("m_temp"+str(m_temp))

                #then evens (X partial)
                #X
            self.A[i,j] = -(self.c/w**2)*(m_temp[2,0]*u-m_temp[0,0]*w)
                #Y
            self.A[i,j + 1] = -self.c/w**2*(m_temp[2,1]*u-m_temp[0,1]*w)
                #Z
            self.A[i,j + 2] = -self.c/w**2*(m_temp[2,2]*u-m_temp[0,2]*w)
                #w
            self.A[i,j + 3] = -self.c/w**2*((self.Y_i - self.Y_cj)*(u*m_temp[2,2]-w*m_temp[0,2])
                                                       -(self.Z_i - self.Z_cj)*(u*m_temp[2,1]-w*m_temp[0,1]))
                #o
            self.A[i,j + 4] = self.c/w**2*((self.X_i - self.X_cj)*(-w*m.sin(self.o)*m.cos(self.k)-u*m.cos(self.o))
                                                      +(self.Y_i - self.Y_cj)*(w*m.sin(self.w)*m.cos(self.o)*m.cos(self.k)-u*m.sin(self.w)*m.sin(self.o))
                                                      +(self.Z_i - self.Z_cj)*(-w*m.cos(self.w)*m.cos(self.o)*m.cos(self.k)+u*m.cos(self.w)*m.sin(self.o)))
                #k
            self.A[i,j + 5] = -self.c*v/w
            
                #then odds (Y partial)
                #X
            self.A[i+1,j] = -self.c/w**2*(m_temp[2,0]*v-m_temp[1,0]*w)
                #Y
            self.A[i+1,j + 1] = -self.c/w**2*(m_temp[2,1]*v-m_temp[1,1]*w)
                #Z
            self.A[i+1,j + 2] = -self.c/w**2*(m_temp[2,2]*v-m_temp[0,2]*w)
                #w
            self.A[i+1,j + 3] = -self.c/w**2*((self.Y_i - self.Y_cj)*(v*m_temp[2,2]-w*m_temp[1,2])
                                                        -(self.Z_i - self.Z_cj)*(v*m_temp[2,1]-w*m_temp[1,1]))
                #o
            self.A[i+1,j + 4] = self.c/w**2*((self.X_i - self.X_cj)*(w*m.sin(self.o)*m.sin(self.k)-v*m.cos(self.o))
                                                      +(self.Y_i - self.Y_cj)*(-w*m.sin(self.w)*m.cos(self.o)*m.sin(self.k)-v*m.sin(self.w)*m.sin(self.o))
                                                      +(self.Z_i - self.Z_cj)*(w*m.cos(self.w)*m.cos(self.o)*m.sin(self.k)+v*m.cos(self.w)*m.sin(self.o)))
                #k
            self.A[i+1,j + 5] = -self.c*u/w
            
