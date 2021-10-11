import numpy as np
import pandas as pd

class File_Reader():
    """
    Contains a bunch of file reading functions so that the class may be imported when desired files what to be read in
    """
    
    def __init__(self, tie_file = 'engo531_lab1.tie',
                       ext_file = 'engo531_lab1.ext',
                        int_file = 'engo531_lab1.int',
                        pho_file = "engo531_lab1.pho",
                 con_file = "engo531_lab1.con"
                ):
        """
        Desc:
            does not have any need to setup anything. More of just a function container
            all id's are in strings
        In:
        Out:
            self.tie: DF of tie points 
            self.ext: data frame of exterior orientation parameters
            self.int: DF of interior orientation parameters
            self.pho: Dataframe of image (photo) point obs 
            self.con: Df of control points 
            self.obj: control and tie point dataframes
        """
        self.tie_file = tie_file
        self.ext_file = ext_file
        self.int_file = int_file
        self.pho_file = pho_file
        self.con_file = con_file
        
        self.con = self.read_con()
        self.tie = self.read_tie()
        self.ext = self.read_ext()
        self.int = self.read_int()
        self.pho = self.read_pho()
        self.obs_points()
        
    def read_tie(self):
        """
        Desc:
            Reads in the tie points as returns dataframe of the values
        In:
            filename, default set to lab1 filename
        Out:
            dataframe with columns "X, Y, Z" and index not set to point_id
        """
        df = pd.read_csv(self.tie_file, sep = "\t", header = None)
        df.columns = ["point_id", "X", "Y", "Z"] 
        #df = df.set_index("point_id")
        
        #convert all value columns to flaots
        df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].astype(float)
        
        #convert ID's to strings
        df[["point_id"]] = df[["point_id"]].astype(str)
        
        #std for tie points is 1 pixel
        
        return df
    
    def read_con(self):
        """
        Desc:
            Reads in the control points as returns dataframe of the values
        In:
            filename, default set to lab1 filename
        Out:
            dataframe with columns "X, Y, Z" and index not set to point_id
        """
        df = pd.read_csv(self.con_file, sep = "\t", header = None)
        
        #cleaning the data
        df = df.drop(4, axis=1)
        
        df.columns = ["point_id", "X", "Y", "Z"] 
        #df = df.set_index("point_id")
        
        #convert all value columns to flaots
        df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].applymap(np.float64)
        
        #convert ID's to strings
        df[["point_id"]] = df[["point_id"]].astype(str)
        
        return df

    def read_ext(self):
        """
        Desc:
            Reads in the tie points as returns dataframe of the values
        In:
            filename, default set to lab1 filename
        Out:
            dataframe with columns "image_id","camera_id","Xc", "Yc", "Zc", "w", "o", "k" and index set to natural incrementation
        """
        df = pd.read_csv(self.ext_file, sep = "\t", header = None)

        #cleaning the data
        df = df.drop([8,9,10,11,12,13,14], axis=1)

        df.columns = ["image_id","camera_id","Xc", "Yc", "Zc", "w", "o", "k"] 
        
        #convert all value columns to flaots
        #df = df[["Xc", "Yc", "Zc", "w", "o", "k"]].astype(float)
        df[["Xc", "Yc", "Zc", "w", "o", "k"]] = df[["Xc", "Yc", "Zc", "w", "o", "k"]].applymap(np.float64)
        
        #convert ID's to strings
        df[["camera_id"]] = df[["camera_id"]].astype(str)
        df[["image_id"]] = df[["image_id"]].astype(str)
        
        return df

    def read_int(self):
        """
        Desc:
            Reads in the tie points as returns dataframe of the values
            Currently only formatted for a single row. Multiple rows will need reformatting
        In:
            filename, default set to lab1 filename
        Out:
            dataframe with columns "camera_id", 'xp', 'xp', "c" and index set to natural incrementation
        """
        df = pd.read_csv(self.int_file, sep = "\t", header = None)

        #cleaning the data
        df = df.drop([0], axis=1)

        #break column 2 into the proper X, Y, Z string
        l = df.loc[0][2].split(" ")
        l.remove('')
        l = [x for x in l if x!='']
        corrected = [df.loc[0][1]] + l

        #recombine data again
        df = pd.DataFrame([corrected], columns = ["camera_id", 'xp', 'yp', "c"])
        
        #convert ID's to strings
        df[["camera_id"]] = df[["camera_id"]].astype(str)

        df[["c"]] = df[["c"]].astype(float)
        
        return df

    def read_pho(self):
        """
        Desc:
            Reads in the pho (observation) points as returns dataframe of the values
            Must have self.tie initialized
        In:
            self.tie
            filename, default set to lab1 filename
        Out:
            dataframe with columns "point_id", "image_id", "x", "y" and index set to natural incrementation
        """
        #uses mixed spacing to read in files... nbd ;-)
        df = pd.read_csv(self.pho_file, header = None, delim_whitespace =True)

        #assign column values
        df.columns = ["point_id", "image_id", "x", "y"] 
        
        #combines point_id and image_id for a unique identifier
        df["unique_id"] = df["point_id"].to_numpy()+df["image_id"].astype(str).to_numpy()

        #convert ID's to strings
        df[["point_id"]] = df[["point_id"]].astype(str)
        df[["image_id"]] = df[["image_id"]].astype(str)
        
        #sort values in ascending inage_id's
        df = df.sort_values(by=['image_id'])
        
        #std for control points is .01mm and temporarily 10mm for tie
        temp = []
        for index, row in df.iterrows():
           # print(row['point_id'])
            if any(self.tie["point_id"] == row['point_id']):
                temp.append("u")
            else:
                temp.append('n')
        df["knowns"] = temp
        
        return df
    
    def obs_points(self):
        """
        Desc:
            Initializes the object point dataframe
            ***may bee differentiating between tie points and control points***
        Input:
            self.tie
            self.con
        Output:
            self.obj
        """
        self.obj = pd.concat([self.tie, self.con])
        
        #convert ID's to strings
        self.obj[["point_id"]] = self.obj[["point_id"]].astype(str)