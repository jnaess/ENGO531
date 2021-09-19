import numpy as np
import pandas as pd

class File_Reader():
    """
    Contains a bunch of file reading functions so that the class may be imported when desired files what to be read in
    """
    
    def __init__(self, tie_file = 'engo531_lab1.tie',
                       ext_file = 'engo531_lab1.ext',
                        int_file = 'engo531_lab1.int',
                        pho_file = "engo531_lab1.pho"
                ):
        """
        Desc:
            does not have any need to setup anything. More of just a function container
        In:
        Out:
        """
        self.tie_file = tie_file
        self.ext_file = ext_file
        self.int_file = int_file
        self.pho_file = pho_file
        
    def read_tie(self):
        """
        Desc:
            Reads in the tie points as returns dataframe of the values
        In:
            filename, default set to lab1 filename
        Out:
            dataframe with columns "X, Y, Z" and index set to point_id
        """
        df = pd.read_csv(self.tie_file, sep = "\t", header = None)
        df.columns = ["point_id", "X", "Y", "Z"] 
        df = df.set_index("point_id")

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

        return df

    def read_pho(self):
        """
        Desc:
            Reads in the pho (observation) points as returns dataframe of the values
        In:
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

        return df