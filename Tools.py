import numpy as np
import matplotlib.pyplot as plt
import os

class Tools():
    """
    Desc:
        This class was made as a toolbox for plotting and converting values
    """
    def __init__(self):
        """
        Just exsists :-)
        """
        
    def plot_mat(self, matrix, title = "Title", round_to = 6):
        """
        Desc:
            Checks to see if a "Figures" folder has been made. If it is not made then it makes it. 
            Then saves the input matrix as a .png to the folder with "Title" as the name
        Input:
            matrix: the numpy array to plot
            title: the title of the array and output image (default "Title")
            round_to: decimals to round to (default 6)
        Output:
        """   
        #set up figure with decently sized boxes
        fig, ax = plt.subplots(figsize = (10,15))
        ax.imshow(matrix)

        plt.title(title)

        # Loop over data dimensions and create text annotations.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                #inputs numerical values
                text = ax.text(j, i, round(matrix[i, j],round_to),
                               ha="center", va="center", color="w")

        #plt.axis('off')

        #folder is just called figures
        folder_path = 'Figures/'
        file_name = title

        #makes folder if not already there
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        #saves to the folder using the title name
        fig.savefig(os.path.join(folder_path,file_name))
        
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        
    def save_df(self, df, title):
        """
        Desc:
            saves df to the file
        Input:
        Output:
        """
        #folder is just called files
        folder_path = 'Files/'
        file_name = title

        #makes folder if not already there
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        #saves to the folder using the title name
        df.to_csv(os.path.join(folder_path,file_name))