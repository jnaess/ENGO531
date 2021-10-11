from scipy import misc 
from scipy import stats
import pandas as pd
import numpy as np

class Tables():
    """
    Parent class to PostAdjustmentTester which generates the significant values to increase modularity
    """
    def __init__(self):
        """
        """
        
    def newtons_method(self, x, tolerance=0.0001):
        while True:
            x1 = x - self.f(x) / misc.derivative(self.f, x) 
            t = abs(x1 - x)
            if t < tolerance:
                break
            x = x1
        return x

    def f(self, x):
        return 1 - stats.chi2.cdf(x, self.r) - self.pvalue

    def x_2(self):
        """
        Reference:
            Code reformatted to return a single line of the desired x_2 value based on our DOF (instead of a given value)
            Code refers to functions "newtons_method", "f", "x_2"
            https://moonbooks.org/Articles/How-to-create-a-Chi-square-table-using-python-/
        Desc:
            returns a chi-square dataframe row for the designated DOF
        Input:
            r: defrees of freedom
        Output:
        """
        self.pvalueList = [0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005]
        results = []
        for i in range(self.r,self.r+1):
            self.r = i 
            Result = []
            for self.pvalue in self.pvalueList:
                x0 = self.r  # x0 approximation
                x = self.newtons_method(x0)
                Result.append(x)
            for i in range(10):
                Result[i] = round(Result[i],3)
            results.append(Result)
        return pd.DataFrame(results, columns = self.pvalueList)
        
    