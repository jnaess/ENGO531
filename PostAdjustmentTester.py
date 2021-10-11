from numpy import transpose as t
from numpy import matrix as mat, matmul as mm
import matplotlib as plt

from numpy import linalg as lin
from numpy.linalg import inv
import math as m
import numpy as np
import pandas as pd
from LeastSquares import LS
from Level import Delta
from Tables import Tables

from scipy import stats as st
from scipy.stats import t as stu
from scipy.stats import chi2



class PostAdjustmentTester(Tables):
    """
    Desc:
        Assumes that the LSA has been conducted and outputs results for post adjustment tests
    """
    def __init__(self):
        """
        Desc:
            Figuring out if we need to take in matrices or if we'll just inherit the class and assume that they're build
        """
        Tables.__init__(self)
        
    def global_a_posteriori(self, alpha = .05):
        """
        Desc:
            Tests the statistical sifnigicance of the aposteriori to a priori variance factor
        Input:
            alpha: to generate the two confidence intervals. Be sure to make sure that these values are generated in the respective dataframe of values, otherwise they won't be found :-)
            self.u: # of unknowns
            self.n: # of observations
            self.a_post: final computed a posteriori variance factor
            self.apriori: initial apriori variance factor
        Output:
            Prints the output and respective indication
        """
        #set up DOF (r)
        self.r = self.n - self.u
        
        #retrieves dataframe of chi values for our respective DOF
        ch_df = self.x_2()
        
        low = ch_df[alpha][0]
        high = ch_df[1-alpha][0]
        
        y = (self.r * self.a_post**2)/self.apriori**2
        
        #if fails this check then there is an indication that the residuals or math model may be off
        
        
        print("{} tested with chi_square boundries of {} and {}".format(y, low, high))
        if y > low and y < high:
            print("Global A-Posteriori Variance Factor Test passes at a {} confidence level".format((1 - alpha)*100))
            print("There is no indication for errors within residual or the math models")
        else:
            print("Global A-Posteriori Variance Factor Test **failed** at a {}% confidence level".format((1 - alpha)*100))
            print("There is indication that errors exsist within residual or the math models")
            
    def significance_estimated_param(self, alpha = .05):
        """
        Desc:
            Determines whether there is statistical signifiance to beleive the final estimated value of parameters
        Input:
            alpha: to generate the two confidence intervals. Be sure to make sure that these values are generated in the respective dataframe of values, otherwise they won't be found :-)
            self.n: # of observations
            self.x_hat
            self.u_list: for labelling
            self.Cx: for extracting std dev values of parameters
        Output:
            retrunds dataframe of values [Unknown 	Final Value 	Value Standard Deviation 	Test Value 	Indicated Significance 	Alpha Tested 	Confidence Level 	Test Bounds]
        """
        #set up DOF (r)
        self.r = self.n - self.u
        
        high = stu.ppf(1.0 - alpha, self.r)
        low = stu.ppf(alpha, self.r)
        
        #final paramter values
        xs = []
        
        #unknown names in string format
        self.u_list
        us = []
        
        #list to store their signifiance as Signifiance or Not Significant
        sig = []
        
        #list to store the value that was checked
        sig_value = []
        
        #list of standard deviation values
        std = []
        
        #test values
        y = []
        
        #confidence levels
        conf = []
        
        #confidence levels
        alphs = []
        
        #test bounds
        bounds = []
        
        
        for i in range(0,self.u):
            std.append(m.sqrt(self.Cx[0,0]))
            
            y.append((self.x_hat[i]/std[i])[0,0])
            
            if y[i] > low and y[i] < high:
                #if fails then there IS statistical significance
                sig.append("No")
            else:
                sig.append("Yes")
                
            xs.append(self.x_hat[i][0,0])
            us.append(self.u_list[i])
            conf.append((1-alpha)*100)
            alphs.append(alpha)
            bounds.append(str([low, high]))
        #to store values in a dictionary before conversion to dataframe
        dict_list = {
                "Unknown": xs,
                "Final Value": us,
                "Value Standard Deviation": std,
                "Test Value": y,
                "Indicated Significance": sig,
                "Alpha Tested": alphs,
                "Confidence Level": conf,
                "Test Bounds": bounds
            }
        #return dict_list
        return pd.DataFrame.from_dict(dict_list)
    
    def semi_global_residuals(self, alpha = .05):
        """
        Desc:
            Conducts the semi global test on residuals, also known as the gooness-of-fit or normality test on residuals
        Input:
            self.r_hat: residuals
            alpha = .05: to find confidence level
            self.Cr: extracting std of residuals
            self.n: number of observations
        Output:
            prints whether the test passed and the reccomended interpretation
        """
        #normalize residuals
        norm_r = []
        
        for i in range(0,self.n):            
            norm_r.append(self.r_hat[i,0]/m.sqrt(self.Cr[i,i]))
            
        #number of bins
        M = round(m.sqrt(self.n))
        
        counts, bins = np.histogram(norm_r, bins = M)
        
        #compute estimated number of residuals per bin
        e = []
        for i in range(M):
            #get probability of total bin
            p_start = st.norm.cdf(bins[i])
            p_end = st.norm.cdf(bins[i+1])
            p = p_end - p_start
            
            #append total number of expected residuals
            e.append(p*self.n)
            
        #compute X_2 for each bin
        chis = []
        for i in range(M):
            chis.append((e[i]-counts[i])**2/e[i])
            
        #sum all chis for test statistic y
        y = sum(chis)
        
        #conduct statistical test
        dof = M - 1
        prob = 1 - alpha
        chi = chi2.ppf(prob, dof)
        
        print("{} tested with chi_square of {} ".format(y, chi))
        if y > chi:
            print("The Semi-Global, goodness-of-fit test on the residuals **Failled**")
            print("There is a sign that either there are outliers or the functional model was not appropriate for the data set")
        else:
            print("The Semi-Global, goodness-of-fit test on the residuals **Passed**")
            print("There is no sign of outliers or functional model errors")
        #plt.hist(norm_r, m)
        
    def blunder_detection(self, alpha = .01):
        """
        Desc:
            Conducts the local test on the residuals, aka blunder detection
        Input:
            alpha = .01: for 99% confidence of a blunder
            self.Cr: for extracting std of residuals
            self.r_hat: for extracting residuals
        Output:
            Returns a dataframe with columns ["Observation", "Outlier", "Test Value", "Test Bounds"]
        
        """
        #statistical test values
        low = st.norm.ppf(alpha/2)
        high = st.norm.ppf(1-alpha/2)
        
        #normalize residuals (test statistic)
        y = []
        
        #list of Yes or No outliers
        outlier = []
        
        #confidence levels
        conf = []
        
        #observations
        observations = []
        
        #test bounds
        bounds = []
        
        for i in range(0,self.n):  
            #for DF
            observations.append(i)
            bounds.append(str([low, high]))
            conf.append((1-alpha)*100)
            
            y.append(self.r_hat[i,0]/m.sqrt(self.Cr[i,i]))
            
            if y[i] > low and y[i] < high:
                #passes test --> not an outlier
                outlier.append("No")
            else:
                outlier.append("Yes")
                
        dic = {
            "Observation": observations,
            "Outlier": outlier,
            "Confidence Level": conf,
            "Test Value": y,
            "Test Bounds": bounds
        }
        return pd.DataFrame.from_dict(dic)
                
            
        
            