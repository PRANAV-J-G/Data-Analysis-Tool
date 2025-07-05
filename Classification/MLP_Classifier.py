import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics,preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN
from typing import NamedTuple
from numpy.core.fromnumeric import shape
import streamlit as st #streamlit backend
import collections


class NN_Classifier:
    """Neural Network Classifier Class:

    This Class contains the methods used for Neural Network Classification using MLP Classifier from sklearn library

    Class input parameters:

    :param df: The input data frame
    :type df: Pandas DataFrame
    :param NN_Inputs: Tuple of parameters for the Classifier clarified by the user
    :type NN_Inputs: Named Tuple
    :param dependant_var_index: The index of the target column in the df for the Classification
    :type dependant_var_index: int

    Class Output Parameters:

    :param y_pred: The resulting output of the Classification test
    :type y_pred: float 
    :param y_actual: The expected output of the Classification test
    :type y_actual: float 
    :param length: The length of the output of the Classification test set
    :type length: int 
    :param Train_score: Model Score (Accuracy) on the Training data
    :type Train_score: float 
    :param test_score: Model Score (Accuracy) on the Testing data  
    :type test_score: float
    :param model: The MLP Classifier model created using the specified inputs
    :type model: MLPClassifier
    :param Error_message: Error message if an exception was encountered during the processing of the code
    :type Error_message: str
    :param flag: internal flag for marking if an error occurred while processing a previous method
    :type flag: bool
    """
    
    
    y_pred:                     int # resulting output
    y_actual:                   int # expected output
    length:                     int   # length of y_test
    model:                      MLPClassifier #Outputting the Classifier to use outside the class
    Train_score:                float #Model score on Training data
    X_test:                     float #Testing samples
    test_score:                 float #Model score on the Testing data
    Report:                     pd.DataFrame #Comlete Report of the classifier performance on the testing data
    Report_dic:                 dict
    Error_message:              str #Error message to be sent to the user if any issues occur
    flag:                       bool #Flag to signal an Error occurred in a previous method


    dependant_var_index =0
    flag=False
    Error_message='No Error occurred in the processing'
 
    #Constructor
    #External inputs are the Data frame object, the Named Tuple of NN_Inputs and the index of the dependant variable in the Data frame
    def __init__(self,df,NN_Inputs,dependant_var_index):
        """Class Constructor:

        :param df: The input data frame
        :type df: _type_
        :param NN_Inputs: _description_
        :type NN_Inputs: _type_
        :param dependant_var_index: _description_
        :type dependant_var_index: _type_
        """
        self.df=df
        self.NN_Inputs=NN_Inputs
        
        self.k=dependant_var_index
        self.handle()
        #After initial Handling of the data we check to see if the user wants to normalize the X values prior to training the model
        if self.NN_Inputs.Normalize:
            self.preprocess()
        elif self.flag!=True:
            #if Normalization is not checked, the X values fed into the regressor are the same as the output from the initial handle method
            self.x_n=self.X
        


    def preprocess(self):
        """Method Used to Normalize the X data if the user required

        This method is called when the class instance is created and the Normalize flag in the input NN_Inputs tuple is True.

        """
        #Simple Normalization method for X Data
        scaler=preprocessing.MinMaxScaler()
        self.x_n = scaler.fit_transform(self.X)
    
    #Data handling method, creates the X and Y arrays that go into the Train_test_split method
    #Called when the object is instantiated within the constructor 
    def handle(self):
        """Data Handling Method:

        This method takes the Target column index and splits the data frame "df" into X and Y numpy arrays so they are ready for being split into train and test sets

        This method is called internally once the class instance is created and the X,Y output arrays are fed to the "Classify" method 
        """
        try:
            self.internal_data =self.df.drop(self.df.iloc[:,[self.k]],axis=1)
            nxm=shape(self.internal_data)
            n=nxm[0]
            m=nxm[1]
        
            
            self.X=np.ndarray(shape=(n,m),dtype=float, order='F')
            

            for l in range(0,m):  
            
                for i in range(0,n):
                
                    self.X[i,l]=self.internal_data.iloc[i,l]
            
            Y=np.ndarray(shape=(n,1), dtype=float, order='F')
            Y[:,0]=self.df.iloc[:,self.k]

            self.y=np.ravel(Y)
            self.Class_len=len(collections.Counter(self.y))
        except Exception as e:
            self.Error_message='Error in Handling Method: ' + str(e)
            self.flag=True

    #Data handling method, Used by the Regressor to re-handle the data after resampling and shuffling is done
    def handle2(self,df):
        """
        Data Handling Method Version 2:

        This method takes the Target column index and splits the data into X and Y numpy arrays for internal use

        This method is used internally by the "Classify" method after the data is resampled and reshuffled, it takes input dataframe and returns the X and Y arrays to the caller

        :param df: the Data frame passed to handle
        :type df: Pandas DataFrame
        :return: X,Y
        :rtype: numpy arrays
        """
        internal_data =df.drop(df.iloc[:,[self.k]],axis=1)
        nxm=shape(internal_data)
        n=nxm[0]
        m=nxm[1]
    
        
        X=np.ndarray(shape=(n,m),dtype=float, order='F')
        

        for l in range(0,m):  
        
            for i in range(0,n):
            
                X[i,l]=internal_data.iloc[i,l]
        
        Y=np.ndarray(shape=(n,1), dtype=float, order='F')
        Y[:,0]=df.iloc[:,self.k]

        y_t=np.ravel(Y)
        return X, y_t

    #Method that creates the MLP Classifier and returns the Named Tuple of to be used in other methods  
    def Classify(self):
        """ 
        Classifier Creation Method:

        In this method 3 different steps are done:

            1. This method splits the data into train and test sets, then creates the MLP Classifier based on the user inputs from NN_Inputs Named Tuple.
            2. The extracted Train data is resampled (or not) based on the user input Normalize flag from the NN_Inputs parameter.
            3. Model is fitted on the resampled/normal data and  returns some metrics for the performance of the model on the test and train data sets.
            
            

        :return: Modified set of class parameters
        
        """
        if (self.flag) !=True:
            try:
                    
                X_train, self.X_test, y_train, self.y_actual= train_test_split(self.x_n,self.y,test_size=self.NN_Inputs.test_size,shuffle=True, random_state=109)
                self.model = MLPClassifier(hidden_layer_sizes = self.NN_Inputs.hidden_layers, 
                                    activation = self.NN_Inputs.activation_fun, solver = self.NN_Inputs.solver_fun, 
                                    learning_rate = 'adaptive', max_iter = self.NN_Inputs.Max_iterations, random_state = 109,shuffle=True,batch_size=15,alpha=0.0005 )
                if self.NN_Inputs.resample:
                    #Re-sampling method used to handle imbalanced data 
                    if self.Class_len >2:
                        method = SMOTEENN(random_state=109,sampling_strategy='minority')
                    else:
                        method = SMOTEENN(random_state=109,sampling_strategy=0.48)    
                    X_res, y_res = method.fit_resample(X_train, y_train)
                    df=pd.DataFrame(X_res)
                    #print(df)
                    new_df=pd.concat([df,pd.DataFrame(y_res,columns=[self.k])],axis=1)
                    #print(new_df.head)
                    df1=new_df.sample(frac=1,random_state=1).reset_index(drop=True)
                    #After Resampling and shuffling, we feed the df to the handle method to generate X,y arrays used to train the model
                    x2,y2 = self.handle2(df1)
                else:
                    x2,y2 = X_train,y_train
            
                #print(shape(x2))
                #print(shape(y2))
                

                
                self.model.fit(x2, y2)

                self.Train_score= self.model.score(X_train,y_train)
                self.test_score= self.model.score(self.X_test,self.y_actual)
                self.y_pred = self.model.predict(self.X_test)

                y_pred_int = np.ndarray.tolist(self.y_pred)
                self.length = len(y_pred_int)
                #target_names = ['class 0', 'class 1']
                self.Report_dic=classification_report(self.y_actual, self.y_pred, output_dict=True)#, target_names=target_names)
                self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)

                # st.write(self.y_actual)
            except Exception as e:
                self.Error_message= 'Error in Classifier Creation: ' + str(e)
                self.flag=True
                st.warning(self.Error_message)

                self.Train_score= 'Refer To error in Classifier Creation'
                self.test_score= 'Refer To error in Classifier Creation'
                #self.coeff=self.model.coefs_

                self.y_actual='Refer To error in Classifier Creation'
                self.y_pred = 'Refer To error in Classifier Creation'

                self.y_pred = 'Refer To error in Classifier Creation'
                self.length = 'Refer To error in Classifier Creation'
                
                #Mean squared error and accuracy
                self.Report_dic ={'1': ['Refer To error in Handling Method 1'] }
                self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)
                    #self.Report = "Vasya"
        else:
            st.warning(self.Error_message)

            self.Train_score= 'Refer To error in Handling Method'
            self.test_score= 'Refer To error in Handling Method'
            #self.coeff=self.model.coefs_

            self.y_actual='Refer To error in Handling Method'
            self.y_pred = 'Refer To error in Handling Method'

            self.y_pred = 'Refer To error in Handling Method'
            self.length = 'Refer To error in Handling Method'
            
            #Mean squared error and accuracy
            self.Report_dic ={'1': ['Refer To error in Handling Method 1'] }
            self.Report=pd.DataFrame.from_dict(self.Report_dic)#, target_names=target_names)
    


        return self
    
    def printing(self):
        """Printing Outputs:

        This method prints the chosen metrics to the user after the model is trained and fitted

        The metrics are:
            1. Model Accuracy on the Training Data
            2. Model Accuracy on the Testing Data
            3. Length of the output array
            4. Classification Report
        """
        if (self.flag) != True:
            self.Error_message= ' No Error Occurred during processing of the code'
        
        try:
            st.warning(self.Error_message)
            cc1, cc2 = st.columns(2)
            with cc1:
                # st.metric('Expected output:        ', self.NN_Outputs.y_actual)
                # st.write('Predicted Output:       ', self.NN_Outputs.y_pred)
                st.metric('Model Accuracy on the Training Data:',  round(self.Train_score, 4))
                st.metric('Model Accuracy on the Testing Data:',  round(self.test_score, 4))
                st.write('Accuracy: it is simply a ratio of correctly predicted observations to the total observations')
                st.metric('Length of output array: ',  self.length)
            with cc2:
                st.write('Classification Report: ')
                st.dataframe(self.Report)
                st.write("Precision:  Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. ")
                st.write('Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class')
                st.write('The F1 score represents the balance of accuracy and recall. F1 Score is the weighted average of Precision and Recall. Good for Unbalanced dataset.')

        except Exception as e:
            self.Error_message = 'Error while printing outputs: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)

        
    def Conf(self):
        """Creation of Confusion matrix:

        This method outputs a confusion matrix figure to show the quality of the classification on the test data.

        The Confusion matrix dimensions and Labels are derived from the number and names of the different unique values in the Labels Target column for the classification. 
        """
        fig = plt.figure(figsize=(10, 4))
        ax=fig.add_subplot(111)
        if (self.flag) !=True:
            try:
                conf_matrix=metrics.confusion_matrix(self.y_actual, self.y_pred)
                df_conf=pd.DataFrame(conf_matrix,range(self.Class_len),range(self.Class_len))
                sn.set(font_scale=1.4)
                sn.heatmap(df_conf, annot=True, annot_kws={"size":16},ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                key=list(self.Report_dic)
                labels=key[0:(self.Class_len)]
                #print(labels)
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

                st.pyplot(fig)
                #plt.show()
            except Exception as e:
                self.Error_message = 'Error while creating Confusion Matrix: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')


class classifier_inputs(NamedTuple):
    """
    This class is used to parse inputs from the user into this Named Tuple structure for easy use inside the NN_Classifier class.

    Below is a description of the Named Tuple Elements:

    """
    
    test_size:              float  ;"""Test size percentage"""
    activation_fun:         tuple  ;"""Activation function selection"""
    hidden_layers:          tuple  ;"""Size of hidden layer and number of neurons in each layer"""
    solver_fun:             tuple  ;"""Solver function Selection"""
    Max_iterations:         int    ;"""Number of Maximum iterations"""
    Normalize:              bool   ;"""Flag to normalize X data or not"""
    resample:               bool   ;"""Flag to resample for imbalanced data or not"""

# data2 = pd.read_csv("D:\MAIT\OOP\Datasets/transfusion.csv",',')
# data = pd.read_csv("D:\\TH Koeln\\Wolf\\Project\\Data\\Classification.data", ',')

# activation_fun1 = ("identity", "logistic", "tanh", "relu")
# solver_fun1 = ("lbfgs", "sgd", "adam")
# hidden_layers2=(20,5)
# inputs=classifier_inputs(0.2,activation_fun1[1],hidden_layers2,solver_fun1[2],500,False)

# Classifier = NN_Classifier(data,inputs,4)
# Classifier.Classify()
# Classifier.printing()
#Classifier.Conf()
