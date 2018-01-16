# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:57:44 2017

@author: Administrator
"""

#第一段代码是一个抽象的监督学习的模型类，可以用于神经网络等监督学习模型
    import numpy as np  
    from dp.common.optimize import minFuncSGD  
    import scipy.optimize as spopt  
      
    class SupervisedLearningModel(object):  
      
        def flatTheta(self):  
            ''''' 
            convert weight and intercept to 1-dim vector 
            '''  
            pass  
          
        def rebuildTheta(self,theta):  
            ''''' 
            overwrite the method in SupervisedLearningModel         
            convert 1-dim theta to weight and intercept 
            Parameters: 
                theta    - The vector hold the weights and intercept, needed by scipy.optimize function 
                           size:outputSize*inputSize 
            '''  
                  
        def cost(self, theta,X,y):  
            ''''' 
            This method is used to some optimize function such as fmin_cg,fmin_l_bfgs_b in scipy.optimize 
            Parameters: 
                theta        - 1-Dim vector of weight 
                X            - samples, numFeatures by numSamples 
                y            - labels,  numSamples elements vector 
            return: 
                the model cost 
            '''  
            pass  
          
        def gradient(self, theta,X,y):  
            ''''' 
            This method is used to some optimize function such as fmin_cg,fmin_l_bfgs_b in scipy.optimize 
            Parameters: 
                theta        - 1-Dim vector of weight 
                X            - samples, numFeatures by numSamples 
                y            - labels,  numSamples elements vector 
            return: 
                the model gradient 
            '''          
            pass  
          
        def costFunc(self,theta,X,y):  
            ''''' 
            This method is used to some optimize function such as minFuncSGD in this package 
            Parameters: 
                theta        - 1-Dim vector of weight 
                X            - samples, numFeatures by numSamples 
                y            - labels,  numSamples elements vector 
            return: 
                the model cost and gradient 
            '''       
            pass  
          
        def predict(self, Xtest):  
            ''''' 
            predict the test samples 
            Parameters: 
                X            - test samples, numFeatures by numSamples  
            return: 
                the predict result,a vector, numSamples elements 
            '''  
            pass  
          
        def performance(self,Xtest,ytest):  
            ''''' 
            Before calling this method, this model should be training 
            Parameter: 
                Xtest    - The data to be predicted, numFeatures by numData 
            '''              
            pred = self.predict(Xtest)     
            return np.mean(pred == ytest) * 100          
      
        def train(self,X,y):   
            ''''' 
            use this method to train the model. 
            Parameters: 
                theta        - 1-Dim vector of weight 
                X            - samples, numFeatures by numSamples 
                y            - labels,  numSamples elements vector         
            '''                        
            theta =self.flatTheta()  
              
            ret = spopt.fmin_l_bfgs_b(self.cost, theta, fprime=self.gradient,args=(X,y),m=200,disp=1, maxiter=100)  
            opttheta=  ret[0]      
              
            ''''' 
            opttheta = spopt.fmin_cg(self.cost, theta, fprime=self.gradient,args=(X,y),full_output=False,disp=True, maxiter=100)         
            '''  
            ''''' 
            options=dict() 
            options['epochs']=10 
            options['alpha'] = 2 
            options['minibatch']=256 
            opttheta = minFuncSGD(self.costFunc,theta,X,y,options) 
             
            '''  
            self.rebuildTheta(opttheta)  
            
            
            


#第二段代码定义了一个单一神经网络层NNLayer，从第一段代码中的SupervisedModel类继承下来。

#它在softmax和多层神经网络中用得到。            
    class NNLayer(SupervisedLearningModel):  
        ''''' 
        This class is single layer of Neural network  
        '''  
        def __init__(self, inputSize,outputSize,Lambda,actFunc='sigmoid'):  
            ''''' 
            Constructor: initialize one layer w.r.t params 
            parameters :  
                inputSize         - the number of input elements 
                outputSize        - the number of output 
                lambda            - weight decay parameter 
                actFunc        - the can be sigmoid,tanh,rectified linear function 
            '''  
            super().__init__()  
            self.inputSize = inputSize  
            self.outputSize = outputSize  
            self.Lambda = Lambda          
            self.actFunc=sigmoid  
            self.actFuncGradient=sigmodGradient  
              
            self.input=0            #input of this layer  
            self.activation=0       #output of the layer  
            self.delta=0            #the error of this layer          
            self.W=0                #the weight  
            self.b=0                #the intercept  
                      
            if actFunc=='sigmoid':    
                self.actFunc =  sigmoid  
                self.actFuncGradient = sigmodGradient          
            if actFunc=='tanh':              
                self.actFunc =  tanh  
                self.actFuncGradient =tanhGradient  
            if actFunc=='rectfiedLinear':              
                self.actFunc =  rectfiedLinear    
                self.actFuncGradient =  rectfiedLinearGradient  
      
            #epsilon的值是一个经验公式    
            #initialize weights and intercept (bias)  
            epsilon_init = 2.4495/np.sqrt(self.inputSize+self.outputSize)*0.001  
            theta = np.random.rand(self.outputSize, self.inputSize + 1) * 2 * epsilon_init - epsilon_init  
            self.rebuildTheta(theta)  
                              
        def flatTheta(self):  
            ''''' 
            convert weight and intercept to 1-dim vector 
            '''  
            W = np.hstack((self.W, self.b))  
            return W.ravel()   
          
        def rebuildTheta(self,theta):  
            ''''' 
            overwrite the method in SupervisedLearningModel         
            convert 1-dim theta to weight and intercept 
            Parameters: 
                theta    - The vector hold the weights and intercept, needed by scipy.optimize function 
                           size:outputSize*inputSize 
            '''  
            W=theta.reshape(self.outputSize,-1)  
            self.b=W[:,-1].reshape(self.outputSize,1)   #bias b is a vector with outputSize elements  
            self.W = W[:,:-1]    
      
        def forward(self):  
            ''''' 
            Parameters: 
                X -  The examples in a matrix,  
                    it's dimensionality is inputSize by numSamples 
            '''    
            Z = np.dot(self.W,self.input)+self.b     #Z          
            self.activation= self.actFunc(Z)             #activations  
            return self.activation  
          
        def backpropagate(self):  
            ''''' 
            parameter: 
                inputMat - the actviations of previous layer, or input of this layer, 
                             inputSize by numSamples 
                delta - the next layer error term, outputSize by numSamples 
             
            assume current layer number is l, 
            delta is the error term of layer l+1. 
            delta(l) = (W(l).T*delta(l+1)).f'(z) 
            If this layer is the first hidden layer,this method should not 
            be called 
            The f' is re-writed to void the second call to the activation function 
            '''  
            return np.dot(self.W.T,self.delta)*self.actFuncGradient(self.input)  
          
        def layerGradient(self):  
            ''''' 
            grad_W(l)=delta(l+1)*input.T 
            grad_b(l) = SIGMA(delta(l+1)) 
            parameters: 
                inputMat - input of this layer, inputSize by numSamples 
                delta    - the next layer error term 
            '''  
            m=self.input.shape[1]  
            gw = np.dot(self.delta,self.input.T)/m  
            gb = np.sum(self.delta,1)/m  
            #combine gradients of weights and intercepts  
            #and flat it  
            grad = np.hstack((gw, gb.reshape(-1,1)))  
               
            return grad  
              
          
    def sigmoid(Z):  
        return 1.0 /(1.0 + np.exp(-Z))  
      
    def sigmodGradient (a):  
        #a = sigmoid(Z)  
        return a*(1-a)  
      
    def tanh(Z):  
        e1=np.exp(Z)  
        e2=np.exp(-Z)  
        return (e1-e2)/(e1+e2)  
      
    def tanhGradient(a):  
        return 1-a**2  
      
    def rectfiedLinear(Z):  
        a = np.zeros(Z.shape)+Z  
        a[a<0]=0  
        return a  
      
    def rectfiedLinearGradient(a):  
        b = np.zeros(a.shape)+a      
        b[b>0]=1  
        return b  




#第三段代码是softmax回归的实现，它从NNLayer继承。
    import numpy as np  
    #import scipy.optimize as spopt  
    from dp.supervised import NNBase  
    from time import time  
    #from dp.common.optimize import minFuncSGD  
    class SoftmaxRegression(NNBase.NNLayer):  
        ''''' 
        We assume the last class weight to be zeros in this implementation. 
        The weight decay is not used here. 
     
        '''  
        def __init__(self, numFeatures, numClasses,Lambda=0):  
            ''''' 
            Initialization of weights,intercepts and other members  
            Parameters: 
                numClasses    - The number of classes to be classified 
                X             - The training samples, numFeatures by numSamples 
                y             - The labels of training samples, numSamples elements vector 
            '''        
      
            # call the super constructor to initialize the weights and intercepts  
            # We do not need the last weights and intercepts of the last class  
            super().__init__(numFeatures, numClasses - 1, Lambda, None)  
              
            #self.X=0          
            self.y_mat=0    
                 
        def predict(self, Xtest):  
            ''''' 
            Prediction. 
            Before calling this method, this model should be training 
            Parameter: 
                Xtest    - The data to be predicted, numFeatures by numData 
            '''  
            Z = np.dot(self.W, Xtest) + self.b  
            #add the prediction of the last class,they are all zeros  
            lastClass = np.zeros((1, Xtest.shape[1]))  
            Z = np.vstack((Z, lastClass))  
            #get the index of max value in each column, it is the prediction  
            return np.argmax(Z, 0)         
             
        def forward(self):  
            ''''' 
            get the matrix of softmax hypothesis 
            this method  will be called by cost and gradient methods 
            Parameters: 
                 
            '''  
            h = np.dot(self.W, self.input) + self.b  
            h = np.exp(h)  
            #add probabilities of the last class, they are all ones   
            h = np.vstack((h, np.ones((1, self.input.shape[1]))))  
            #The probability of all classes  
            hsum = np.sum(h, axis=0)  
            #get the probability of each class  
            self.activation = h / hsum  
            #delta = -(self.y_mat-h)  
            self.delta = self.activation - self.y_mat  
            self.delta=self.delta[:-1, :]  
              
            return self.activation  
      
        def setTrainingLabels(self,y):  
            # convert Vector y to a matrix y_mat.  
            # For sample i, if it belongs to the k-th class,   
            # y_mat[k,i]=1 (k==j), y_mat[k,i]=0 (k!=j)          
            y = y.astype(np.int64)  
            m=y.shape[0]  
            yy = np.arange(m)  
            self.y_mat = np.zeros((self.outputSize+1, m))            
            self.y_mat[y, yy] = 1  
              
        def softmaxforward(self,theta,X,y):  
            self.input = X  
            self.setTrainingLabels(y)  
            self.rebuildTheta(theta)  
            return self.forward()  
      
        def cost(self, theta,X,y):  
            ''''' 
            The cost function. 
            Parameters: 
                theta    - The vector hold the weights and intercept, needed by scipy.optimize function 
                           size: (numClasses - 1)*(numFeatures + 1)         
            '''  
            h = np.log(self.softmaxforward(theta,X,y))  
            #h * self.y_mat, apply the indicator function  
            cost = -np.sum(h *self.y_mat, axis=(0, 1))  
              
            return cost / X.shape[1]  
          
        def gradient(self, theta,X,y):  
            ''''' 
            The gradient function. 
            Parameters: 
                theta    - The vector hold the weights and intercept, needed by scipy.optimize function 
                           size: (numClasses - 1)*(numFeatures + 1)         
            '''  
            self.softmaxforward(theta,X,y)          
      
            #get the gradient  
            grad = super().layerGradient()  
                     
            return grad.ravel()  
          
        def costFunc(self,theta,X,y):  
       
            grad=self.gradient(theta, X, y)  
            h=np.log(self.activation)  
            cost = -np.sum(h * self.y_mat, axis=(0, 1))/X.shape[1]  
            return cost,grad      
      
      
    def checkGradient(X,y):  
              
        sm = SoftmaxRegression(X.shape[0], 10)  
        #W = np.hstack((sm.W, sm.b))  
        #sm.setTrainData(X, y)  
        theta = sm.flatTheta()      
        #grad = sm.gradient(theta,X, y)  
        cost,grad=sm.costFunc(theta, X, y)     
        numgrad = np.zeros(grad.shape)  
          
        e = 1e-6  
          
        for i in range(np.size(grad)):           
            theta[i]=theta[i]-e  
            loss1,g1 =sm.costFunc(theta,X, y)  
            theta[i]=theta[i]+2*e  
            loss2,g2 = sm.costFunc(theta,X, y)  
            theta[i]=theta[i]-e              
              
            numgrad[i] = (-loss1 + loss2) / (2 * e)  
              
        print(np.sum(np.abs(grad-numgrad))/np.size(grad))   
        
        
  




#测试代码      
    X = np.load('../../common/trainImages.npy') / 255  
    X = X.T  
    y = np.load('../../common/trainLabels.npy')  
    '''''     
    X1=X[:,:10] 
    y1=y[:10] 
    checkGradient(X1,y1) 
    '''      
    Xtest = np.load('../../common/testImages.npy') / 255  
    Xtest = Xtest.T  
    ytest = np.load('../../common/testLabels.npy')  
    sm = SoftmaxRegression(X.shape[0], 10)  
    t0=time()  
    sm.train(X,y)  
      
    print('training Time %.5f s' %(time()-t0))  
      
    print('test acc :%.3f%%' % (sm.performance(Xtest,ytest)))  