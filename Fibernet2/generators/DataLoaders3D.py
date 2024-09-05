import jax as jx
import numpy as np


class DataLoader():
    def __init__(self,triangs, X, P_p_predict, X_e, T_e, batch_size=1, seed=1234,
                 scaled=True, **kwargs):#model=MLP, layers for call
        self.key = jx.random.PRNGKey(seed)

        # Assign class parameters  
        self.triangs = triangs
        self.X = X

        # Eikonal Params
        self.Tmax = np.array([t.max() for t in T_e])
        self.P_p_predict = P_p_predict
        # Activation Time
        self.T_e = [t/t.max() for t in T_e]     
        self.X_e = X_e
        

        # Model Info
        self.batch_size = batch_size
        self.setScaled(scaled)
    
    def setScaled(self, scaled):
        self.scaled = scaled
        self.lb = 0
        self.ub = 1
        if scaled:
            self.lb = self.X.min(0)
            self.ub = self.X.max(0)
            self.ub+=1*(self.ub==self.lb)

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = jx.random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = jx.random.choice(key, self.X.shape[0], (self.batch_size,), replace = False)

        # Make inputs, outputs
        inputs  = (self.X_e, self.X[idx])
        outputs = (self.T_e, self.P_p_predict[idx])
        return inputs, outputs
    

class DataDeltaLoader():
    def __init__(self, triangs, F, P_p_predict, F_e, T_e, operator,
                 batch_size=1, seed=1234,
                 scaled=True, **kwargs):
        self.key = jx.random.PRNGKey(seed)
  

        # Assign class parameters
        self.triangs = triangs
        self.F = F
        self.operator = operator
        # Eikonal Params
        self.Tmax = np.array([t.max() for t in T_e])
        self.P_p_predict = P_p_predict

        # Activation Maps
        self.F_e = F_e #funciones propias del sampling X_eig anteriormente
        self.T_e = [t/t.max() for t in T_e]  

        # Model Info
        self.batch_size = batch_size
        self.setScaled(scaled)


    def setScaled(self, scaled):
        self.scaled = scaled
        self.lb = 0
        self.ub = 1
        if scaled:
            self.lb = self.F.min(0)
            self.ub = self.F.max(0)
            self.ub+=1*(self.ub==self.lb)

    def __getitem__(self, index):
        # It doesnt do permutation dataset[0] and dataset[0] are not equal
        'Generate one batch of data'
        self.key, subkey = jx.random.split(self.key)
        inputs, targets = self.__data_generation(subkey)
        return inputs, targets

    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx_e = jx.random.choice(key, self.operator.shape[0], (self.batch_size,), replace = False)

        Xc = self.F[np.ravel(self.triangs[idx_e]),:] # AT
        Xd = self.F[(self.triangs[idx_e]),:].mean(axis=1) # For Tensor D
        P_p = self.P_p_predict[(self.triangs[idx_e])[:,0]]
        # Make inputs, outputs
        inputs  = (self.F_e, Xc, Xd, self.operator[idx_e], )
        outputs = (self.T_e, P_p,)
        return inputs, outputs
    