import jax as jx
import jax.numpy as jnp
import jaxpinns.architectures  as jxp_ar
import jaxpinns.optimizers  as jxp_op

from jaxpinns.loggers import logger

from .FiberNet import FiberNet, DeltaFiberNet

class Handler(object):
    def __init__(self, hiperparams, gen):

        dataset = gen.dataset(hiperparams)
        
        self.dataset = dataset
        self.hiperparams = hiperparams.copy()

        init_key = jx.random.PRNGKey(self.hiperparams["init_key"])
        type_model = self.hiperparams["type_model"]
        models = {"original": FiberNet,
                "delta": DeltaFiberNet}
        n_eigs=None
        if type_model == "delta":
            n_eigs = dataset.F.shape[-1]
        
        inputs = {"original":[3],
                "delta":[n_eigs],}
        
        self.model = models[type_model](
            dataset=dataset,
            CVmax=self.hiperparams["CVmax"],
            lambda_pde=self.hiperparams["lambda_pde"],
            lambda_tve=self.hiperparams["lambda_tve"],
            lambda_tva=self.hiperparams["lambda_tva"],
            lambda_df=self.hiperparams["lambda_df"]

        )
        layers = inputs[type_model]+self.hiperparams["layers"]
        CVlayers = inputs[type_model]+self.hiperparams["CVlayers"]
        # Setup architecture
        CV_args = (CVlayers,)
        _args = (layers,)
        self.model.architecture(jxp_ar.MLP, CV_args, dataset.Tmax.shape[0], jxp_ar.MLP, _args, init_key=init_key)
        # Setup optimizer
        saved_state = None
        # saved_state = 'chkpt_it_10000.npy'
        # learning_rate =  jx.example_libraries.optimizers.exponential_decay(1e-3, decay_steps=100, decay_rate=0.99)
        learning_rate = self.hiperparams["learning_rate"]
        args = (learning_rate, self.model.loss)
        self.model.optimizer(jxp_op.adam, *args, saved_state=saved_state)

        # Setup logger
        self.batch_size=self.dataset.batch_size
        io_keys = ['loss', 'loss_data','loss_pde', "loss_regu"]
        log_keys = ['loss', 'loss_data','loss_pde', "loss_regu"]
        log_funs = [self.model.loss, self.model.loss_data, self.model.loss_pde, self.model.loss_regu]#, model.loss_pde, model.loss_data]
        args = (io_keys, log_keys, log_funs)
        self.model.logger(logger, *args, io_step = 100)

    def train(self, epochs=1000):
        self.model.train(self.model.dataset, nIter = epochs , ntk_weights = False)

    def predictAT(self, return_times=False):
        type_model = self.hiperparams["type_model"]
        params = self.model.get_params(self.model.opt_state)
        if type_model=="original":
            AT_S = []  #Predicted at sampled points
            for i in range(len(self.dataset.X_e)):
                AT_S.append(self.model.AT_NN(params, self.dataset.X_e[i])[...,i]*self.model.Tmax[i])
            AT_F = self.model.AT_NN(params ,self.dataset.X)*self.model.Tmax 
        if type_model=="delta":
            AT_S = [] #Predicted at sampled points
            for i in range(len(self.dataset.F_e)):
                AT_S.append(self.model.AT_NN(params, self.dataset.F_e[i])[...,i]*self.model.Tmax[i]) 
            AT_F = self.model.AT_NN(params, self.dataset.F)*self.model.Tmax
        AT_T = [self.model.Tmax[i]*e for i,e in enumerate(self.dataset.T_e)]
        if return_times:
            return AT_S, AT_F, AT_T 
        else:
            return AT_S, AT_F
    
    def predictAngles(self):
        type_model = self.hiperparams["type_model"]
        if type_model=="original":
            input = self.dataset.X, self.dataset.P_p_predict
            triangs = self.dataset.triangs
            
            centroids = input[1][triangs].mean(axis=1)
            eig_centro = input[0][triangs].mean(axis=1)
        if  type_model=="delta":
            input = self.dataset.F, self.dataset.P_p_predict
            triangs = self.dataset.triangs
            
            centroids = input[1][triangs].mean(axis=1)
            eig_centro = input[0][triangs].mean(axis=1)


        predicted_times, cv, dcv, D_2D, D_3D, evals, evecs, eik_mismatch = self.model.predict(input[0], input[1])

        _batchs = len(eig_centro)
        D_cen_3D = []
        Nbat = 2
        for i in range(Nbat):
            st, ed = int(_batchs*i/Nbat), int(_batchs*(i+1)/Nbat)
            #print(st, ed)
            outs = self.model.predict(eig_centro[st:ed],centroids[st:ed])
            D_cen_3D.append(outs[4])
        D_cen_3D = jnp.vstack(D_cen_3D) # Model values at centroids
        predicted_vals_3D, predicted_vecs_3D = jnp.linalg.eigh(D_cen_3D)

        return predicted_times, eik_mismatch, predicted_vals_3D, predicted_vecs_3D
    


