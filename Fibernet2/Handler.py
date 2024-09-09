import jax as jx
import jax.numpy as jnp
from tqdm import tqdm

import jaxpinns.architectures  as jxp_ar
import jaxpinns.optimizers  as jxp_op
from jaxpinns.loggers import logger

from .FiberNet import FiberNet, DeltaFiberNet
from .FiberNetRPN import FiberNetRPN, DeltaFiberNetRPN

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


        # predicted_times, cv, dcv, D_2D, D_3D, evals, evecs, eik_mismatch = self.model.predict(input[0], input[1])

        _batchs = len(eig_centro)
        D_cen_3D = []
        Nbat = 20
        for i in tqdm(range(Nbat)):
            st, ed = int(_batchs*i/Nbat), int(_batchs*(i+1)/Nbat)
            #print(st, ed)
            outs = self.model.predict(eig_centro[st:ed],centroids[st:ed])
            D_cen_3D.append(outs[4])
        D_cen_3D = jnp.vstack(D_cen_3D) # Model values at centroids
        predicted_vals_3D, predicted_vecs_3D = jnp.linalg.eigh(D_cen_3D)

        return predicted_vals_3D, predicted_vecs_3D
    


class HandlerRPN(Handler):
    def __init__(self, hiperparams, gen):

        dataset = gen.dataset(hiperparams)
        
        self.dataset = dataset
        self.hiperparams = hiperparams.copy()

        init_key = jx.random.PRNGKey(self.hiperparams["init_key"])
        type_model = self.hiperparams["type_model"]
        models = {"original": FiberNetRPN,
                "delta": DeltaFiberNetRPN}
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
        self.model.architecture(jxp_ar.MLP, CV_args, len(self.dataset.Tmax), 
                    jxp_ar.MLP, _args, 
                    init_key=init_key,
                    ensemble_size=self.hiperparams["ensemble_size"],
                    lambda_prior=self.hiperparams["lambda_prior"])
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

    def predictAT(self, return_times=False):
        type_model = self.hiperparams["type_model"]
        params = self.model.get_params(self.model.opt_state)
        if type_model=="original":
            AT_S = []  #Predicted at sampled points
            for i in range(len(self.dataset.X_e)):
                AT_S.append(self.model.AT_PNN(params,self.model.prior_params, self.dataset.X_e[i])[...,i]*self.model.Tmax[i])
            AT_F = self.model.AT_PNN(params, self.model.prior_params ,self.dataset.X)*self.model.Tmax 
        if type_model=="delta":
            AT_S = [] #Predicted at sampled points
            for i in range(len(self.dataset.F_e)):
                AT_S.append(self.model.AT_PNN(params, self.model.prior_params, self.dataset.F_e[i])[...,i]*self.model.Tmax[i]) 
            AT_F = self.model.AT_PNN(params, self.model.prior_params, self.dataset.F)*self.model.Tmax
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


        _batchs = len(eig_centro)
        predicted_vals_3D = []
        predicted_vecs_3D = []
        Nbat = 100
        for i in tqdm(range(Nbat)):
            st, ed = int(_batchs*i/Nbat), int(_batchs*(i+1)/Nbat)
            outs = self.model.predict(eig_centro[st:ed],centroids[st:ed])
            vals_3D, vecs_3D = jx.vmap(jnp.linalg.eigh, (0))(outs[4])
            predicted_vals_3D.append(vals_3D)
            predicted_vecs_3D.append(vecs_3D)
        predicted_vals_3D = jnp.concatenate(predicted_vals_3D, axis=1) # Model values at centroids
        predicted_vecs_3D = jnp.concatenate(predicted_vecs_3D, axis=1) # Model values at centroids
        
        return predicted_vals_3D, predicted_vecs_3D
    
    def aggragateAngles(self, method="Medoid"):
        methods = {
            "Mean_Tensor":self._MeanTensorAgg,
            "Medoid":self._MedoidAgg
        }
        return methods[method]()
    
    def _MeanTensorAgg(self):
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


        _batchs = len(eig_centro)
        D_cen_3D = []
        D_cv = []
        Nbat = 100
        for i in tqdm(range(Nbat)):
            st, ed = int(_batchs*i/Nbat), int(_batchs*(i+1)/Nbat)
            outs = self.model.predict(eig_centro[st:ed],centroids[st:ed])
            D_cen_3D.append(outs[4])
            D_cv.append(outs[1])
        D_cen_3D = jnp.concatenate(D_cen_3D, axis=1) # Model values at centroids
        D_cv = jnp.concatenate(D_cv, axis=1) # Model values at centroids
        theta = jnp.arccos(jnp.clip(D_cv[...,2], -1, 1))
        
        logT = jx.vmap(jx.vmap(Tensor, (0,0,0)), (0,0,0))(theta, jnp.log(D_cv[...,:1]), jnp.log(D_cv[...,1:2]))

        L = logT.mean(0)
        eL = jx.vmap(jx.scipy.linalg.expm)(L)

        _batchs = len(eig_centro)
        D_cv_n = []
        D_cen_3D_m = []
        # predicted_vals_3D_m = []
        predicted_vecs_3D_m = []
        Nbat = 1000
        for i in tqdm(range(Nbat)):
            st, ed = int(_batchs*i/Nbat), int(_batchs*(i+1)/Nbat)
            d = jx.vmap(d_mean)(eL[st:ed])
            eV, aV = d[...,:2], d[...,2]
            outs = self.model.D_calculate(centroids[st:ed], eV, aV )[2]
            vals_3D, vecs_3D = jx.vmap(jnp.linalg.eigh, (0))(outs)
            D_cv_n.append(d)
            D_cen_3D_m.append(outs)
            # predicted_vals_3D_m.append(vals_3D)
            predicted_vecs_3D_m.append(vecs_3D)
        D_cv_n = jnp.vstack(D_cv_n) # Model values at centroids
        D_cen_3D_m = jnp.vstack(D_cen_3D_m) # Model values at centroids
        # predicted_vals_3D_m = jnp.vstack(predicted_vals_3D_m) # Model values at centroids
        predicted_vecs_3D_m = jnp.vstack(predicted_vecs_3D_m) # Model values at centroids
        return predicted_vecs_3D_m[...,2]
    
    def _MedoidAgg(self):
        vals, vecs = self.predictAngles()
        pred_fibers = vecs[...,2]
        pred_fibers = jnp.moveaxis(pred_fibers, 0, 1)
        simila0 = lambda v: jnp.array([(1-jnp.abs(v@v[i])) for i in range(len(v))])
        simila = lambda v: jnp.mean(simila0(v)**2, axis=1)
        representative = lambda v: v[jnp.argmin(simila(v))]
        pred_mean_fiber_B = jx.vmap(representative)(pred_fibers)
        # to_surf = lambda vects: vects-n*jnp.sum(vects * n, axis=-1, keepdims=True)
        normaliz = lambda vects: vects/jnp.linalg.norm(vects, axis=1)[:,None]
        pred_mean_fiber_B = normaliz(pred_mean_fiber_B)
        return pred_mean_fiber_B

    


def rot_mat(theta):
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

def rot_mat_trans(theta):
    return rot_mat(theta).T

def Tensor(theta, val1, val2):
    return rot_mat(theta)*jnp.array([val1, val2]).T@rot_mat_trans(theta)

def d_mean(M):
    vals, vects = jnp.linalg.eigh(M)
    #vals are in ac order
    #vect the same
    return jnp.vstack([vals[:, None][::-1], -vects[1:2,0:1]]).T[0]
