import jax as jx
import jax.numpy as jnp
import itertools
from functools import partial
from .FiberNet import FiberNet, DeltaFiberNet


class FiberNetRPN(FiberNet):
    # Initialize the class
    def __init__(self, 
                 dataset,
                 CVmax=1.0, 
                 lambda_df=1., 
                 lambda_pde=1e-4,
                 lambda_tve=1e-2, 
                 lambda_tva=1e-9
                 ):
        super().__init__(
                 dataset,
                 CVmax, 
                 lambda_df, 
                 lambda_pde,
                 lambda_tve, 
                 lambda_tva)

    def architecture(self, 
                     CV_net, CV_args,
                     maps_number, map_net, map_args, 
                     init_key = jx.random.PRNGKey(0),
                     ensemble_size=10,
                     lambda_prior=1):
        ''' Network initialization and evaluation functions '''
        self.CV_init, self.CV_apply = CV_net(*CV_args)
        prior_layers = CV_args[0][:-1]+ [CV_args[0][-1]+maps_number]
        self.init_prior, self.apply_prior = CV_net(prior_layers)

        self.enssiz =ensemble_size
        self.lambda_prior = lambda_prior
        # Random keys
        k1, k2, k3 = jx.random.split(init_key, 3)
        keys_1 = jx.random.split(k1, ensemble_size)
        keys_2 = jx.random.split(k2, ensemble_size)
        
        CV_params = jx.vmap(self.CV_init)(keys_1)
        prior_params = jx.vmap(self.init_prior)(keys_2)

        # CV_params = self.CV_init(init_key)

        self.p_NN = maps_number
        maps_init = []
        maps_apply = []
        maps_params = []
        for m_number in range(maps_number):
            keys_3 = jx.random.split(k3, ensemble_size)
            m_init, m_apply = map_net(*map_args)

            # m_params = m_init(init_key)
            m_params = jx.vmap(m_init)(keys_3)
            maps_init.append(m_init)
            maps_apply.append(m_apply)
            maps_params.append(m_params)
        self.maps_init, self.maps_apply = maps_init, maps_apply

        self.net_params = (CV_params, *maps_params)
        self.prior_params = prior_params
    
    def optimizer(self, opt, *args, saved_state=None):
        ''' Optimizer initialization and update functions '''
        self.opt_init, \
        self.opt_update, \
        self.get_params = opt(*args)
        if saved_state == None:
            self.opt_state = self.opt_init(self.net_params)
        else:
            state  = jnp.load(saved_state, allow_pickle=True)
            self.opt_state = [jx.device_put(s) for s in state]
            self.net_params = self.get_params(self.opt_state)
        self.itercount = itertools.count()

    # Conduction velocity tensor nn
    @partial(jx.jit, static_argnums=(0,))
    def CVT_NN(self, net_params, input): #net_params = self.net_params[0]
        input = self.scaler(input)
        d_pre = self.CV_apply(net_params, input)
        return d_pre
    
    @partial(jx.jit, static_argnums=(0))
    def CVT_PNN(self, net_params, prior_params, input):
        NN = jx.vmap(self.CVT_NN, (0, None))(net_params[0], input)
        PRIO = jx.vmap(self.apply_prior, (0, None))(prior_params, input)
        d_pre = NN+PRIO[...,:3]*self.lambda_prior

        vel = self.C * (jx.nn.sigmoid(d_pre[...,:2]))
        ang = jnp.tanh(d_pre[...,2:])
        
        d = jnp.concatenate([vel, ang], axis=-1)
        return d

    
    @partial(jx.jit, static_argnums=(0))
    def AT_PNN(self, net_params, prior_params, input):
        NN = jx.vmap(self.AT_NN, (0, None))(net_params, input)
        PRIO = jx.vmap(self.apply_prior, (0, None))(prior_params, input)
        return NN+PRIO[...,3:]*self.lambda_prior

    # Gradient of Activation time for EIKONAL EQ
    @partial(jx.jit, static_argnums=(0,))
    def GRAD_AT(self, net_params, prior_params, input):
        fun = lambda x: self.AT_PNN(net_params, prior_params, x)
        res = jx.vmap(jx.jacrev(fun))(input) #output -> Samples, model, output, coord 
        res = res.reshape((*res.shape[:2],-1)) # to -> samples, model, output*coords
        res = jnp.moveaxis(res, 0,1) #to-> model, samples, output*coords
        return res

    # Gradient of CV for Huber Regularization
    @partial(jx.jit, static_argnums=(0,))
    def GRAD_CVT(self, net_params, prior_params, input):
        fun = lambda x: self.CVT_PNN(net_params, prior_params,x)
        res = jx.vmap(jx.jacrev(fun))(input) #output -> Samples, model, output, coord 
        aV_x = res[:,:, 2, :]
        eV_x = res[:,:, :2, :]
        eV_x = eV_x.reshape((*eV_x.shape[:2],-1))
        aV_x = jnp.moveaxis(aV_x, 0,1)
        eV_x = jnp.moveaxis(eV_x, 0,1)
        d_x = [eV_x, aV_x]
        return d_x
    
    # Loss of data (Activation time)
    @partial(jx.jit, static_argnums=(0,))
    def loss_data(self, net_params, batch, *args):
        X, Y = batch
        X_e = X[0]
        phi_e = Y[0]

        phi_square = []
        for i in range(self.p_NN):
            input = X_e[i]
            phi_pred =  self.AT_PNN(net_params, self.prior_params, input)[...,i]
            phi_square.append(jnp.mean((phi_e[i]-phi_pred)**2))
    
        loss = jnp.mean(jnp.array(phi_square))
        return loss

    # Loss of regularization
    @partial(jx.jit, static_argnums=(0,))
    def loss_regu(self, net_params, batch, *args):
        X, Y = batch
        _, Xs = X
        eV_x, aV_x = self.GRAD_CVT(net_params,self.prior_params, Xs)
        # Huber Regularization
        eV_TV = self.TVHuber(eV_x, 1e-3)[0]
        aV_TV = self.TVHuber(aV_x, 1e-3)[0]

        loss1 = jnp.mean(eV_TV, axis=(1,2))
        loss2 = jnp.mean(aV_TV, axis=(1,2))

        loss1 = jnp.mean(loss1)
        loss2 = jnp.mean(loss2)

        return loss1, loss2
    
    @partial(jx.jit, static_argnums=(0,))
    def loss_pde(self, net_params, batch, *args):
        X, Y = batch
        _, Xs = X
        _, P_p = Y

        eik_loss = self.operator_net(net_params, Xs, P_p)

        loss = jnp.mean(eik_loss**2, axis=(1,2))
        loss = jnp.mean(loss)
        return loss


    def loss(self, net_params, batch, *args):
        self.net_params = net_params
        
        A = self.loss_data(net_params, batch, *args)
        B = self.loss_pde(net_params, batch, *args)
        C1, C2 = self.loss_regu(net_params, batch, *args)
        A = jnp.nan_to_num(A)
        B = jnp.nan_to_num(B)
        C1 = jnp.nan_to_num(C1)
        C2 = jnp.nan_to_num(C2)

        L = self.lambda_DF*A
        L +=self.lambda_PDE*B
        L +=self.alpha_e*C1 +self.alpha*C2
        return L
    

    @partial(jx.jit, static_argnums=(0,))
    def operator_net(self, net_params, X_star, P_p_predict):

        d = self.CVT_PNN(net_params, self.prior_params, X_star)
        eV, aV = d[...,:2], d[...,2]

        D, evecs, D_canon_3D = jx.vmap(lambda ev, av:self.D_calculate(P_p_predict, ev, av, eps=1.e-9), (0, 0) )(eV, aV)

        T_x = self.GRAD_AT(net_params, self.prior_params, X_star)
        
        eik_loss = jx.vmap(self.eikloss, (0,0))(T_x, D_canon_3D)

        return eik_loss
    
    def updateSelf(self, net_params, X_star, P_p_predict):
        d = self.CVT_PNN(net_params, self.prior_params, X_star)
        eV, aV = d[...,:2], d[...,2]

        D, evecs, D_canon_3D = jx.vmap(lambda ev, av:self.D_calculate(P_p_predict, ev, av, eps=1.e-9), (0, 0) )(eV, aV)

        T_x = self.GRAD_AT(net_params,self.prior_params, X_star)
        
        eik_loss = jx.vmap(self.eikloss, (0,0))(T_x, D_canon_3D)

        self.T_pred = self.AT_PNN(net_params,self.prior_params, X_star)        
        self.CV_pred = d
        self.CV_x = self.GRAD_CVT(net_params,self.prior_params, X_star)
        self.D = D
        self.D_canon_3D = D_canon_3D
        self.evals = eV
        self.evecs = evecs
        self.f_T_pred = eik_loss
    
    
    
    

class DeltaFiberNetRPN(FiberNetRPN):
    # Initialize the class
    def __init__(   self, 
                    dataset,
                    CVmax=1.0, 
                    lambda_df=1., 
                    lambda_pde=1e-4,
                    lambda_tve=1e-2, 
                    lambda_tva=1e-9
                    ):
        super().__init__(   dataset,
                            CVmax, 
                            lambda_df, 
                            lambda_pde,
                            lambda_tve, 
                            lambda_tva)
      
    # Loss of regularization
    @partial(jx.jit, static_argnums=(0,))
    def loss_regu(self, net_params, batch, *args):
        
        X, _ = batch
        _, Xc, _, operator, = X

        eV_x, aV_x = self.CVT_grad(net_params, self.prior_params,Xc, operator)

        # Huber Regularization
        eV_TV = self.TVHuber(eV_x, 1e-3)[0]
        aV_TV = self.TVHuber(aV_x, 1e-3)[0]

        loss1 = jnp.mean(eV_TV, axis=(1,2))
        loss2 = jnp.mean(aV_TV, axis=(1,2))

        loss1 = jnp.mean(loss1)
        loss2 = jnp.mean(loss2)

        return loss1, loss2
    
    @partial(jx.jit, static_argnums=(0,))
    def CVT_grad(self, net_params, prior_params, Xc, operator):
        d = self.CVT_PNN(net_params, prior_params, Xc)
        e1 = d[...,0].reshape([self.enssiz,-1,3])
        e2 = d[...,1].reshape([self.enssiz,-1,3])
        a = d[...,2].reshape([self.enssiz,-1,3])
        fun = lambda B, x: B@x
        vfun = jx.vmap(lambda x: jx.vmap(fun)(operator, x))
        e1V_x = vfun(e1)
        e2V_x = vfun(e2)
        eV_x = jnp.concatenate([e1V_x, e2V_x], -1)
        aV_x = vfun(a)

        return eV_x, aV_x
    
    
    @partial(jx.jit, static_argnums=(0,))
    def loss_pde(self, net_params, batch, *args):
        X, Y = batch
        _, Xc, Xs, operator = X
        _, P_p = Y

        eik_loss = self.operator_net(net_params, Xc, Xs, P_p, operator)


        loss = jnp.mean(eik_loss**2, axis=(1,2))
        loss = jnp.mean(loss)
        return loss
    
    @partial(jx.jit, static_argnums=(0,))
    def AT_element(self, net_params, X):
        phi_pred = []
        for i in range(self.p_NN):
            phi_pred.append(self.AT_PNN(net_params, self.prior_params, X)[...,i].reshape([self.enssiz,-1,3]))
        phi_pred = jnp.concatenate(phi_pred,-1)
        return phi_pred
    
    
    @partial(jx.jit, static_argnums=(0,))
    def operator_net(self, net_params, Xc, X_star, P_p_predict, operator):
        
        d = self.CVT_PNN(net_params, self.prior_params, X_star)
        eV, aV = d[...,:2], d[...,2]

        D, evecs, D_canon_3D = jx.vmap(lambda ev, av:self.D_calculate(P_p_predict, ev, av, eps=1.e-9), (0, 0) )(eV, aV)

        T_n = self.AT_element(net_params, Xc)
        
        eik_loss = jx.vmap(lambda T, D :self.eikloss(T, D, operator), (0,0))(T_n, D_canon_3D)

        
        return eik_loss
    
    
    def updateSelf(self, net_params, X_star, P_p_predict, operator=None):
            d = self.CVT_PNN(net_params, self.prior_params, X_star)
            eV, aV = d[...,:2], d[...,2]

            D, evecs, D_canon_3D = jx.vmap(lambda ev, av:self.D_calculate(P_p_predict, ev, av, eps=1.e-9), (0, 0) )(eV, aV)

            self.T_pred = self.AT_PNN(net_params, self.prior_params, X_star)        
            self.CV_pred = d
            self.D = D
            self.D_canon_3D = D_canon_3D
            self.evals = eV
            self.evecs = evecs
            if operator is not None:
                self.CV_x = self.CVT_grad(net_params, self.prior_params, X_star, operator)

                T_n = self.AT_element(net_params, X_star)
                eik_loss = jx.vmap(lambda T, D :self.eikloss(T, D, operator), (0,0))(T_n, D_canon_3D)
                self.f_T_pred = eik_loss
            else:
                self.CV_x = None
                self.f_T_pred = None


    @partial(jx.jit, static_argnums=(0))
    def eiknorm(self, D, RB, u):
        U = RB@u
        return jnp.sqrt(jnp.abs(jnp.dot(U.T, jnp.dot(D, U))))
        # return jnp.sqrt(jnp.abs(jnp.dot(u.T, jnp.dot(D, u))))
    
    @partial(jx.jit, static_argnums=(0,))
    def eikloss(self, T_n, D_canon_3D, B):
        # Eikonal Residuals
        eik_loss = []
        for i in range(self.p_NN):
            _Tx = T_n[...,3*i:3*i+3]
            _eikloss = jx.vmap(self.eiknorm)(D_canon_3D,B, _Tx)
            _eikloss *=self.Tmax[i]
            _eikloss -=1
            eik_loss.append(_eikloss)
        eik_loss = jnp.transpose(jnp.stack(eik_loss,0))
        return eik_loss
