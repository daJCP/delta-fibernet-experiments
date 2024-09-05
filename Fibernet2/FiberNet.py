import jax as jx
import jax.numpy as jnp
from functools import partial

from jaxpinns.base import PINN as jxPINN


class FiberNet(jxPINN):
    # Initialize the class
    def __init__(self, 
                 dataset,
                 CVmax=1.0, 
                 lambda_df=1., 
                 lambda_pde=1e-4,
                 lambda_tve=1e-2, 
                 lambda_tva=1e-9
                 ):
        super().__init__(mu_X=0, sigma_X=0)
        
        self.dataset = dataset

        # Assign class parameters
        self.Tmax = self.dataset.Tmax

        # Assign constants
        self.C = CVmax
        self.alpha_e = lambda_tve
        self.alpha = lambda_tva
        self.lambda_DF = lambda_df
        self.lambda_PDE = lambda_pde

        self.num_loss_terms = 3 
        
    # TV-Huber regularization function 
    @partial(jx.jit, static_argnums=(0))
    def TVHuber(self, nabla_x, huber_norm_eps):
        nabla_x_norm_squared = jnp.sum(nabla_x**2, axis=-1, keepdims=True)
        nabla_x_norm = jnp.sqrt(nabla_x_norm_squared)
        nabla_x_reg_term = jnp.where(nabla_x_norm <= huber_norm_eps,
                0.5/huber_norm_eps * nabla_x_norm_squared,
                (jnp.sqrt(jnp.maximum(nabla_x_norm_squared, huber_norm_eps**2))
                    - 0.5 * huber_norm_eps))

        return nabla_x_reg_term, nabla_x_norm_squared, nabla_x_norm


    @partial(jx.jit, static_argnums=(0))
    def eiknorm(self, BB, u):
        # we take the abs to avoid the generation of imaginary numbers
        return jnp.sqrt(jnp.abs(jnp.dot(u, jnp.dot(BB, u))))

    def architecture(self, CV_net, CV_args, 
                        maps_number, map_net, map_args, 
                        init_key = jx.random.PRNGKey(0)):
        ''' Network initialization and evaluation functions '''
        self.CV_init, self.CV_apply = CV_net(*CV_args)
        CV_params = self.CV_init(init_key)

        self.p_NN = maps_number
        maps_init = []
        maps_apply = []
        maps_params = []
        for m_number in range(maps_number):
            m_init, m_apply = map_net(*map_args)
            m_params = m_init(init_key)
            maps_init.append(m_init)
            maps_apply.append(m_apply)
            maps_params.append(m_params)
        self.maps_init, self.maps_apply = maps_init, maps_apply

        self.net_params = (CV_params, *maps_params)
        
    @partial(jx.jit, static_argnums=(0))
    def scaler(self, X):
        return (X-self.dataset.lb)/(self.dataset.ub-self.dataset.lb)
    
    # Conduction velocity tensor nn
    @partial(jx.jit, static_argnums=(0,))
    def CVT_NN(self, net_params, input):
        input = self.scaler(input)
        d_pre = self.CV_apply(net_params[0], input)
        vel = self.C * (jx.nn.sigmoid(d_pre[:,:2]))
        ang = jnp.tanh(d_pre[:,2]).reshape([-1,1])
        d = jnp.concatenate([vel, ang], axis=-1)
        return d
    
    # Activation time nn (MAPS)
    @partial(jx.jit, static_argnums=(0,))
    def AT_NN(self, net_params, input):
        input = self.scaler(input)
        maps_params = net_params[1:]
        phi = []
        for m_number in range(self.p_NN):
            map_param = maps_params[m_number]
            map = self.maps_apply[m_number](map_param, input)
            phi.append(map)
        phi = jnp.concatenate(phi,-1)
        return phi

    # Gradient of Activation time for EIKONAL EQ
    @partial(jx.jit, static_argnums=(0,))
    def GRAD_AT(self, net_params, input):
        phi_x = []
        for m_number in range(self.p_NN):
            map_fn = lambda x:self.AT_NN(net_params, x)[:,m_number]
            map_x = self.jax_gradient(map_fn, input)
            phi_x.append(map_x.reshape([-1,3]))
        phi_x = jnp.concatenate(phi_x,-1)
        return phi_x

    # Gradient of CV for Huber Regularization
    @partial(jx.jit, static_argnums=(0,))
    def GRAD_CVT(self, net_params, input):
        e1V_fn = lambda x: self.CVT_NN(net_params,x)[:,0]
        e2V_fn = lambda x: self.CVT_NN(net_params,x)[:,1]
        aV_fn = lambda x: self.CVT_NN(net_params,x)[:,2]
        
        e1V_x = self.jax_gradient(e1V_fn, input)
        e2V_x = self.jax_gradient(e2V_fn, input)
        aV_x = self.jax_gradient(aV_fn, input)

        eV_x = jnp.vstack([e1V_x, e2V_x])
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
            phi_pred =  self.AT_NN(net_params, self.prior_params, input)[...,i]
            phi_square.append(jnp.mean((phi_e[i]-phi_pred)**2))
    
        loss = jnp.mean(jnp.array(phi_square))
        return loss

    # Loss of regularization
    @partial(jx.jit, static_argnums=(0,))
    def loss_regu(self, net_params, batch, *args):
        X, Y = batch
        _, Xs = X
        eV_x, aV_x = self.GRAD_CVT(net_params, Xs)
        # Huber Regularization
        eV_TV = self.TVHuber(eV_x, 1e-3)[0]
        aV_TV = self.TVHuber(aV_x, 1e-3)[0]

        loss1 = jnp.mean(eV_TV)
        loss2 = jnp.mean(aV_TV)

        return loss1, loss2
    
    @partial(jx.jit, static_argnums=(0,))
    def loss_pde(self, net_params, batch, *args):
        X, Y = batch
        _, Xs = X
        _, P_p = Y

        eik_loss = self.operator_net(net_params, Xs, P_p)

        loss = jnp.mean(eik_loss**2)
        return loss


    def loss(self, net_params, batch, *args):
        self.net_params = net_params
        
        A = self.loss_data(net_params, batch, *args)
        B = self.loss_pde(net_params, batch, *args)
        C = self.loss_regu(net_params, batch, *args)

        L = self.lambda_DF*A
        L +=self.lambda_PDE*B
        L +=self.alpha_e*C[0] +self.alpha*C[1]
        return L

    @partial(jx.jit, static_argnums=(0,))
    def D_calculate(self, P_p_loc, eV, aV, eps=1.e-9):
        #D = R*S@R.T on 2 p
        #D = R@P*S@R.T on 3 cartesian
        matMulProdSum = lambda A, B: jnp.einsum('...xy,...yz->...xz', A, B)
        eigenDecompProd = lambda A, B: matMulProdSum( matMulProdSum(A, B),jnp.transpose(A, (0, 2, 1)))
        eV_flat = jnp.reshape(eV, [-1])
        aV_flat = jnp.reshape(aV, [-1])

        zero_e = jnp.zeros_like(eV_flat[0::2])
        aVr = jnp.sqrt(jnp.maximum(1-aV_flat**2,eps))
        eVM_mat = jnp.reshape(jnp.stack([eV_flat[0::2], zero_e, zero_e, eV_flat[1::2]], axis=-1), [-1, 2, 2])
        aVM_mat = jnp.reshape(jnp.stack([aV_flat, -1.*aVr,aVr, aV_flat], axis=-1), [-1, 2, 2])
        # En coordenadas locales, sistema p
        D = eigenDecompProd(aVM_mat, eVM_mat)
        # Pasa a coordenadas globales, sistema cartesiano
        zeros = jnp.zeros_like(aVM_mat[..., 0, 0])
        ones = jnp.ones_like(aVM_mat[..., 0, 0])
        aVM_3D = jnp.reshape(jnp.stack([aVM_mat[..., 0, 0], aVM_mat[..., 0, 1], zeros,
                                    aVM_mat[..., 1, 0], aVM_mat[..., 1, 1], zeros,
                                    zeros, zeros, ones], axis=-1), [-1, 3, 3])

        evecs = matMulProdSum(P_p_loc, aVM_3D)

        evals3D = jnp.reshape(jnp.stack([eVM_mat[..., 0, 0], zeros, zeros,
                                        zeros, eVM_mat[..., 1, 1], zeros,
                                        zeros, zeros, zeros], axis=-1), [-1, 3, 3])

        D_canon_3D = eigenDecompProd(evecs, evals3D)

        return D, evecs, D_canon_3D
    
    @partial(jx.jit, static_argnums=(0,))
    def eikloss(self, T_x, D_canon_3D):
        # Eikonal Residuals
        eik_loss = []
        for i in range(self.p_NN):
            _Tx = T_x[...,3*i:3*i+3]
            _eikloss = jx.vmap(self.eiknorm)(D_canon_3D, _Tx)
            _eikloss *=self.Tmax[i]
            _eikloss -=1
            eik_loss.append(_eikloss)
        eik_loss = jnp.transpose(jnp.stack(eik_loss,0))
        return eik_loss

    @partial(jx.jit, static_argnums=(0,))
    def operator_net(self, net_params, X_star, P_p_predict):

        d = self.CVT_NN(net_params, X_star)
        eV, aV = d[:,:2], d[:,2]
        D, evecs, D_canon_3D = self.D_calculate(P_p_predict, eV, aV,eps=1.e-9)

        T_x = self.GRAD_AT(net_params, X_star)
        
        eik_loss = self.eikloss(T_x, D_canon_3D)

        return eik_loss
    
    def updateSelf(self, net_params, X_star, P_p_predict):
        d = self.CVT_NN(net_params, X_star)
        eV, aV = d[:,:2], d[:,2]
        D, evecs, D_canon_3D = self.D_calculate(P_p_predict, eV, aV,eps=1.e-9)

        T_x = self.GRAD_AT(net_params, X_star)
        
        eik_loss = self.eikloss(T_x, D_canon_3D)

        self.T_pred = self.AT_NN(net_params, X_star)        
        self.CV_pred = d
        self.CV_x = self.GRAD_CVT(net_params, X_star)
        self.D = D
        self.D_canon_3D = D_canon_3D
        self.evals = eV
        self.evecs = evecs
        self.f_T_pred = eik_loss
    
    def predict(self, X_star, P_p_predict ):
        params = self.get_params(self.opt_state)
        
        self.updateSelf(params, X_star, P_p_predict)

        result = self.Tmax*self.T_pred, self.CV_pred, self.CV_x, self.D, self.D_canon_3D, self.evals, self.evecs, self.f_T_pred

        return result
    
    def jax_gradient(self, fun, x):
        y, vjp_fn = jx.vjp(fun, x)
        return vjp_fn(jnp.ones(y.shape))[0]
    
    
class DeltaFiberNet(FiberNet):
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
        _, Xc,_, operator = X

        eV_x, aV_x = self.CVT_grad(net_params, Xc, operator)

        # Huber Regularization
        eV_TV = self.TVHuber(eV_x, 1e-3)[0]
        aV_TV = self.TVHuber(aV_x, 1e-3)[0]

        loss1 = jnp.mean(eV_TV)
        loss2 = jnp.mean(aV_TV)

        return loss1, loss2
    
    @partial(jx.jit, static_argnums=(0,))
    def CVT_grad(self, net_params, Xc, operator):
        d = self.CVT_NN(net_params, Xc)
        e1 = d[:,0].reshape([-1,3])
        e2 = d[:,1].reshape([-1,3])
        a = d[:,2].reshape([-1,3])
        fun = lambda B, x: B@x
        vfun = jx.vmap(fun)
        e1V_x = vfun(operator, e1)
        e2V_x = vfun(operator, e2)
        eV_x = jnp.vstack([e1V_x, e2V_x])
        aV_x = vfun(operator, a)
        return eV_x, aV_x
    
    @partial(jx.jit, static_argnums=(0,))
    def loss_pde(self, net_params, batch, *args):
        X, Y = batch
        _, Xc, Xs, operator, _ = X
        _, P_p, _ = Y

        eik_loss = self.operator_net(net_params, Xc, Xs, P_p, operator)

        loss = jnp.mean(eik_loss**2)
        return loss
    
    @partial(jx.jit, static_argnums=(0,))
    def AT_element(self, net_params, X):
        phi_pred = []
        for i in range(self.p_NN):
            phi_pred.append(self.AT_NN(net_params, X)[:,i].reshape([-1,3]))
        phi_pred = jnp.concatenate(phi_pred,-1)
        return phi_pred
    
    
    @partial(jx.jit, static_argnums=(0,))
    def operator_net(self, net_params, Xc, X_star, P_p_predict, operator):
        d = self.CVT_NN(net_params, X_star)
        eV, aV = d[:,:2], d[:,2]
        D, evecs, D_canon_3D = self.D_calculate(P_p_predict, eV, aV,eps=1.e-9)
        
        T_n = self.AT_element(net_params, Xc)
        
        eik_loss = self.eikloss(T_n, D_canon_3D, operator)
        
        return eik_loss
    
    
    def updateSelf(self, net_params, X_star, P_p_predict, operator=None):
            d = self.CVT_NN(net_params, X_star)
            eV, aV = d[:,:2], d[:,2]
            D, evecs, D_canon_3D = self.D_calculate(P_p_predict, eV, aV, eps=1.e-9)
            
            self.T_pred = self.AT_NN(net_params, X_star)        
            self.CV_pred = d
            self.D = D
            self.D_canon_3D = D_canon_3D
            self.evals = eV
            self.evecs = evecs
            if operator is not None:
                self.CV_x = self.CVT_grad(net_params, X_star, operator)

                T_n = self.AT_element(net_params, X_star)
                eik_loss = self.eikloss(T_n, D_canon_3D, operator)
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