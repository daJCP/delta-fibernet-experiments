import jax as jx
import jax.numpy as jnp
from functools import partial

from Fibernet2.FiberNet import FiberNet


class QAlphaFiberNet(FiberNet):
    """Keeps the base FiberNet CVT_NN contract: [e1, e2, alpha]."""

    @partial(jx.jit, static_argnums=(0,))
    def Q_from_alpha(self, net_params, input):
        alpha = self.CVT_NN(net_params, input)[:, 2:3]
        beta = jnp.sqrt(jnp.maximum(1.0 - alpha**2, 1e-9))
        q1 = 2.0 * alpha**2 - 1.0
        q2 = 2.0 * alpha * beta
        return jnp.concatenate([q1, q2], axis=-1)

    @partial(jx.jit, static_argnums=(0,))
    def GRAD_CVT(self, net_params, input):
        e1V_fn = lambda x: self.CVT_NN(net_params, x)[:, 0]
        e2V_fn = lambda x: self.CVT_NN(net_params, x)[:, 1]
        q1_fn = lambda x: self.Q_from_alpha(net_params, x)[:, 0]
        q2_fn = lambda x: self.Q_from_alpha(net_params, x)[:, 1]

        e1V_x = self.jax_gradient(e1V_fn, input)
        e2V_x = self.jax_gradient(e2V_fn, input)
        q1_x = self.jax_gradient(q1_fn, input)
        q2_x = self.jax_gradient(q2_fn, input)

        eV_x = jnp.vstack([e1V_x, e2V_x])
        q_x = jnp.vstack([q1_x, q2_x])
        return eV_x, q_x


class QDirectFiberNet(FiberNet):
    """Uses the q_direct CVT_NN contract: [e1, e2, q1, q2]."""

    @partial(jx.jit, static_argnums=(0,))
    def CVT_NN(self, net_params, input):
        input = self.scaler(input)
        d_pre = self.CV_apply(net_params[0], input)
        vel = self.C * jx.nn.sigmoid(d_pre[:, :2])
        q_raw = d_pre[:, 2:4]
        q_norm = jnp.sqrt(jnp.sum(q_raw**2, axis=-1, keepdims=True) + 1e-6)
        q = q_raw / q_norm
        return jnp.concatenate([vel, q], axis=-1)

    @partial(jx.jit, static_argnums=(0,))
    def D_calculate_q(self, P_p_loc, eV, q):
        matMulProdSum = lambda A, B: jnp.einsum("...xy,...yz->...xz", A, B)
        eigenDecompProd = lambda A, B: matMulProdSum(
            matMulProdSum(A, B), jnp.transpose(A, (0, 2, 1))
        )

        e1 = eV[:, 0:1]
        e2 = eV[:, 1:2]
        q1 = q[:, 0:1]
        q2 = q[:, 1:2]

        trace = e1 + e2
        anisotropy = e1 - e2
        d11 = 0.5 * (trace + anisotropy * q1)
        d22 = 0.5 * (trace - anisotropy * q1)
        d12 = 0.5 * anisotropy * q2

        D = jnp.reshape(jnp.stack([d11, d12, d12, d22], axis=-1), [-1, 2, 2])

        zeros = jnp.zeros_like(d11)
        D_local_3D = jnp.reshape(
            jnp.stack(
                [
                    d11,
                    d12,
                    zeros,
                    d12,
                    d22,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                ],
                axis=-1,
            ),
            [-1, 3, 3],
        )

        D_canon_3D = eigenDecompProd(P_p_loc, D_local_3D)
        evecs = P_p_loc

        return D, evecs, D_canon_3D

    @partial(jx.jit, static_argnums=(0,))
    def operator_net(self, net_params, X_star, P_p_predict):
        d = self.CVT_NN(net_params, X_star)
        eV = d[:, :2]
        q = d[:, 2:4]
        D, evecs, D_canon_3D = self.D_calculate_q(P_p_predict, eV, q)

        T_x = self.GRAD_AT(net_params, X_star)
        eik_loss = self.eikloss(T_x, D_canon_3D)

        return eik_loss

    def updateSelf(self, net_params, X_star, P_p_predict):
        d = self.CVT_NN(net_params, X_star)
        eV = d[:, :2]
        q = d[:, 2:4]
        D, evecs, D_canon_3D = self.D_calculate_q(P_p_predict, eV, q)

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

    @partial(jx.jit, static_argnums=(0,))
    def GRAD_CVT(self, net_params, input):
        e1V_fn = lambda x: self.CVT_NN(net_params, x)[:, 0]
        e2V_fn = lambda x: self.CVT_NN(net_params, x)[:, 1]
        q1_fn = lambda x: self.CVT_NN(net_params, x)[:, 2]
        q2_fn = lambda x: self.CVT_NN(net_params, x)[:, 3]

        e1V_x = self.jax_gradient(e1V_fn, input)
        e2V_x = self.jax_gradient(e2V_fn, input)
        q1_x = self.jax_gradient(q1_fn, input)
        q2_x = self.jax_gradient(q2_fn, input)

        eV_x = jnp.vstack([e1V_x, e2V_x])
        q_x = jnp.vstack([q1_x, q2_x])
        return eV_x, q_x
