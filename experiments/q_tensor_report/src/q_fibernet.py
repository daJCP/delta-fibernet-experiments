import jax as jx
import jax.numpy as jnp
from functools import partial

from Fibernet2.FiberNet import FiberNet


class QAlphaFiberNet(FiberNet):
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
    @partial(jx.jit, static_argnums=(0,))
    def CVT_NN(self, net_params, input):
        input = self.scaler(input)
        d_pre = self.CV_apply(net_params[0], input)
        vel = self.C * jx.nn.sigmoid(d_pre[:, :2])
        q_raw = d_pre[:, 2:4]
        q_norm = jnp.linalg.norm(q_raw, axis=-1, keepdims=True)
        q = q_raw / jnp.maximum(q_norm, 1e-9)
        theta = 0.5 * jnp.arctan2(q[:, 1:2], q[:, 0:1])
        alpha = jnp.cos(theta)
        return jnp.concatenate([vel, alpha, q], axis=-1)

    @partial(jx.jit, static_argnums=(0,))
    def GRAD_CVT(self, net_params, input):
        e1V_fn = lambda x: self.CVT_NN(net_params, x)[:, 0]
        e2V_fn = lambda x: self.CVT_NN(net_params, x)[:, 1]
        q1_fn = lambda x: self.CVT_NN(net_params, x)[:, 3]
        q2_fn = lambda x: self.CVT_NN(net_params, x)[:, 4]

        e1V_x = self.jax_gradient(e1V_fn, input)
        e2V_x = self.jax_gradient(e2V_fn, input)
        q1_x = self.jax_gradient(q1_fn, input)
        q2_x = self.jax_gradient(q2_fn, input)

        eV_x = jnp.vstack([e1V_x, e2V_x])
        q_x = jnp.vstack([q1_x, q2_x])
        return eV_x, q_x
