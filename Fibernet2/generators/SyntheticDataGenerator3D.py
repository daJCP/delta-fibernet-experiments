import pyvista as pv
import vtk
import numpy as np
from fimpy.solver import FIMPY

from .mesh_tools import calculateSurfaceNormalsManifold


class SyntheticDataGenerator3D:
  """
  Create a set of cardiac activation maps from a geometry file with fiber orientations

  Parameters:
  vtk_file: geometry file in .vtk format with a cell data field called "fibers" (a vector)
  maps: int or vector: if int, number of activation maps desired; if vector, ids of init sites
  ppm: int of the number of sample points per map
  noise: A factor in milliseconds by which a standard normal distribution of noise 
  is applied to the activation maps
  x0: initial sites 
  """

  def __init__(self, vtk_file, maps=1, ppm=100, noise=0., seed=0):
    rng_agent = np.random.default_rng(seed)
    vf = pv.UnstructuredGrid(vtk_file)
    self.points = vf.points
    self.triangs = vf.cells_dict[vtk.VTK_TRIANGLE]
    self.l = vf.cell_data["fibers"]

    self.n = calculateSurfaceNormalsManifold(self.points, self.triangs)
    self.l = self.l - self.n * np.sum(self.l * self.n, axis=-1, keepdims=True)
    self.l /= np.linalg.norm(self.l, axis=-1, keepdims=True)
    self.t = np.cross(self.l, self.n, axis=-1)
    self.t /= np.linalg.norm(self.t, axis=-1, keepdims=True)
    D_init = (1. ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + 1. ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1. ** 2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_init = 0.5*(D_init + np.transpose(D_init, axes=(0, 2, 1)))

    fim = FIMPY.create_fim_solver(self.points, self.triangs, D_init, device='cpu', use_active_list=False)
    if not np.isscalar(maps):
        x0 = maps
        maps = len(maps)
    else:
        first_point = rng_agent.choice(self.points.shape[0])
        x0 = [first_point]
        for i in range(maps):
            dist = fim.comp_fim(x0, [0.0]*(i + 1))
            x0.append(np.argmax(dist))

    x0_vals = np.zeros(maps)
    D_n = (.6 ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + .4 ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1e-2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_n = 0.5*(D_n + np.transpose(D_n, axes=(0, 2, 1)))
    self.evecs = np.linalg.eigh(D_n)[1]

    phis = []
    mm = []
    x_e = []
    t_e = []
    inds = [0]
    m_ind = rng_agent.choice(self.points.shape[0],[ppm,maps],replace=False)
    for i in range(maps):
      phi = fim.comp_fim(x0[i], x0_vals[i], D_n)
      m_mask = np.zeros(self.points.shape[0], dtype=bool)
      m_mask[m_ind[:,i]] = True
      phis.append(phi)
      mm.append(m_mask)
      x_e.append(self.points[m_mask])
      t_e.append(phi[m_mask][...,np.newaxis])
      inds.append(len(phi[m_mask]) + inds[-1])
    
    self.x_e = np.squeeze(np.vstack(x_e))
    self.mm = np.stack(mm, axis=-1)
    self.t_e = np.vstack(t_e)
    self.noise = noise * rng_agent.standard_normal(self.t_e.shape)
    self.phis = np.stack(phis, axis=-1)
    self.inds = np.stack(inds)

    self.ppm = ppm
    self.maps = maps
  
  def get_values(self):
    return self.phis, self.t_e+self.noise, self.x_e, self.mm, self.evecs, self.inds
  
  def update(self, noise_ms, gen_key, **kwargs):
    print(f"UPDATE SEED: {gen_key} with Noise: {noise_ms}.")
    rng_agent = np.random.default_rng(gen_key)
    phis = []
    mm = []
    x_e = []
    t_e = []
    inds = [0]
    m_ind = rng_agent.choice(self.points.shape[0],[self.ppm,self.maps],replace=False)
    for i in range(self.maps):
      phi = self.phis[:,i]
      m_mask = np.zeros(self.points.shape[0], dtype=bool)
      m_mask[m_ind[:,i]] = True
      phis.append(phi)
      mm.append(m_mask)
      x_e.append(self.points[m_mask])
      t_e.append(phi[m_mask][...,np.newaxis])
      inds.append(len(phi[m_mask]) + inds[-1])
    
    self.x_e = np.squeeze(np.vstack(x_e))
    self.mm = np.stack(mm, axis=-1)
    self.t_e = np.vstack(t_e)
    self.noise = noise_ms * rng_agent.standard_normal(self.t_e.shape)
    self.phis = np.stack(phis, axis=-1)
    self.inds = np.stack(inds)

class ActivationMapsGenerator:
  """
  Create a set of cardiac activation maps from a geometry file with fiber orientations

  Parameters:
  vtk_file: geometry file in .vtk format with a cell data field called "fibers" (a vector)
  maps: int or vector: if int, number of activation maps desired; if vector, ids of init sites
  x0: initial sites 
  """

  def __init__(self, vtk_file, maps=1, seed=0):
    rng_agent = np.random.default_rng(seed)
    if type(vtk_file)==str:
        vf = pv.UnstructuredGrid(vtk_file)
    else:
       vf = vtk_file
    self.points = vf.points
    self.triangs = vf.cells_dict[vtk.VTK_TRIANGLE]

    self.l = vf.cell_data["fibers"]

    self.n = calculateSurfaceNormalsManifold(self.points, self.triangs)
    self.l = self.l - self.n * np.sum(self.l * self.n, axis=-1, keepdims=True)
    self.l /= np.linalg.norm(self.l, axis=-1, keepdims=True)
    self.t = np.cross(self.l, self.n, axis=-1)
    self.t /= np.linalg.norm(self.t, axis=-1, keepdims=True)
    D_init = (1. ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + 1. ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1. ** 2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_init = 0.5*(D_init + np.transpose(D_init, axes=(0, 2, 1)))

    fim = FIMPY.create_fim_solver(self.points, self.triangs, D_init, device='cpu', use_active_list=False)
    if not np.isscalar(maps):
        x0 = maps
        maps = len(maps)
    else:
        first_point = rng_agent.choice(self.points.shape[0])
        x0 = [first_point]
        for i in range(maps-1):
            dist = fim.comp_fim(x0, [0.0]*(i + 1))
            x0.append(np.argmax(dist))

    x0_vals = np.zeros(maps)
    D_n = (.6 ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + .4 ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1e-2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_n = 0.5*(D_n + np.transpose(D_n, axes=(0, 2, 1)))
    self.evecs = np.linalg.eigh(D_n)[1]
    self.D_n = D_n

    phis = []
    for i in range(maps):
      phi = fim.comp_fim(x0[i], x0_vals[i], D_n)
      phis.append(phi)
    
    self.phis = np.stack(phis, axis=-1)

    self.maps = x0

