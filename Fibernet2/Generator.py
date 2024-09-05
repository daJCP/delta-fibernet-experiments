from .generators.SyntheticDataGenerator3D import ActivationMapsGenerator
from .generators.DataLoaders3D import DataLoader, DataDeltaLoader

from .generators.mesh_tools import calculateSurfaceNormalsManifold, createLocalManifoldBasis, cellToPointData
from .generators.Mesh import Mesh
from .generators.meshutils import extendMesh

from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh

import pyvista as pv
import vtk
import numpy as np
import csv

from fimpy.solver import FIMPY #<- maybe implement a jax version

#https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class Generator():
    def __init__(self, params) -> None:
        self.params = params.copy()
        self.init_datasets = {"original":False, 
                              "delta":False}
        self.datasets = {"original":DataLoader,
                     "delta":DataDeltaLoader}
        
        self.fun_init_data = {"original":self._init_cartesian, 
                              "delta":self._init_delta,}
        self.init_sampling = False
        self.init_mesh = False
        self._init_mesh_info()
        
        if self.params["create_TA_maps"]:
            self.createmaps()
        else:
            self.loadmaps()
        
    def _init_mesh_info(self):      
        def Rmatrix(mesh, Nele):
            nodeCoords = mesh.verts[mesh.connectivity[Nele]]
            e1 = (nodeCoords[1,:] - nodeCoords[0,:])/np.linalg.norm(nodeCoords[1,:] - nodeCoords[0,:])
            e2 = ((nodeCoords[2,:] - nodeCoords[0,:]) - np.dot((nodeCoords[2,:] - nodeCoords[0,:]),e1)*e1)
            e2 = e2/np.linalg.norm(e2) # normalize
            R = np.vstack((e1,e2)).T
            return R
        
        vf = pv.UnstructuredGrid(self.params["geometry_file"])
        if self.params["area_multiplier"] is None:
            scale = int(np.log10(vf.points.max()-vf.points.min()))
            self.params["area_multiplier"] = 10**(-scale*2)
        self.points = vf.points*np.sqrt(self.params["area_multiplier"])
        self.triangs = vf.cells_dict[vtk.VTK_TRIANGLE]
        
        normals = calculateSurfaceNormalsManifold(self.points, self.triangs)
        # Creation of smooth basis mesh
        if self.params["geometry_file_smooth"] is None:
            smooth_basis_mesh = self.createbasis()
        else:
            smooth_basis_mesh = pv.UnstructuredGrid(self.params["geometry_file_smooth"])
        try:
            smooth_basis = smooth_basis_mesh.cell_data["vf_smooth"]
        except Exception as e:
            print(e)
            smooth_basis_mesh = self.createbasis()
            smooth_basis = smooth_basis_mesh.cell_data["vf_smooth"]

        if not np.allclose(np.sum(smooth_basis * normals, axis=-1),0.,atol=2e-5):
            smooth_basis = smooth_basis - normals * np.sum(smooth_basis * normals, axis=-1)[:,None]
            smooth_basis /= np.linalg.norm(smooth_basis, axis=-1)[:,None]
        P = createLocalManifoldBasis(self.points, self.triangs, smooth_basis)

        # Check measurement points are subset of the collocation points
        self.kdtree_X = cKDTree(self.points)
        indices = self.kdtree_X.query(self.points)[1]

        self.P_p = cellToPointData(self.points, self.triangs, P.reshape([-1, 9])).reshape([-1, 3, 3]).astype(np.float32)        
        self.P_p_predict = np.array(self.P_p[indices])        
        self.smooth_basis = smooth_basis

        self.init_mesh = True
        
        m = Mesh(verts=self.points, connectivity=self.triangs)
        
        RBs = [] #NEW
        Area = 0
        for e in range(m.connectivity.shape[0]):
            B, J = m.Bmatrix(e)
            R = Rmatrix(m, e)
            RB = (np.dot(R,B)/J)
            RBs.append(RB)
            Area+=J/2
        RBs = np.array(RBs)
        self.area = Area
        self.operator = RBs

    @timeit
    def createbasis(self):
           # Create smooth base
            datanew = pv.UnstructuredGrid(self.params["geometry_file"])

            n = datanew.extract_surface().compute_normals()["Normals"]
            to_surf = lambda vects: vects-n*np.sum(vects * n, axis=-1)[:,None]

            triangs = datanew.cells_dict[5]
            D = np.eye(3, dtype=np.float32)[np.newaxis] * np.ones(triangs.shape[0])[...,None, None] #Isotropic propagation
            fim = FIMPY.create_fim_solver(datanew.points, triangs, D, device="cpu")

            maps = 1

            first_point = datanew.points.shape[0]-1
            x0 = [first_point]
            for i in range(maps-1):
                dist = fim.comp_fim(x0, [0.0]*(i + 1))
                # x0.append(np.argmax(dist.get()))
                x0.append(np.argmax(dist))
            x0_vals = np.array([0.]*len(x0))

            #Create a FIM solver, by default the GPU solver will be called with the active list
            phis = []
            for i, _ in enumerate(x0_vals):
                # phis.append(fim.comp_fim(x0[i], x0_vals[i]).get())
                phis.append(fim.comp_fim(x0[i], x0_vals[i]))
            phis = np.stack(phis, axis=-1)

            datanew["isotro"] = phis
            gra = datanew.compute_derivative("isotro").point_data_to_cell_data()["gradient"]
            gra = to_surf(gra)
            gra /=np.linalg.norm(gra, axis=1)[:,None]
            datanew["vf_smooth"] = gra
            return datanew
    
    @timeit
    def createmaps(self):
        # These data files are 3D meshes with data
        gen = ActivationMapsGenerator(vtk_file = self.params["geometry_file"], 
                                            maps=self.params["maps"], 
                                            seed=self.params["gen_key_1"])
        self.phis = gen.phis*np.sqrt(self.params["area_multiplier"])#*1e1
        self.D = gen.evecs #canonico con el que se compara
        self.D_n = gen.D_n # el usado para generar los mapas
        self.params["maps"] = gen.maps
        
    def loadmaps(self):
        def read_csv_map(dir):
            with open(dir, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                res = []
                for i, row in enumerate(spamreader):
                    if i ==0: continue
                    res.append(row)
            res = np.array(res).astype(float)
            return res[:,:3], res[:,-1]
        self.phis = np.zeros((self.points.shape[0],1))
        res = [read_csv_map(e) for e in self.params["maps_files"]]
        
        self.X_e = [r[0]*np.sqrt(self.params["area_multiplier"]) for r in res]
        self.T_e = [r[1] for r in res]        
        self.init_sampling = True


    def dataset(self, info):
        type_model = info["type_model"]
        if self.params["create_TA_maps"]:
            # On demand initialization
            self.sampling_TA(info)
        if not self.init_datasets[type_model]:
            self.fun_init_data[type_model]()
        paquete = self.generatePack(info)

        return self.datasets[type_model](*paquete, **info)

    def sampling_TA(self, info):
        rng_agent = np.random.default_rng(info["gen_key_2"])
        ppm = int(self.area*info["density"])
        maps = len(self.params["maps"])
        x_e = []
        t_e = []
        for i in range(maps):
            m_ind = rng_agent.choice(self.points.shape[0],[ppm,],replace=False)
            phi = self.phis[:,i]
            m_mask = np.zeros(self.points.shape[0], dtype=bool)
            m_mask[m_ind] = True
            x_e.append(self.points[m_mask])
            t_e.append(phi[m_mask][...,np.newaxis])
        
        X_e = np.squeeze(np.vstack(x_e))
        X_e = [X_e[ppm*i:ppm*(i+1)] for i in range(maps)]
        X_e = np.stack(X_e, axis=2)

        T_e = np.vstack(t_e)
        T_e += info["noise_ms"] * rng_agent.standard_normal(T_e.shape)*1e-1
        T_e = [T_e[ppm*i:ppm*(i+1)] for i in range(maps)]
        T_e = np.stack(T_e, axis=1)[:,:,0]   

        self.X_e = [X_e[...,i] for i in range(maps)]
        self.T_e = [T_e[...,i] for i in range(maps)]
        self.init_sampling = True
    
    def _init_cartesian(self):
        if self.init_mesh and self.init_sampling:
            self.init_datasets["original"] = True

    def _init_eigvalvects(self, fun):
        
        m = Mesh(verts=self.points, connectivity=self.triangs)
        
        eigvals, eigvecs = fun(mesh=m)

        return eigvals, eigvecs
    
    @timeit
    def _init_delta(self):
        
        def computeEig(mesh, layers=self.params["extend_n_layers"]):
            points = mesh.verts
            triangs = mesh.connectivity
            if layers>0:
                Ncut = points.shape[0]
                print(f"Original mesh has {Ncut} vertices and {triangs.shape[0]} faces.")
                newPoints, newTriangs = extendMesh(points, triangs, layers= layers)
                newMesh = Mesh(verts=newPoints, connectivity=newTriangs)  
                print('Computing Laplacian')
                K, M = newMesh.computeLaplacian()
                print('Computing eigen values')
                eigvals, eigvecs = eigsh(
                    K,
                    self.params["N_eig_max"],
                    M,
                    which="LM",
                    sigma=0
                    )
                return eigvals[:Ncut,], eigvecs[:Ncut,]
            else:
                print('Computing Laplacian')
                K, M = mesh.computeLaplacian()
                print('Computing eigen values')
                
                eigvals, eigvecs = eigsh(
                    K,
                    self.params["N_eig_max"],
                    M,
                    which="LM",
                    sigma=0
                    )
                return eigvals, eigvecs

        if self.params["input_precalculated"] is None:
            eigvals, eigvecs = self._init_eigvalvects(computeEig)
        else:
            eigvecs = np.load(self.params["input_precalculated"])["eigvecs"]
            eigvals = np.load(self.params["input_precalculated"])["eigvals"]
        
        if not eigvecs.shape[0] == self.points.shape[0]:
            message = f"Every point ({self.points.shape[0]}) is not asociated with a eigenfunct ({eigvecs.shape[0]})"
            raise Exception(message)

        self.f_full = eigvals
        self.F_full = eigvecs
        
        if self.init_sampling:
            self.init_datasets["delta"] = True


    def generatePack(self, info):
        type_model = info["type_model"]
        if type_model=="original":
            pack = [
                self.triangs, 
                self.points, 
                self.P_p_predict, 
                self.X_e, 
                self.T_e
                ]
        elif type_model=="delta":
            F = self.F_full[:,:info["n_eigs"]]
            F_e = []
            for i in range(len(self.X_e)):
                F_e.append(F[self.kdtree_X.query(self.X_e[i])[1]])
            pack = [#triangs, F, P_p_predict, F_e, T_e, operator
                self.triangs, 
                F, 
                self.P_p_predict, 
                F_e,
                self.T_e,
                self.operator
                ]
        else:
            raise Exception(f"type_model:{type_model} not valid")
        
        return pack

