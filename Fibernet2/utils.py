import pyvista as pv
import vtk

def showMaps(self, notebook=True, off_screen = False):
    maps = self.params["maps"]

    for i in range(self.T_e.shape[-1]):
        print(f"MAP {i+1}")

        mesh = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: self.triangs}, self.points)
        plotter = pv.Plotter(notebook=notebook, off_screen=off_screen, shape=(1,2))

        mesh["Times [ms]"] = self.phis[:,i]*1e1
        plotter.subplot(0,0)
        plotter.add_mesh(mesh, show_scalar_bar=False, show_edges=not notebook, edge_color ="k", scalars="Times [ms]")
        plotter.add_points(self.points[maps[i],:], point_size=10, color="r",render_points_as_spheres=True)
        plotter.add_points(self.X_e[:,:,i], point_size=5, color="w",render_points_as_spheres=True)

        # plotter.camera_position = "xz"#[34.90010739823705, -120.06186620991983, -69.20885085393053]
        # plotter.camera.zoom(2)
        plotter.camera_position = "yz"
        plotter.camera.zoom(2)
        plotter.camera.azimuth = 180
        plotter.camera.elevation = 15
        # plotter.camera.position = [34.90010739823705, -120.06186620991983, -69.20885085393053]
        # plotter.camera.focal_point = [4.318968801436604, -9.099057508859794, 11.316830437617247]
        # plotter.camera.up = [-0.23149304827623654, 0.5287886763913683, -0.8165742491164174]
        # plotter.camera.parallel_scale = 53.2299

        plotter.subplot(0, 1)
        plotter.add_mesh(mesh, show_scalar_bar=True, show_edges=not notebook, edge_color ="k", scalars="Times [ms]")
        plotter.add_points(self.points[maps[i],:], point_size=10, color="r",render_points_as_spheres=True)
        plotter.add_points(self.X_e[:,:,i], point_size=5, color="w",render_points_as_spheres=True)
        # plotter.camera_position = "xz"#[-65.16720771058229, 103.97243886022419, 65.96968260082595]
        # plotter.camera.azimuth = 180
        # plotter.camera.zoom(2)
        # plotter.export_gltf(f"map_{i+1}.gltf")
        plotter.camera_position = "yz"
        plotter.camera.zoom(2)
        plotter.camera.elevation = -15


        # plotter.camera.position = [-65.16720771058229, 103.97243886022419, 65.96968260082595]
        # plotter.camera.focal_point = [8.569376850316937, -1.10271544535806, 8.921769885950457]
        # plotter.camera.up = [-0.31968827697870306, 0.26893578053171824, -0.9085554201655415]
        # plotter.camera.parallel_scale = 53.2299
        if notebook:
            plotter.show(jupyter_backend="static", window_size=(1000, 500))
        if not notebook:
            plotter.show(window_size=(1000, 500))#jupyter_backend="static")