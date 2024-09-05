import numpy as np
import trimesh


#{{{ fit a 2D plane to a set of 3D points using lstsq
def __fitPlaneLSTSQ(XYZ):
    """Fit 2D plane to 3D points using LSTSQ.
       Borrowed from: https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
    """

    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z, rcond = None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)
#}}}


#{{{ extend mesh
def extendMesh(X, Tri, layers, use_average_edge = False):
    """ Extend the mesh with layers of new triangles.
        
        layers: how many layers of new elements to add
        holes: how many topological holes (e.g. mitral valve, pulmondary veins)
        use_average_edge: instead of using local edge lengths for mesh extension, use the average edge length for all edges around the holes.
                          This can help to extend meshes with jagged/sawtooth edges (but perhaps it works less well with an irregular triangulation?).
    """

    print("Extending holes on mesh with {:d} layers of triangles...".format(layers))

    # loop over the proceedure to build up layers
    for num in range(layers):
        print("Layer {:02d}/{:02d}".format(num+1, layers))

        # trimesh
        mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)

        # find edges that belong to one face only
        edges = mesh.edges_unique
        unique, counts = np.unique(mesh.faces_unique_edges, return_counts = True)
        args = np.unique(edges[unique[counts == 1]]) # this gives me the vertices in the edge

        # useful edge information 
        which_edges = np.sum( np.isin(edges[unique[counts == 1]], args), axis = 1 ) == 2 
        good_edges = edges[unique[counts == 1]][which_edges]
        edgeList_total = []

        # loop over holes in the mesh
        hole = 0
        while True:

            # let's try to group the vertices
            while True:
                vert = args[np.random.choice(np.arange(args.shape[0]))]#[0]
                if hole == 0: break

                if vert not in edgeList_total: break

            firstVert = vert  # save firstVert so we now when we get back there

            edgeList = [firstVert]

            while True:

                result = good_edges[ np.any(np.isin(good_edges, vert), axis = 1) , : ]
                #print("vert:", vert)
                #print("result:", result)

                # if anti-clockwise triangles, use next vertex along in a looping fashion (0 -> 1, 1 -> 2, 2 -> 0)...
                if len(edgeList) > 1:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) & np.all( np.isin(result, edgeList[-2], invert = True), axis = 1)].flatten() 
                else:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) ][0,:]
                #print("face_next:", face_next)

                vert_next = face_next[face_next != vert]

                if vert_next == firstVert: break

                edgeList = edgeList + [vert_next[0]]
                vert = vert_next

            edge_X = X[edgeList]

            if use_average_edge:
                av_edge_length = np.linalg.norm(edge_X[1:,:] - edge_X[:-1,:], axis = 1).mean()

            # calculate vector 'normal' which points away from holes
            c, normal = __fitPlaneLSTSQ(edge_X)

            # flip normal vector if it points inwards
            av_X = np.mean(edge_X, axis = 0)
            inward_vector = av_X - np.mean(X, axis = 0)
            if normal.dot(inward_vector) < 0: normal = normal * -1
            # add new layer of triangles to the edge in question
            # --------------------------------------------------

            new_X = np.copy(X)
            new_Tri = np.copy(Tri)

            # firstly connect new vertices to old vertices
            for ii in range(0, len(edgeList)):
                
                next_i = ii + 1 if ii < len(edgeList) - 1 else 0
                i = ii
                
                if use_average_edge:
                    newPoint = np.mean(X[edgeList][[i, next_i], :], axis = 0) \
                             + (np.tan(60*np.pi/180) * av_edge_length/2.0) * normal
                else:
                    newPoint = np.mean(X[edgeList][[i, next_i], :], axis = 0) \
                             + (np.tan(60 * np.pi/180) * np.linalg.norm(X[edgeList[next_i]] - X[edgeList[i]])/2) * normal

                new_X = np.vstack([new_X, newPoint])
                new_Tri = np.vstack([new_Tri, np.array([edgeList[i], edgeList[next_i], new_X.shape[0]-1])])

            # secondly connect new vertices to each other
            for e, ii in enumerate(range(X.shape[0], new_X.shape[0])):

                next_i = ii + 1 if ii < new_X.shape[0] - 1 else X.shape[0]
                i = ii

                if next_i == X.shape[0]:
                    ee = edgeList[0]
                else:
                    ee = edgeList[e + 1]

                new_Tri = np.vstack([new_Tri, np.array([i, next_i, ee])])


            # keep track of which vertices we have dealt with already
            edgeList_total = edgeList_total + edgeList


            # set old mesh equal to new mesh
            X = new_X
            Tri = new_Tri
            hole +=1
            if len(edgeList_total) == args.shape[0]:break
            if hole>10: break
    print(f"{hole} holes extended")
    print("Extended mesh has {:d} vertices and {:d} faces.".format(X.shape[0], Tri.shape[0]))


    # create another mesh, get edges and centroids
    # mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)
    # edges = mesh.edges_unique
    # centroids = mesh.triangles_center

    return X, Tri#, edges, centroids

