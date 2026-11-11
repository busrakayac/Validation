from nutils.mesh import gmsh
from nutils import cli, export, function
from nutils.solver import System
from nutils.SI import Length, Density, Viscosity, Velocity, Time, Pressure, Acceleration
from nutils.expression_v2 import Namespace
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import treelog as log
import numpy
from scipy.interpolate import LinearNDInterpolator
import subprocess
import meshio

#Calculates the particle's deformation (omega) and orientation(theta) from its current boundary coordinates.
#This method uses principal component analysis (PCA) to estimate the particle axes.
# It does not fit an ellipse or calculate the exact maximum and minimum diameters. Its results therefore depend on the distribution of sampled boundary points.
def particle_shape_from_boundary(xb_m):
    pts = numpy.asarray(xb_m, dtype=float)
    pts = pts[numpy.isfinite(pts).all(axis=1)]

    if pts.shape[0] < 3:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    center = pts.mean(axis=0)
    q = pts - center

    cov = q.T @ q / pts.shape[0]
    eigvals, eigvecs = numpy.linalg.eigh(cov)
    order = numpy.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    proj = q @ eigvecs
    L_particle_m = proj[:, 0].max() - proj[:, 0].min()
    B_particle_m = proj[:, 1].max() - proj[:, 1].min()
    orientation_rad = numpy.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    if B_particle_m > L_particle_m:
        L_particle_m, B_particle_m = B_particle_m, L_particle_m
        orientation_rad += numpy.pi / 2
        orientation_rad = (orientation_rad + numpy.pi/2) % numpy.pi - numpy.pi/2

    deformation_index = (
        (L_particle_m - B_particle_m) / (L_particle_m + B_particle_m)
        if (L_particle_m + B_particle_m) > 0 else numpy.nan
    )
    aspect_ratio = B_particle_m / L_particle_m

    return L_particle_m, B_particle_m, deformation_index, aspect_ratio, orientation_rad

#Vector dot product: a.b = a1b1 + a2b2.
#The function explicitly computes the scalar product while preserving dimensional consistency.
def _dot2(a, b):
    """Contract two length-2 vectors.
    nutils.SI.Quantity does not implement .sum() or numpy.einsum, so an explicit component sum is the only way to contract dimensional arrays."""
    return a[0] * b[0] + a[1] * b[1]


#Eliminates extremely small numerical errors in geometric coordinates before generating a Gmsh mesh.
#The function replaces sufficiently small coordinates with exact zeros to improve geometric robustness.
def _snap_small(v, scale, rel=1e-12):
    """Snap |v| below rel*scale to exactly 0.
    Boundary coordinates that should be exactly zero often carry floating-point residues (e.g. 3.7e-21). Written out at %.16e these become literals that some Gmsh kernels treat as degenerate."""
    return 0.0 if abs(v) < rel * scale else v


#Generates a new Gmsh geometry file (.geo) for the fluid domain only, using the current deformed particle boundary and writes the mesh generation instructions.
#The current particle boundary is represented using quadratic Bézier curves: B(s)=(1-s)^2 P_0 + 2(1-s)s P_c +s^2 P_1
#It preserves the quadratic representation of the deformed particle boundary.
def write_fluid_remesh_geo(boundary_xy_m, domain, filename, bezier_edges=None):
    boundary_xy_m = numpy.asarray(boundary_xy_m, dtype=float)

    L = float(domain.channel_length / 'm')
    W = float(domain.channel_length / 'm')
    R = float(domain.cylinder_radius / 'm')

    h_particle = float(domain.elemsize / 'm')
    h_far = h_particle * domain.coarsening
    Lambda0 = 0.8
    a0 = R / numpy.sqrt(Lambda0)
    b0 = R * numpy.sqrt(Lambda0)
    Lgrow = 10.0 * a0

    first_point = 1001
    first_line  = 2001

    particle_point_ids = [first_point + i for i in range(len(boundary_xy_m))]
    particle_line_ids  = [first_line  + i for i in range(len(boundary_xy_m))]

    with open(filename, 'w') as f:

        f.write('Mesh.MshFileVersion = 2.2;\n')
    
        # Square outer boundary: [-L,L] × [-L,L]
        f.write(f'L = {L:.16e};\n')
        f.write('Point(1) = {-L,-L,0,1};\n')
        f.write('Point(2) = { L,-L,0,1};\n')
        f.write('Point(3) = { L, L,0,1};\n')
        f.write('Point(4) = {-L, L,0,1};\n')
        for i in range(1, 5):
            f.write(f'Line({i}) = {{{i},{i % 4 + 1}}};\n')
        f.write('Curve Loop(100) = {1,2,3,4};\n')

        # Particle boundary points
        for pid, (x, y) in zip(particle_point_ids, boundary_xy_m):
            x = _snap_small(x, L); y = _snap_small(y, L)
            f.write(f'Point({pid}) = {{{x:.16e},{y:.16e},0,{h_particle:.16e}}};\n')
        f.write('\n')

        n = len(particle_point_ids)

        if bezier_edges is None:
            # Fallback: straight segments (low geometric fidelity).
            for i in range(n):
                p1 = particle_point_ids[i]
                p2 = particle_point_ids[(i + 1) % n]
                f.write(f'Line({particle_line_ids[i]}) = {{{p1},{p2}}};\n')
        else:
            ctrl_first = first_point + n
            for i, (p0_xy, pmid_xy, p1_xy) in enumerate(bezier_edges):
                p0_xy   = numpy.asarray(p0_xy,   dtype=float)
                pmid_xy = numpy.asarray(pmid_xy, dtype=float)
                p1_xy   = numpy.asarray(p1_xy,   dtype=float)
                cx, cy  = 2.0 * pmid_xy - 0.5 * (p0_xy + p1_xy)
                cx = _snap_small(cx, L); cy = _snap_small(cy, L)
                cid = ctrl_first + i
                f.write(f'Point({cid}) = {{{cx:.16e},{cy:.16e},0,{h_particle:.16e}}};\n')
            f.write('\n')
            for i in range(n):
                p1 = particle_point_ids[i]
                p2 = particle_point_ids[(i + 1) % n]
                cid = ctrl_first + i
                f.write(f'Bezier({particle_line_ids[i]}) = {{{p1},{cid},{p2}}};\n')

        line_string = ','.join(str(i) for i in particle_line_ids)

        f.write(f'\nCurve Loop(500) = {{{line_string}}};\n')
        f.write('Plane Surface(400) = {100,500};\n\n')
        f.write(f'Transfinite Curve {{{line_string}}} = 2;\n')
        
        # Physical groups — same names as the original mesh
        f.write('Physical Surface("fluid") = {400};\n')
        f.write(f'Physical Line("cylinder") = {{{line_string}}};\n')
        f.write('Physical Line("outer") = {1,2,3,4};\n\n')

        # Mesh sizing
        f.write('Mesh.MeshSizeFromPoints = 0;\n')
        f.write('Mesh.MeshSizeFromCurvature = 0;\n')
        f.write('Mesh.MeshSizeExtendFromBoundary = 0;\n')
        f.write('Mesh.Algorithm = 6;\n')
        f.write('Mesh.Smoothing = 30;\n')
        f.write('Mesh.Optimize = 0;\n')
        f.write('Mesh.OptimizeNetgen = 0;\n\n')

        # Exponential refinement around the particle
        f.write('Field[1] = Distance;\n')
        f.write(f'Field[1].CurvesList = {{{line_string}}};\n')
        
        f.write('\nField[2] = MathEval;\n')
        f.write(
            'Field[2].F = '
            f'"Min({h_far:.16e},'
            f'{h_particle:.16e}*Exp('
            f'Log({h_far:.16e}/{h_particle:.16e})*'
            f'Max(F1-{b0:.16e},0)/{Lgrow:.16e}))";\n'
        )
        
        f.write('\nBackground Field = 2;\n')


def remesh_fluid(current_t_s, xb_current_m, domain, ns, solid, fluid, dynamic, args, topo, fluid_state, remesh_index):
    log.info(f'=== REMESHING FLUID at t={current_t_s:.6f} s ===')

    # 1. Extract the fluid solution and Newmark history before remeshing
    # Before changing the fluid mesh, the algorithm must preserve the information required to continue the simulation.
    # This includes:
    #    The current physical fluid solution.
    #    The current mesh motion.
    #    The relative fluid motion.
    #    The stored Newmark history needed to reconstruct consistent temporal derivatives.
    # All these quantities are evaluated on the existing mesh and stored as numerical arrays.

    #Retrieve the existing fluid fields
    old_topo_fluid = fluid_state['topo_fluid'] #existing fluid topology.
    x_field        = fluid_state['x_field']    #current physical coordinates.
    v_field        = fluid_state['v_field']    #mesh velocity.
    a_field        = fluid_state['a_field']    #mesh acceleration.
    urel_field     = fluid_state['urel_field'] #relative fluid velocity
    arel_field     = fluid_state['arel_field'] #relative fluid acceleration
    p_field        = fluid_state['p_field']    #fluid pressure
    d_field        = fluid_state['d_field']    #displacement field used for the existing fluid mesh.
    d_name         = fluid_state['d_name']     #argument names associated with its displacement history.
    a0dt2_key      = fluid_state['a0dt2_key']  #argument names associated with its acceleration history.  

    #Recover the Newmark history variables
    #The code creates fourth-order Gauss quadrature points on the existing fluid mesh:
    old_sample = old_topo_fluid.sample('gauss', 4)

    #For the mesh displacement, the code retrieves the history. 
    #"function.replace_arguments" creates a new symbolic expression by replacing the coefficient argument of an existing finite-element field with the corresponding history argument.
    a0dt2_expr     = function.replace_arguments(d_field, [(d_name, a0dt2_key)])
    u0_expr        = function.replace_arguments(urel_field, [('u', 'u0')]) #uses the same finite-element basis as urel_field, but evaluates it using the stored coefficients args['u0'] instead of args['u']
    a0dt_urel_expr = function.replace_arguments(urel_field, [('u', 'a0δt')])
    
    #Evaluate the old solution
    #The function.eval() operation evaluates all the selected fields at the old mesh quadrature points using the current solution dictionary args.
    #These are values evaluated at quadrature points, not the original finite-element coefficient vectors.
    (old_x_phys, old_v, old_a, old_urel, old_arel, old_p, old_a0dt2, old_u0, old_a0dt_urel) = function.eval([
            old_sample.bind(x_field),
            old_sample.bind(v_field),
            old_sample.bind(a_field),
            old_sample.bind(urel_field),
            old_sample.bind(arel_field),
            old_sample.bind(p_field),
            old_sample.bind(a0dt2_expr),
            old_sample.bind(u0_expr),
            old_sample.bind(a0dt_urel_expr)], arguments=args) 
    
    #Reconstruct the physical fluid velocity
    old_uf = old_v + old_urel   

    #Convert quantities into numerical arrays
    old_x_m       = numpy.asarray(old_x_phys / 'm', dtype=float)
    old_uf_ms     = numpy.asarray(old_uf / 'm/s', dtype=float)
    old_p_Pa      = numpy.asarray(old_p / 'Pa', dtype=float)
    old_vm_ms     = numpy.asarray(old_v / 'm/s', dtype=float)
    old_am_ms2    = numpy.asarray(old_a / 'm/s2', dtype=float)
    old_arel_ms2  = numpy.asarray(old_arel / 'm/s2', dtype=float)
    old_a0dt2_m   = numpy.asarray(old_a0dt2 / 'm', dtype=float)
    old_urel_ms   = numpy.asarray(old_urel / 'm/s', dtype=float)
    old_u0_ms     = numpy.asarray(old_u0 / 'm/s', dtype=float)
    old_a0dt_urel_ms = numpy.asarray(old_a0dt_urel / 'm/s', dtype=float)

    # 2. Extract the solid interface and prepare its geometry for remeshing

    #Record the current solid interface
    solid_cyl_boundary  = topo['solid'].boundary['cylinder']
    solid_cyl_gauss     = solid_cyl_boundary.sample('gauss', 4)
    solid_cyl_pts_now_m = numpy.asarray(function.eval(solid_cyl_gauss.bind(ns.x), arguments=args) / 'm', dtype=float) #Current physical position
    d_s_remesh_const_m  = numpy.asarray(function.eval(solid_cyl_gauss.bind(ns.d), arguments=args) / 'm', dtype=float) #Current solid displacement

    #Evaluate the old fluid traction
    #The code therefore uses Topology.locate() to find points on the old fluid interface corresponding to the current solid interface quadrature points.
    #The traction is evaluated using the old fluid solution. This operation is diagnostic: it does not apply forces to the solid or enforce traction continuity.
    old_traction_located = old_topo_fluid.boundary['cylinder'].locate(x_field / Length('1m'), solid_cyl_pts_now_m, tol=float(domain.cylinder_radius / 'm') * 1e-4, arguments=args, skip_missing=False)
    
    ns_told = Namespace()
    ns_told.x = x_field
    ns_told.define_for('x', gradient='∇', normal='n')
    ns_told.u = v_field + urel_field
    ns_told.p = p_field
    ns_told.μf = fluid.viscosity
    
    ns_told.tpres_i = '-p n_i' # n : outward normal of the fluid domain.
    ns_told.tvisc_i = 'μf (∇_j(u_i) + ∇_i(u_j)) n_j'
    ns_told.ttot_i  = 'tpres_i + tvisc_i'
    
    told_pres, told_visc, told_tot = function.eval( #pressure traction, viscous traction, total traction
        [old_traction_located.bind(ns_told.tpres),
         old_traction_located.bind(ns_told.tvisc),
         old_traction_located.bind(ns_told.ttot)], arguments=args)

    #Verify mesh–solid velocity compatibility before remeshing 
    # v_m = v_s at the interface
    #If a stored interface mapping "dm_zipped_interface" exists, it is reused. Otherwise, Topology.locate() identifies corresponding points on the old fluid boundary.
    if fluid_state.get('dm_zipped_interface') is not None:
        old_zipped = fluid_state['dm_zipped_interface']
        vm_old_g, vs_old_g = function.eval(
            [old_zipped.bind(v_field), 
             old_zipped.bind(ns.v)],
            arguments=args)
    else:
        old_interface_located = old_topo_fluid.boundary['cylinder'].locate(
            x_field / Length('1m'),
            solid_cyl_pts_now_m,
            tol=float(domain.cylinder_radius / 'm') * 1e-4,
            arguments=args,
            skip_missing=False)
        #Velocities evaluated on the existing meshes, before remeshing
        vm_old_g = function.eval(old_interface_located.bind(v_field), arguments=args) #old fluid mesh velocity evaluated at the fluid–solid interface.
        vs_old_g = function.eval(solid_cyl_gauss.bind(ns.v), arguments=args)          #solid velocity evaluated at the same corresponding interface locations.
        
    dv_old = numpy.linalg.norm(numpy.asarray((vm_old_g - vs_old_g) / 'm/s'), axis=-1)
    vs_old_ref = max(numpy.linalg.norm(numpy.asarray(vs_old_g / 'm/s'), axis=-1).max(), 1e-30)

    log.info(
        f'[INTERFACE BEFORE REMESH] ||v_m-v_s|| '
        f'max={dv_old.max():.6e} m/s, ' #max abs error
        f'rel={dv_old.max()/vs_old_ref:.6e}') #relative error


    #Extract the quadratic solid boundary
    solid_cyl_bez3 = solid_cyl_boundary.sample('bezier', 3) # extract the three physical points of each element.
    solid_cyl_bez3_pts_m = numpy.asarray(function.eval(solid_cyl_bez3.bind(ns.x), arguments=args) / 'm', dtype=float)
    solid_p2_edges = [solid_cyl_bez3_pts_m[solid_cyl_bez3.getindex(ielem)] for ielem in range(solid_cyl_bez3.nelems)]  #resulting element coordinates


    #Verify that the particle boundary forms a closed loop, the last element must also connect to the first.
    def _edges_form_closed_loop(edges, tol):
        for e_this, e_next in zip(edges, edges[1:] + edges[:1]):
            if numpy.linalg.norm(e_this[2] - e_next[0]) > tol:
                return False
        return True

    _loop_tol = float(domain.cylinder_radius / 'm') * 1e-6
    if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
        log.info('[REMESH] solid P2 edges are not in traversing order; reordering by endpoint chaining.')
        remaining = list(solid_p2_edges)
        ordered   = [remaining.pop(0)]
        
        while remaining:
            tail = ordered[-1][2]
            j = min(range(len(remaining)),
                    key=lambda k: min(numpy.linalg.norm(remaining[k][0] - tail), numpy.linalg.norm(remaining[k][2] - tail)))
            e = remaining.pop(j)
            
            if (numpy.linalg.norm(e[2] - tail) < numpy.linalg.norm(e[0] - tail)): e = e[::-1] # flip edge so it continues the chain
            ordered.append(e)
        solid_p2_edges = ordered
        
        if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
            raise RuntimeError(
                'solid cylinder boundary edges do not form a closed loop; '
                'cannot build a consistent fresh fluid boundary.')
        log.info('[REMESH] edge reordering succeeded; closed loop verified.')
    else:
        log.info('[REMESH] solid P2 edge connectivity verified (closed loop).')
# the old fluid solution and solid geometry are known, but no new fluid mesh has been generated and no interface conditions have been modified.
#_____________________________________________________________________________________________________________________________________________

    # 3.Generate the new fluid mesh and match its boundary to the solid
    #The procedure is:
        # 1. Generate a new P2 fluid mesh using the deformed solid boundary.
        # 2. Match each fluid boundary element to its corresponding solid element.
        # 3. Correct the fluid midpoint nodes to reproduce the quadratic solid geometry.
        # 4. Import the corrected fluid mesh into a new Nutils coordinate space.
        # 5. Verify that the two interface geometries coincide numerically.
    
    # The current deformed solid boundary was extracted in the previous block as a collection of quadratic elements:
    xb_ordered = numpy.array([e[0] for e in solid_p2_edges], dtype=float) #the starting points of these elements
    geo_file   = Path(f'remesh_fluid_{current_t_s:.6f}s.geo')
    write_fluid_remesh_geo(xb_ordered, domain, geo_file, bezier_edges=solid_p2_edges) #creates a Gmsh geometry file using the complete quadratic boundary information stored in solid_p2_edges

    fluid_space_name = f'Xfluid{remesh_index}'
    msh_file = geo_file.with_suffix('.msh')

    subprocess.run(['gmsh', str(geo_file), '-2', '-order', '2','-format', 'msh2', '-o', str(msh_file)], check=True)

    mesh = meshio.read(msh_file)
    cylinder_id = int(mesh.field_data['cylinder'][0])

    # The generated Gmsh mesh contains quadratic boundary elements of type line3.
    # Each element contains three nodes, ordered by Gmsh as: [endpoint 0, endpoint 1, midpoint]

    # It first verifies that the number of fluid boundary elements equals the number of solid boundary elements.
    # Then, for every solid element, it searches for the corresponding fluid element by comparing their endpoints.
    # Because the two meshes may have different element orientations, the code considers both possible endpoint arrangements.
    fluid_edge_nodes = []
    for cells, physical in zip(
            mesh.cells, mesh.cell_data['gmsh:physical']):
        if cells.type == 'line3':
            fluid_edge_nodes.extend(nodes for nodes, group in zip(cells.data, physical) if group == cylinder_id)

    if len(fluid_edge_nodes) != len(solid_p2_edges):
        raise RuntimeError(
            f'Cylinder edge count differs: fluid={len(fluid_edge_nodes)}, '
            f'solid={len(solid_p2_edges)}')

    unused = set(range(len(fluid_edge_nodes)))
    endpoint_tol = 1e-10 * float(domain.cylinder_radius / 'm')

    for solid_edge in solid_p2_edges:
        def endpoint_error(j):
            nodes = fluid_edge_nodes[j]
            p0, p1 = mesh.points[nodes[:2], :2]
            return min(
                max(numpy.linalg.norm(p0 - solid_edge[0]), numpy.linalg.norm(p1 - solid_edge[2])),
                max(numpy.linalg.norm(p1 - solid_edge[0]), numpy.linalg.norm(p0 - solid_edge[2])))

        j = min(unused, key=endpoint_error)
        error = endpoint_error(j)
        if error > endpoint_tol:
            raise RuntimeError(
                f'Cannot match fluid edge to solid edge: '
                f'endpoint error={error:.3e} m')
        
        #Correct the fluid boundary midpoint  to ensure that quadratic fluid and solid elements describe exactly the same curved boundary.
        #Only the fluid boundary midpoint coordinates are modified. The solid mesh is not changed.
        unused.remove(j)
        midpoint_node = fluid_edge_nodes[j][2]
        mesh.points[midpoint_node, :2] = solid_edge[1]

    meshio.write(msh_file, mesh, file_format='gmsh22', binary=False)
    #The corrected mesh is imported using:
    #- `new_topo`: new fluid topology, including elements and boundary connectivity.
    #- `new_geom_bare`: new fluid reference coordinates, stored as numerical coordinates without explicit physical units.
    new_topo, new_geom_bare = gmsh(msh_file, space=fluid_space_name) 

    # Verify the new fluid–solid interface geometry
    #After importing the corrected mesh, the code extracts the quadratic boundary elements again.
    #It compares every solid boundary element with its corresponding new fluid element.
    fluid_bz3 = new_topo['fluid'].boundary['cylinder'].sample('bezier', 3)
    fluid_pts = numpy.asarray(function.eval(fluid_bz3.bind(new_geom_bare)), dtype=float)
    fluid_edges = [fluid_pts[fluid_bz3.getindex(i)] for i in range(fluid_bz3.nelems)]

    if len(fluid_edges) != len(solid_p2_edges):
        raise RuntimeError(
            f'Interface edge count changed: solid={len(solid_p2_edges)}, '
            f'fluid={len(fluid_edges)}')

    unused = set(range(len(fluid_edges)))
    endpoint_errors = []
    midpoint_errors = []
    worst_edge = None
    worst_midpoint_error = -1.0

    for solid_edge in solid_p2_edges:
        def endpoint_error(j):
            fluid_edge = fluid_edges[j]
            forward = max(
                numpy.linalg.norm(solid_edge[0] - fluid_edge[0]),
                numpy.linalg.norm(solid_edge[2] - fluid_edge[2]))
            reverse = max(
                numpy.linalg.norm(solid_edge[0] - fluid_edge[2]),
                numpy.linalg.norm(solid_edge[2] - fluid_edge[0]))
            return min(forward, reverse)

        j = min(unused, key=endpoint_error)
        unused.remove(j)
        endpoint_errors.append(endpoint_error(j))
        midpoint_errors.append(numpy.linalg.norm(solid_edge[1] - fluid_edges[j][1]))

        if midpoint_errors[-1] > worst_midpoint_error:
            worst_midpoint_error = midpoint_errors[-1]
            worst_edge = (len(midpoint_errors) - 1, j, solid_edge.copy(), fluid_edges[j].copy())

    log.info(
        f'[INTERFACE EDGE GEOMETRY] '
        f'endpoint max={max(endpoint_errors):.6e} m, '
        f'midpoint max={max(midpoint_errors):.6e} m, '
        f'endpoint RMS={numpy.sqrt(numpy.mean(numpy.square(endpoint_errors))):.6e} m, '
        f'midpoint RMS={numpy.sqrt(numpy.mean(numpy.square(midpoint_errors))):.6e} m')

    solid_i, fluid_i, solid_edge, fluid_edge = worst_edge
    log.info(
        f'[WORST INTERFACE EDGE] solid edge={solid_i}, fluid edge={fluid_i}; '
        f'solid P0={solid_edge[0]}, Pmid={solid_edge[1]}, P1={solid_edge[2]}; '
        f'fluid P0={fluid_edge[0]}, Pmid={fluid_edge[1]}, P1={fluid_edge[2]}')
    
    solid_midpoint = solid_edge[1]
    midpoint_located = new_topo['fluid'].boundary['cylinder'].locate(
        new_geom_bare,
        solid_midpoint[None, :],
        tol=3e-7,
        skip_missing=False)
    
    nearest_fluid_point = numpy.asarray(function.eval(midpoint_located.bind(new_geom_bare)), dtype=float)[0]

    log.info(
        f'[WORST EDGE NEAREST POINT] '
        f'solid midpoint={solid_midpoint}, '
        f'nearest fluid point={nearest_fluid_point}, '
        f'distance={numpy.linalg.norm(nearest_fluid_point - solid_midpoint):.6e} m')
    
    new_geom = new_geom_bare * Length('m')
    log.info(f'New fluid mesh generated from {geo_file} (space={fluid_space_name})')

#The new fluid mesh and its reference geometry are available.
#No fluid solution has been transferred yet, and no fluid–solid interface conditions have been imposed on the new mesh.
#_____________________________________________________________________________________________________________________________________________

    # 4.Define the fluid unknowns on the new mesh
    #The new fluid mesh has different finite-element degrees of freedom from the old mesh.
    #Therefore, the algorithm must redefine the fluid unknowns on the new topology.
    #Three primary fields are introduced: dm, u, p

    new_ns = ns   #Reuse the existing namespace, preserving the solid-related definitions while replacing the fluid-related fields.

    #The new fluid mesh displacement is defined using quadratic (P2) finite elements: The corresponding unknown is dm.
    dm_basis = new_topo['fluid'].basis('std', degree=2)
    new_ns.dm = function.field('dm', dm_basis, shape=(2,)) * domain.cylinder_radius

    if dynamic:
        new_ns.vm, new_ns.am = dynamic.newmark_defo_named(new_ns.dm, name='dm', d0_name='dm0', u0dt_name='vm0δt', a0dt2_name='am0δt2')
    else:
        new_ns.vm = Velocity.wrap(function.zeros((2,)))
        new_ns.am = Acceleration.wrap(function.zeros((2,)))
         
    #The relative fluid velocity is also represented using quadratic (P2) finite elements: The unknown is named u, and its coefficients are multiplied by the characteristic fluid velocity U.
    new_ns.urel = new_topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
    if dynamic:
        new_ns.arel = dynamic.newmark_velo(new_ns.urel)
        new_ns.u_i  = 'vm_i + urel_i'
    else:
        new_ns.u_i  = 'urel_i'
        
    #The pressure is represented using linear (P1) finite elements:
    new_ns.p = (new_topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius)

    #The code creates two test-function fields:
        #- `utest`: associated with the relative fluid velocity `u`.
        #- `ptest`: associated with the fluid pressure `p`.
    new_ns.utest = function.replace_arguments(new_ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
    new_ns.ptest = function.replace_arguments(new_ns.p,    'p:ptest') / fluid.viscosity / fluid.velocity**2
    #The functions are constructed using function.replace_arguments(), which replaces the unknown coefficient argument with the corresponding test-function argument.
#At this stage, the code only defines symbolic finite-element fields. Their coefficients have not yet been initialized using the old fluid solution, and no governing equations have been solved.
#_____________________________________________________________________________________________________________________________________________

    # 5. Define the fluid ALE equations and mesh-extension model 
    #After remeshing, two different mathematical descriptions are required.
        # The physical fluid problem (ns_f) uses the current moving geometry to calculate fluid acceleration, velocity gradients, stresses, and traction.
        # The artificial mesh-extension problem (ns_m) uses the new reference geometry to determine how the mesh deforms as the particle continues moving.
        # Both descriptions depend on the same mesh displacement field dm, but they use different gradient operators.

    ns_f        = Namespace() #physical fluid formulation in the current ALE configuration.
    ns_f.δ      = new_ns.δ
    ns_f.ρf     = new_ns.ρf
    ns_f.μf     = new_ns.μf
    ns_f.xfnew  = new_geom
    ns_f.dm     = new_ns.dm
    ns_f.xm_i   = 'xfnew_i + dm_i'

    ns_f.define_for('xm', gradient='∇', normal='n', jacobians=('dV', 'dS')) # all fluid quantities are evaluated in the current configuration rather than the reference configuration.

    ns_f.urel   = new_ns.urel
    ns_f.p      = new_ns.p
    ns_f.vm     = new_ns.vm
    ns_f.am     = new_ns.am

    if dynamic:
        ns_f.arel    = new_ns.arel
        ns_f.u_i     = 'vm_i + urel_i'
        ns_f.DuDt_i  = 'am_i + arel_i + ∇_j(u_i) urel_j'
    else:
        ns_f.u_i     = 'urel_i'
        ns_f.DuDt_i  = '∇_j(u_i) u_j'

    ns_f.edot_ij = '.5 (∇_j(u_i) + ∇_i(u_j))'
    ns_f.σ_ij    = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij'

    ns_f.traction_i = 'σ_ij n_j'   
    ns_f.tpres_i = '-p n_i'
    ns_f.tvisc_i = 'μf (∇_j(u_i) + ∇_i(u_j)) n_j'
    new_ns.tsolid_i = 'σcauchy_ij n_j'

    ns_f.utest   = new_ns.utest
    ns_f.ptest   = new_ns.ptest
    ns_f.dtest   = new_ns.dtest


    ns_m        = Namespace() #artificial mesh-extension formulation relative to the new fluid reference configuration. It describes how the fluid mesh deforms relative to the reference configuration created at remeshing.
    ns_m.δ      = new_ns.δ
    ns_m.xfnew  = new_geom

    ns_m.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))

    ns_m.dm     = new_ns.dm
    ns_m.xm_i   = 'xfnew_i + dm_i'
    ns_m.Fm_ij  = '∇ref_j(xm_i)'
    ns_m.Jm     = numpy.linalg.det(ns_m.Fm) #measures the local change in mesh element area. This resembles a hyperelastic constitutive relation but is not the stress of a physical material.
    ns_m.Fminv  = numpy.linalg.inv(ns_m.Fm)

    ns_m.Pmesh_ij = '2 (Fm_ij - Fminv_ji)' #artificial mesh stress
    ns_m.dmtest = function.replace_arguments(new_ns.dm, 'dm:dmtest') / (domain.cylinder_radius**2 * Pressure('1Pa')) #the test function associated with mesh displacement
#At this stage, the code defines the mathematical quantities required for both problems. Neither the fluid equations nor the mesh-extension equations have been solved yet.
#_____________________________________________________________________________________________________________________________________________

    # 6. Rebuild the governing equations and fluid–solid interface coupling

    # This block first rebuilds the governing equations on the remeshed domain.
    # It then establishes corresponding interface points between the independent fluid and solid meshes.
    # Finally, it introduces two different Lagrange multipliers: 
        #lamf: Physical fluid–solid coupling and interface reaction (urel=0)
        #lam: Artificial mesh–solid displacement compatibility (d_m = d_s - d_s(t_r))

    #Solid momentum equation
    res_d_internal = topo['solid'].integral('∇ref_j(dtest_i) P_ij dVref' @ new_ns, degree=4) #internal elastic force.
    res_d_inertia  = topo['solid'].integral('dtest_i ρs (a_i - g_i) dVref' @ new_ns, degree=4) #solid inertia and body forces
    res_d_cmforce  = topo['solid'].integral('dtest_i cmforce_i dVref' @ new_ns, degree=4) #reaction associated with the center position constrained
    new_res = res_d_internal + res_d_inertia + res_d_cmforce
    new_res += topo['solid'].integral('cmtest_i x_i dVref' @ new_ns, degree=4) / domain.cylinder_radius**3 #to keep the particle fixed

    #Artificial mesh-extension equation
    res_mesh_bulk = Pressure('1Pa') * new_topo['fluid'].integral('∇ref_j(dmtest_i) Pmesh_ij dVref' @ ns_m, degree=4)
    new_res += res_mesh_bulk

    #Fluid momentum equation
    res_fluid_am   = new_topo['fluid'].integral('utest_i ρf am_i dV' @ ns_f, degree=4)
    res_fluid_arel = new_topo['fluid'].integral('utest_i ρf arel_i dV' @ ns_f, degree=4)
    res_fluid_acc  = res_fluid_am + res_fluid_arel

    res_fluid_conv = new_topo['fluid'].integral('utest_i ρf ∇_j(u_i) urel_j dV' @ ns_f, degree=4)
    res_fluid_visc = new_topo['fluid'].integral('∇_j(utest_i) μf (∇_j(u_i) + ∇_i(u_j)) dV' @ ns_f, degree=4)
    res_fluid_pres = new_topo['fluid'].integral('-∇_j(utest_i) p δ_ij dV' @ ns_f, degree=4)
    
    res_fluid_mom = res_fluid_acc + res_fluid_conv + res_fluid_visc + res_fluid_pres
    res_fluid_inc = new_topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns_f, degree=4)
    
    new_res += res_fluid_mom
    new_res += res_fluid_inc

    #find corresponding points between the solid boundary and the newly generated fluid boundary, so that the two independent meshes can be coupled.
    locate_tol_m = float(domain.cylinder_radius / 'm') * 1e-4
    try:
        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False)
    except Exception:
        log.info(
            f'[REMESH] locate failed at tol={locate_tol_m:.3e} m; '
            f'retrying with a looser tolerance.')
        
        locate_tol_m *= 100.
        
        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False)
        #returns a Nutils sample containing the locations on the new fluid boundary corresponding to the supplied solid interface coordinates.

    log.info(f'[REMESH] FSI traction coupling located at tol={locate_tol_m:.3e} m')

    #zip() combines 3 SOLID MESH POINTS and 3 LOCATED FLUID MESH POINTS into a paired interface sample.
    #Consequently, we can evaluate a solid field and a fluid field together in the same interface integral. e.g. new_ns.dtest and new_ns.lamf
    zipped_traction = solid_cyl_gauss.zip(fluid_traction_located)
    x_fluid_loc_m = numpy.asarray(function.eval(fluid_traction_located.bind(new_geom_bare)), dtype=float)
    loc_err = numpy.linalg.norm(x_fluid_loc_m - solid_cyl_pts_now_m, axis=-1)

    log.info(
            f'[TRACTION LOCATE] position error '
            f'max={loc_err.max():.6e} m, '
            f'RMS={numpy.sqrt(numpy.mean(loc_err**2)):.6e} m, '
            f'/R={loc_err.max()/float(domain.cylinder_radius/"m"):.6e}')

    if loc_err.max() > 1e-4 * float(domain.cylinder_radius / 'm'):
        raise RuntimeError(f'Fresh interface differs from solid: max={loc_err.max():.3e} m')

    #The code evaluates the P2 mesh displacement basis on the interface, identifies which DOFs are active there, and uses their indices to construct the interface multiplier spaces. (e.g. lamf)
    #It does not impose any constraint or solve any equation yet.

    #Identify the fluid mesh basis functions at the interface
    Bf4 = function.eval(zipped_traction.bind((new_ns.dm / Length('1m')).derivative('dm'))) #tells us which DOFs influence the displacement at the interface.

    #Find which DOFs contribute to the interface
        #Each column of Bf corresponds to one displacement DOF.
        #If a column is zero, that DOF has no influence on the displacement at the sampled interface points.
        #If a column is nonzero, its basis function contributes to the interface displacement.
    Bf = Bf4.reshape(Bf4.shape[0] * Bf4.shape[1], -1)
    dm_norm = numpy.linalg.norm(Bf, axis=0)

    i_dm_gamma = numpy.flatnonzero(dm_norm > 1e-12 * dm_norm.max()) #vector interface degrees of freedom.
    i_dm_gamma_scalar = numpy.unique(i_dm_gamma // 2) #corresponding scalar P2 basis-function indices.
    i_dm_expected = numpy.sort((2 * i_dm_gamma_scalar[:, None] + numpy.arange(2)).ravel())

    if not numpy.array_equal(numpy.sort(i_dm_gamma), i_dm_expected):
        raise RuntimeError('Unexpected dm interface DOF ordering.')

    log.info(
        f'[INTERFACE TRACE BASIS] '
        f'scalar P2 trace dofs={len(i_dm_gamma_scalar)}'
    )
    # This procedure identifies the interface finite-element basis functions.

    # Introduce the physical fluid–solid coupling through "lamf"
    #The same unknown provides both:
        # 1. The force required to enforce the kinematic constraint.
        # 2. The equal-and-opposite interface force contributions in the fluid and solid equations.
    # This is how the fluid transfers its physical force to the solid after remeshing.
    fluid_stress_scale = fluid.viscosity * fluid.velocity / domain.cylinder_radius
    lamf_basis = dm_basis[i_dm_gamma_scalar]
    new_ns.lamf = (function.field('lamf', lamf_basis, shape=(2,)) * fluid_stress_scale) #This multiplier is represented using the quadratic interface basis identified previously.
    new_ns.lamftest = (function.replace_arguments(new_ns.lamf, 'lamf:lamftest') / (fluid_stress_scale * fluid.velocity * domain.cylinder_radius))
    # Fluid equation: -∫ utest · lamf
    res_fluid_lamf = -zipped_traction.integral(_dot2(new_ns.utest, new_ns.lamf) * new_ns.dS)
    # Solid equation: +∫ dtest · lamf
    res_solid_lamf = zipped_traction.integral(_dot2(new_ns.dtest, new_ns.lamf) * new_ns.dS)
    # No-slip constraint: urel = 0 on the interface
    res_lamf_constraint = zipped_traction.integral(_dot2(new_ns.lamftest, new_ns.urel) * new_ns.dS)
    new_res += res_fluid_lamf
    new_res += res_solid_lamf
    new_res += res_lamf_constraint

    #Construct a diagnostic fluid reaction map
    test_unit = Pressure('1Pa') * Length('1m')
    dtest_dimless = new_ns.dtest * test_unit
    utest_dimless = new_ns.utest * test_unit

    #Evaluate the derivatives of the solid and fluid test functions with respect to their coefficients:
    Bd4 = function.eval(zipped_traction.bind(dtest_dimless.derivative('dtest')))
    Bd = Bd4.reshape(Bd4.shape[0] * Bd4.shape[1], -1)
    
    Bu4 = function.eval(zipped_traction.bind(utest_dimless.derivative('utest')))
    Bu = Bu4.reshape(Bu4.shape[0] * Bu4.shape[1], -1)

    Bd_norm = numpy.linalg.norm(Bd, axis=0)
    Bu_norm = numpy.linalg.norm(Bu, axis=0)
    
    i_d_gamma = numpy.flatnonzero(Bd_norm > 1e-12 * Bd_norm.max()) #solid interface DOFs.
    i_u_gamma = numpy.flatnonzero(Bu_norm > 1e-12 * Bu_norm.max()) #fluid velocity interface DOFs.

    # The corresponding test functions cannot necessarily be matched by simply identifying DOF indices.
    # Instead, a least-squares transformation is calculated:
    Tdu = numpy.linalg.lstsq(Bu[:, i_u_gamma], Bd[:, i_d_gamma], rcond=None)[0] #This transformation approximately represents the solid interface test functions using the fluid interface test functions: B_u T_du = B_d
    #This relationship is useful for comparing interface reactions before and after remeshing.
    trace_error = numpy.linalg.norm(Bu[:, i_u_gamma] @ Tdu - Bd[:, i_d_gamma]) / numpy.linalg.norm(Bd[:, i_d_gamma])
    log.info(f'[FLUID REACTION MAP] trace relerror={trace_error:.6e}')

    #Store the solid displacement at the remeshing instant
    d_s_remesh_coeffs = numpy.array(args['d'], copy=True) #the solid displacement at the remeshing time: d_s(t_r)
    d_s_remesh_field = function.replace_arguments(new_ns.d, {'d': d_s_remesh_coeffs}) #reconstructs this fixed displacement field 
    dm_interface_target_expr = (new_ns.d - d_s_remesh_field) 

    #Enforce the displacement compatibility condition between the new fluid mesh and the persistent solid boundary.
    #The solid displacement remains an unknown of the coupled system. 
    #Therefore, the target mesh displacement evolves during Newton iterations. 
    #This is why the interface displacement is enforced through the coupled multiplier constraint instead of being fixed to its previous value.
    lam_basis = dm_basis[i_dm_gamma_scalar] 
    new_ns.lam = (function.field('lam', lam_basis, shape=(2,)) * solid.shear_modulus)
    new_ns.lamtest = function.replace_arguments(new_ns.lam, 'lam:lamtest') / (solid.shear_modulus * domain.cylinder_radius**2)

    res_mesh_lam = zipped_traction.integral(_dot2(ns_m.dmtest, new_ns.lam) * new_ns.dS)
    res_mesh_constraint = zipped_traction.integral(_dot2(new_ns.lamtest, new_ns.dm - dm_interface_target_expr) * new_ns.dS)
    
    new_res += res_mesh_lam
    new_res += res_mesh_constraint
#At the end of this block, the bulk governing equations and the weak interface coupling terms are assembled in new_res. However, the coupled nonlinear system has not yet been solved.
#_____________________________________________________________________________________________________________________________________________

    # 7. Rebuild the boundary conditions on the remeshed system
    # The fluid–solid interface conditions are not imposed directly through new_cons. They are already enforced weakly using the Lagrange multipliers lamf and lam
    new_cons = {}

    sqr_ds = topo.boundary.integral('d_k d_k dSref' @ new_ns, degree=4) / domain.cylinder_radius**3 # Preserve d = 0 on the original topology's outer boundary
    new_cons = System(sqr_ds, trial='d').solve_constraints(droptol=1e-9, constrain=new_cons)

    sqr_dm_outer = new_topo['fluid'].boundary['outer'].integral('dm_k dm_k dSref' @ ns_m, degree=4) / domain.cylinder_radius**3 #d_m = 0 on the channel walls
    dm_outer_only = System(sqr_dm_outer, trial='dm').solve_constraints(droptol=1e-9)
    dm_outer_cons = dm_outer_only['dm'].copy()

    new_cons = dict(new_cons)
    new_cons['dm'] = dm_outer_cons.copy()

    #Impose the external fluid velocity
    ns_bc = Namespace()
    ns_bc.urel  = new_ns.urel
    ns_bc.xfnew = new_geom

    ns_bc.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))

    ns_bc.δ = new_ns.δ
    # Extensional flow conditions:
    ns_bc.epsdot = fluid.velocity / (2 * domain.cylinder_radius)
    ns_bc.uext_i = 'epsdot (δ_i0 xfnew_0 - δ_i1 xfnew_1)'

    sqr_u = new_topo['fluid'].boundary['outer'].integral('(urel_i - uext_i) (urel_i - uext_i) dSref' @ ns_bc, degree=4) / (2 * domain.cylinder_radius * fluid.velocity**2)
    new_cons = System(sqr_u, trial='u').solve_constraints(droptol=1e-9, constrain=new_cons)
    new_ucons = new_cons['u'].copy()

    #Fix the pressure reference: Fixing one pressure DOF provides a pressure reference without prescribing the pressure throughout the domain.
    pcons = numpy.full(function.arguments_for(new_ns.p)['p'].shape, numpy.nan)
    pcons[0] = 0.0
    new_cons['p'] = pcons
#_____________________________________________________________________________________________________________________________________________

    # 8. Build new Newton system, only the construction not executing the iterations
    new_system = System(new_res, trial=['d', 'cm', 'dm', 'u', 'p', 'lam', 'lamf'], test=['dtest', 'cmtest', 'dmtest', 'utest', 'ptest', 'lamtest', 'lamftest'])
#______________________________________________________________________________________________________________________________________________________________________________

    # 9. Transfer the old fluid solution onto the new mesh
    #  Evaluate the fluid solution and its time-integration history from the old fluid mesh at the quadrature points of the newly generated fluid mesh.

    #Define the evaluation points on the new fluid mesh. These are the physical locations where the old solution must be evaluated.
    new_dof_sample = new_topo['fluid'].sample('gauss', 4) #creates a fourth-order Gauss quadrature points
    new_dof_x_m    = numpy.asarray(new_dof_sample.eval(new_geom) / 'm', dtype=float) # evaluates the physical coordinates of these points

    R_m = float(domain.cylinder_radius / 'm')
    transfer_via_locate = False
    new_graduf = None
    
    for transfer_tol_m in [1e-4*R_m, 5e-4*R_m, 1e-3*R_m]:
        try:
            #Find the corresponding positions in the old fluid topology where the existing finite-element solution can be evaluated.
            old_located = old_topo_fluid.locate(x_field / Length('1m'), new_dof_x_m, tol=transfer_tol_m, arguments=args, skip_missing=False)
            
            #Evaluate the old solution and Newmark history
            (t_uf, t_vm, t_am, t_urel, t_arel, t_p,
             t_a0dt2, t_u0, t_a0dt_urel) = function.eval([
                old_located.bind(v_field + urel_field),
                old_located.bind(v_field),
                old_located.bind(a_field),
                old_located.bind(urel_field),
                old_located.bind(arel_field),
                old_located.bind(p_field),
                old_located.bind(a0dt2_expr),
                old_located.bind(u0_expr),
                old_located.bind(a0dt_urel_expr)], arguments=args)

            #Transfer the physical fluid velocity "gradient"
            ns_old_grad = Namespace()
            ns_old_grad.x = x_field
            ns_old_grad.define_for('x', gradient='∇') #It defines gradients with respect to the old physical fluid geometry
            ns_old_grad.uf = v_field + urel_field
    
            t_graduf = function.eval(old_located.bind('∇_j(uf_i)' @ ns_old_grad), arguments=args) #evaluates the gradient at the located points. 
            new_graduf = numpy.asarray(t_graduf / '1/s', dtype=float) # contains the physical fluid velocity gradient, not the gradient of the relative velocity alone.

            #Convert the evaluated quantities into numerical arrays
            new_uf_x, new_uf_y = numpy.asarray(t_uf / 'm/s', dtype=float).T
            new_vm_x, new_vm_y = numpy.asarray(t_vm / 'm/s', dtype=float).T
            new_am_x, new_am_y = numpy.asarray(t_am / 'm/s2', dtype=float).T
            new_urel_x, new_urel_y = numpy.asarray(t_urel / 'm/s', dtype=float).T
            new_arel_x, new_arel_y = numpy.asarray(t_arel / 'm/s2', dtype=float).T
            new_p_Pa = numpy.asarray(t_p / 'Pa', dtype=float)
            new_a0dt2_x, new_a0dt2_y = numpy.asarray(t_a0dt2 / 'm', dtype=float).T
            new_u0_x, new_u0_y = numpy.asarray(t_u0 / 'm/s', dtype=float).T
            new_a0dt_urel_x, new_a0dt_urel_y = numpy.asarray(t_a0dt_urel / 'm/s', dtype=float).T
    
            transfer_via_locate = True
    
            log.info(
                f'[REMESH] field transfer via Topology.locate '
                f'(tol={transfer_tol_m:.3e} m)'
            )
    
            break
    
        except Exception as ex:
            log.info(
                f'[REMESH] Topology.locate failed with '
                f'tol={transfer_tol_m:.3e} m: {ex}'
            )
    
    #Alternative interpolation if geometric location fails
    if not transfer_via_locate: 

        def _I(vals):
            return LinearNDInterpolator(old_x_m, vals, fill_value=0.0)(new_dof_x_m) #constructs values at the new quadrature points using previously sampled coordinates and values from the old mesh.

        new_uf_x = _I(old_uf_ms[:, 0])
        new_uf_y = _I(old_uf_ms[:, 1])
        new_vm_x = _I(old_vm_ms[:, 0])
        new_vm_y = _I(old_vm_ms[:, 1])
        new_am_x = _I(old_am_ms2[:, 0])
        new_am_y = _I(old_am_ms2[:, 1])
        new_urel_x = _I(old_urel_ms[:, 0])
        new_urel_y = _I(old_urel_ms[:, 1])
        new_arel_x = _I(old_arel_ms2[:, 0])
        new_arel_y = _I(old_arel_ms2[:, 1])
        new_p_Pa = _I(old_p_Pa)
        new_a0dt2_x = _I(old_a0dt2_m[:, 0])
        new_a0dt2_y = _I(old_a0dt2_m[:, 1])
        new_u0_x = _I(old_u0_ms[:, 0])
        new_u0_y = _I(old_u0_ms[:, 1])
        new_a0dt_urel_x = _I(old_a0dt_urel_ms[:, 0])
        new_a0dt_urel_y = _I(old_a0dt_urel_ms[:, 1])

    if new_graduf is None:
        raise RuntimeError('Gradient-preserving restart requires locate-based transfer.')

    #Retrieve the Newmark integration parameters
    beta, gamma, dt = dynamic.beta, dynamic.gamma, float(dynamic.timestep / 's')
    # beta: displacement integration parameter.
    # gamma: velocity integration parameter.
    # dt: physical time-step size in seconds.

#At the end of this block, the code has obtained pointwise values on the new fluid mesh for:
    # - Physical fluid velocity and pressure.
    # - Mesh velocity and acceleration.
    # - Relative fluid velocity and acceleration.
    # - Displacement- and velocity-based Newmark histories.
    # - Physical fluid velocity gradient.
# **Important:** The solution has not yet been projected onto the new finite-element basis.
#_____________________________________________________________________________________________________________________________________________

    # 10. Reconstruct the fluid solution and Newmark histories on the new mesh
    # The reconstruction must also satisfy three requirements:
    #     Preserve the physical fluid solution as accurately as possible.
    #     Maintain compatibility between the fluid mesh motion and the solid interface.
    #     Reconstruct consistent Newmark histories so that time integration can continue without an artificial jump.

    R_scale   = float(domain.cylinder_radius / 'm')
    u_scale   = float(fluid.velocity / 'm/s')
    p_scale_v = float(fluid.viscosity * fluid.velocity / domain.cylinder_radius / 'Pa')

    #L2 projection of a vector field
    #This is an unconstrained projection. It approximates the transferred field but does not independently impose boundary conditions or incompressibility.
    def _project_vector(target_x, target_y, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=2, shape=(2,))
        ns_p.ftarget = new_dof_sample.asfunction(numpy.stack([target_x / scale, target_y / scale], axis=1).astype(float))
        #minimizing f_proj - f_target
        sqr = new_dof_sample.integral(((ns_p.fproj - ns_p.ftarget) ** 2).sum(-1))
        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']

    # Constrained projection of a mesh-related field. This function is subsequently used to reconstruct mesh acceleration and acceleration-history fields.
    def _project_mesh_state(tx, ty, interface_target):
        ns_h = Namespace()
        ns_h.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
        ns_h.htarget = new_dof_sample.asfunction(numpy.stack([tx / R_scale, ty / R_scale], axis=1).astype(float)) * domain.cylinder_radius

        sqr_gamma = zipped_traction.integral(_dot2(ns_h.h - interface_target, ns_h.h - interface_target) * new_ns.dS) / domain.cylinder_radius**3
        hcons = System(sqr_gamma, trial='h').solve_constraints(droptol=1e-9, constrain={'h': dm_outer_cons.copy()}, arguments=args)
        sqr_bulk = new_dof_sample.integral((((ns_h.h - ns_h.htarget) / domain.cylinder_radius)**2).sum(-1))
        return System(sqr_bulk, trial='h').solve(constrain=hcons, arguments={})['h']

    # Nonlinear mesh-extension solve. It determines the interior field by solving an artificial mesh-extension equilibrium equation.
    def _extend_mesh_state(interface_target):
        ns_h = Namespace()
        ns_h.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
        ns_h.htest = function.replace_arguments(new_ns.dm, 'dm:htest') / (domain.cylinder_radius**2 * Pressure('1Pa'))
        ns_h.xfnew = new_geom
        ns_h.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))
    
        ns_h.xh_i = 'xfnew_i + h_i'
        ns_h.Fh_ij = '∇ref_j(xh_i)'
        ns_h.Fhinv = numpy.linalg.inv(ns_h.Fh)
        ns_h.Ph_ij = '2 (Fh_ij - Fhinv_ji)'
    
        sqr_gamma = zipped_traction.integral(_dot2(ns_h.h - interface_target, ns_h.h - interface_target) * new_ns.dS) / domain.cylinder_radius**3
        hcons = System(sqr_gamma, trial='h').solve_constraints(droptol=1e-9, constrain={'h': dm_outer_cons.copy()}, arguments=args)
        res_h = Pressure('1Pa') * new_topo['fluid'].integral('∇ref_j(htest_i) Ph_ij dVref' @ ns_h, degree=4)
        return System(res_h, trial='h', test='htest').solve(constrain=hcons, arguments={'h': numpy.nan_to_num(hcons['h'], nan=0.)}, tol=1e-12)['h']

    #Project the physical fluid velocity transferred from the old mesh:
    uf_proj_dofs = _project_vector(new_uf_x, new_uf_y, u_scale)
    #At this stage, the velocity is only an L2 approximation of the old physical fluid velocity. It has not yet undergone the divergence-constrained and gradient-preserving correction introduced later.

    #Reconstruct the fluid mesh velocity
    V_mesh = _extend_mesh_state(new_ns.v * dynamic.timestep) #(new_ns.v represents the solid velocity obtained from the displacement-based Newmark scheme.)
    #The mesh velocity is reconstructed independently, using the current solid velocity at the interface and the artificial mesh-extension equation in the interior.

    #Recover the relative fluid velocity
    urel_new_dofs = (uf_proj_dofs - V_mesh * R_scale / (dt * u_scale)) 
    #This is not yet the final reconstructed relative velocity. A further constrained projection is performed to improve the velocity gradients and incompressibility.

    # Gao's ramp is complete at the first remesh (0.201 s > 0.1 s).
    u_fixed = numpy.isfinite(new_ucons)
    urel_new_dofs[u_fixed] = new_ucons[u_fixed]

    ns_proj = Namespace()
    ns_proj.x = new_geom / domain.cylinder_radius
    ns_proj.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns_proj.uproj = new_topo['fluid'].field('uproj', btype='std', degree=2, shape=(2,))
    ns_proj.uprojtest = function.replace_arguments(ns_proj.uproj, 'uproj:uprojtest')
    ns_proj.qproj = new_topo['fluid'].field('qproj', btype='std', degree=1)
    ns_proj.qprojtest = function.replace_arguments(ns_proj.qproj, 'qproj:qprojtest')
    ns_proj.vm = (function.replace_arguments(new_ns.dm, [('dm', 'Vproj')]) / dynamic.timestep / fluid.velocity)
    ns_proj.uf_i = 'vm_i + uproj_i'
    ns_proj.utarget = new_dof_sample.asfunction(numpy.stack([new_uf_x / u_scale, new_uf_y / u_scale], axis=1).astype(float))
    ns_proj.gtarget = new_dof_sample.asfunction((new_graduf * R_scale / u_scale).astype(float))
    
    new_cyl_sample = new_topo['fluid'].boundary['cylinder'].sample('gauss', 4)
    new_cyl_x_m = numpy.asarray(new_cyl_sample.eval(new_geom) / 'm', dtype=float)
    
    old_cyl_located = old_topo_fluid.boundary['cylinder'].locate(x_field / Length('1m'), new_cyl_x_m, tol=transfer_tol_m, arguments=args, skip_missing=False)
    
    ns_old_cyl = Namespace()
    ns_old_cyl.x = x_field
    ns_old_cyl.define_for('x', gradient='∇')
    ns_old_cyl.uf = v_field + urel_field
    
    old_graduf_cyl = numpy.asarray(function.eval(old_cyl_located.bind('∇_j(uf_i)' @ ns_old_cyl), arguments=args) / '1/s', dtype=float)
    
    ns_proj.gtargetcyl = new_cyl_sample.asfunction((old_graduf_cyl * R_scale / u_scale).astype(float))
    ns_proj.wH1 = 0.5**2
    ns_proj.wGamma = 0.1
    
    res_proj = new_dof_sample.integral((
        'uprojtest_i (uf_i - utarget_i) '
        '+ 0.25 wH1 '
        '(∇_j(uprojtest_i) + ∇_i(uprojtest_j)) '
        '((∇_j(uf_i) + ∇_i(uf_j)) '
        '- (gtarget_ij + gtarget_ji))') @ ns_proj)
    
    res_proj += ns_proj.wGamma * new_cyl_sample.integral(
        '((∇_j(uprojtest_i) + ∇_i(uprojtest_j)) n_j) '
        '(((∇_k(uf_i) + ∇_i(uf_k)) n_k) '
        '- ((gtargetcyl_ik + gtargetcyl_ki) n_k))' @ ns_proj)
    
    res_proj += new_topo['fluid'].integral('qproj ∇_i(uprojtest_i) dV' @ ns_proj, degree=4)
    res_proj += new_topo['fluid'].integral('qprojtest ∇_i(uf_i) dV' @ ns_proj, degree=4)
    
    uproj_cons = numpy.full_like(new_ucons, numpy.nan)
    uproj_cons[u_fixed] = urel_new_dofs[u_fixed]
    
    qproj_cons = numpy.full(function.arguments_for(res_proj)['qproj'].shape, numpy.nan)
    qproj_cons[0] = 0.0
    
    proj_solution = System(
        res_proj,
        trial=['uproj', 'qproj'],
        test=['uprojtest', 'qprojtest']).solve(constrain={
            'uproj': uproj_cons,
            'qproj': qproj_cons},
        arguments={
            'uproj': urel_new_dofs.copy(),
            'Vproj': V_mesh.copy(),
            'qproj': numpy.zeros(
                function.arguments_for(res_proj)['qproj'].shape)}, tol=1e-12)
    
    urel_proj_dofs = proj_solution['uproj']

    a0dt_new = _project_vector(new_a0dt_urel_x, new_a0dt_urel_y, u_scale)
    A_rel = _project_vector(new_arel_x * dt, new_arel_y * dt, u_scale)

    u0_new = (urel_new_dofs - gamma * A_rel - (1 - gamma) * a0dt_new)
    u0_proj = (urel_proj_dofs - gamma * A_rel - (1 - gamma) * a0dt_new)
    A_mesh = _project_mesh_state(new_am_x * dt**2, new_am_y * dt**2, new_ns.a * dynamic.timestep**2)

    dm_n_dofs = numpy.zeros_like(V_mesh)

    a0dt2_solid = function.replace_arguments(new_ns.d, [('d', 'a0δt2')])
    A0_mesh = _project_mesh_state(new_a0dt2_x, new_a0dt2_y, a0dt2_solid)

    δA = A_mesh - A0_mesh

    am0dt2_new = A0_mesh
    vm0dt_new = V_mesh - A0_mesh - gamma * δA
    dm0_new = dm_n_dofs - vm0dt_new - .5*A0_mesh - beta*δA

    # Check compatibility of reconstructed mesh Newmark history with solid history
    ns_hist     = Namespace()
    ns_hist.h   = function.replace_arguments(new_ns.dm, [('dm', 'h')])
    d0_solid    = function.replace_arguments(new_ns.d, [('d', 'd0')])
    v0dt_solid  = function.replace_arguments(new_ns.d, [('d', 'u0δt')])
    a0dt2_solid = function.replace_arguments(new_ns.d, [('d', 'a0δt2')])

    dm0_g, ds0_g = function.eval([zipped_traction.bind(ns_hist.h), zipped_traction.bind(d0_solid - d_s_remesh_field)], arguments=dict(args, h=dm0_new))
    vm0_g, vs0_g = function.eval([zipped_traction.bind(ns_hist.h / dynamic.timestep), zipped_traction.bind(v0dt_solid / dynamic.timestep)], arguments=dict(args, h=vm0dt_new))
    am0_g, as0_g = function.eval([zipped_traction.bind(ns_hist.h / dynamic.timestep**2), zipped_traction.bind(a0dt2_solid / dynamic.timestep**2)], arguments=dict(args, h=am0dt2_new))

    dd0 = numpy.linalg.norm(numpy.asarray((dm0_g - ds0_g) / 'm'), axis=-1)
    dv0 = numpy.linalg.norm(numpy.asarray((vm0_g - vs0_g) / 'm/s'), axis=-1)
    da0 = numpy.linalg.norm(numpy.asarray((am0_g - as0_g) / 'm/s2'), axis=-1)

    R_m = float(domain.cylinder_radius / 'm')
    vs0_ref = max(numpy.linalg.norm(numpy.asarray(vs0_g / 'm/s'), axis=-1).max(), 1e-30)
    as0_ref = max(numpy.linalg.norm(numpy.asarray(as0_g / 'm/s2'), axis=-1).max(), 1e-30)

    log.info(f'[INTERFACE HISTORY] ||dm0-(ds0-ds_r)|| max={dd0.max():.6e} m, /R={dd0.max()/R_m:.6e}')
    log.info(f'[INTERFACE HISTORY] ||vm0-vs0|| max={dv0.max():.6e} m/s, rel={dv0.max()/vs0_ref:.6e}')
    log.info(f'[INTERFACE HISTORY] ||am0-as0|| max={da0.max():.6e} m/s2, rel={da0.max()/as0_ref:.6e}')

    δaδt2_new = (dm_n_dofs - dm0_new - vm0dt_new - .5*am0dt2_new) / beta

    vm_new_dofs = vm0dt_new + am0dt2_new + gamma*δaδt2_new
    am_new_dofs = am0dt2_new + δaδt2_new

    vm_new_dofs_ms = vm_new_dofs * R_scale / dt

    def _project_scalar(target, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=1)
        ns_p.ftarget = new_dof_sample.asfunction((target / scale).astype(float))
        sqr = new_dof_sample.integral((ns_p.fproj - ns_p.ftarget) ** 2)
        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']


    p_proj_dofs  = _project_scalar(new_p_Pa, p_scale_v)
    p_proj_dofs -= p_proj_dofs[0]
    log.info(f'[PRESSURE GAUGE] p_proj[0]={p_proj_dofs[0]:.6e}, physical={p_proj_dofs[0]*p_scale_v:.6e} Pa')

    # ------------------------------------------------------------------
    # 12. Build new_args
    # ------------------------------------------------------------------

    new_args = dict(args) # Solid d, d0, u0δt, a0δt2 remain unchanged.

    # Fresh mesh displacement
    new_args['dm'] = dm_n_dofs

    # Consistent mesh Newmark history
    new_args['dm0']    = dm0_new
    new_args['vm0δt']  = vm0dt_new
    new_args['am0δt2'] = am0dt2_new

    # Current relative velocity
    new_args['u'] = urel_new_dofs

    # Consistent relative-velocity Newmark history
    new_args['u0']   = u0_new
    new_args['a0δt'] = a0dt_new

    # Pressure
    new_args['p'] = p_proj_dofs

    # Interface Lagrange multiplier
    new_args['lam'] = numpy.zeros(function.arguments_for(new_res)['lam'].shape)
    new_args['lamf'] = numpy.zeros(function.arguments_for(new_res)['lamf'].shape)
    if remesh_index == 1 and 'Rreact_old_diag' in fluid_state:
        Ru_new = numpy.asarray(
            function.eval(res_fluid_mom.derivative('utest'), arguments=new_args),
            dtype=float).reshape(-1)
    
        Rreact_new = Tdu.T @ Ru_new[i_u_gamma]
        Rreact_old = fluid_state['Rreact_old_diag']
    
        log.info(
            f'[REACTION REMESH COMPARE] '
            f'||old||={numpy.linalg.norm(Rreact_old):.6e}, '
            f'||new||={numpy.linalg.norm(Rreact_new):.6e}, '
            f'||new-old||={numpy.linalg.norm(Rreact_new-Rreact_old):.6e}, '
            f'||new+old||={numpy.linalg.norm(Rreact_new+Rreact_old):.6e}, '
            f'rel={numpy.linalg.norm(Rreact_new-Rreact_old)/max(numpy.linalg.norm(Rreact_old),1e-30):.6e}'
        )
    # Old/new traction before projected restart
    old_pres = numpy.asarray(told_pres / 'Pa', dtype=float)
    old_visc = numpy.asarray(told_visc / 'Pa', dtype=float)
    old_tot  = numpy.asarray(told_tot  / 'Pa', dtype=float)
    
    def _traction_error(label, old_t, new_t):
        err = numpy.linalg.norm(new_t - old_t, axis=-1)
        ref = numpy.linalg.norm(old_t, axis=-1)
    
        err_rms = numpy.sqrt(numpy.mean(err**2))
        ref_rms = max(
            numpy.sqrt(numpy.mean(ref**2)), 1e-30)
    
        log.info(
            f'[TRACTION TRANSFER] {label}: '
            f'max={err.max():.6e} Pa, '
            f'RMS={err_rms:.6e} Pa, '
            f'relRMS={err_rms/ref_rms:.6e}'
        )
    
    tnew_pres, tnew_visc, tnew_tot = function.eval(
        [
            fluid_traction_located.bind(ns_f.tpres),
            fluid_traction_located.bind(ns_f.tvisc),
            fluid_traction_located.bind(ns_f.traction),
        ],
        arguments=new_args,
    )
    
    _traction_error(
        'pressure',
        old_pres,
        numpy.asarray(tnew_pres / 'Pa', dtype=float),
    )
    _traction_error(
        'viscous',
        old_visc,
        numpy.asarray(tnew_visc / 'Pa', dtype=float),
    )
    _traction_error(
        'total',
        old_tot,
        numpy.asarray(tnew_tot / 'Pa', dtype=float),
    )
    
    # Raw versus projected fluid restart
    projected_args = dict(new_args)
    projected_args['u'] = urel_proj_dofs
    projected_args['u0'] = u0_proj
    
    ru_expr = res_fluid_mom.derivative('utest')
    rp_expr = res_fluid_inc.derivative('ptest')
    
    free_u = ~numpy.isfinite(new_cons['u']).reshape(-1)
    free_p = ~numpy.isfinite(new_cons['p']).reshape(-1)
    
    ru_transfer = numpy.asarray(
        function.eval(ru_expr, arguments=new_args)
    ).reshape(-1)[free_u]
    
    rp_transfer = numpy.asarray(
        function.eval(rp_expr, arguments=new_args)
    ).reshape(-1)[free_p]
    
    ru_projected = numpy.asarray(
        function.eval(ru_expr, arguments=projected_args)
    ).reshape(-1)[free_u]
    
    rp_projected = numpy.asarray(
        function.eval(rp_expr, arguments=projected_args)
    ).reshape(-1)[free_p]
    
    du_proj_g = function.eval(
        new_dof_sample.bind(new_ns.urel),
        arguments={'u': urel_proj_dofs - urel_new_dofs},
    )
    
    du_proj = numpy.linalg.norm(
        numpy.asarray(du_proj_g / 'm/s'), axis=-1)
    
    log.info(
        f'[PROJECTED RESTART] '
        f'||Ru|| raw={numpy.linalg.norm(ru_transfer):.6e}, '
        f'projected={numpy.linalg.norm(ru_projected):.6e}, '
        f'||Rp|| raw={numpy.linalg.norm(rp_transfer):.6e}, '
        f'projected={numpy.linalg.norm(rp_projected):.6e}, '
        f'max|du|={du_proj.max():.6e} m/s'
    )

    # Diagnostic only: can (u,p) restore momentum + incompressibility?
    up_corr_solution = System(
        res_fluid_mom + res_fluid_inc,
        trial=['u', 'p'],
        test=['utest', 'ptest']).solve(
        constrain={
            'u': new_ucons,
            'p': new_cons['p'],
        },
        arguments=projected_args,
        tol=1e-12)
    
    up_corrected_args = dict(projected_args)
    up_corrected_args['u'] = up_corr_solution['u']
    up_corrected_args['p'] = up_corr_solution['p']
    
    ru_up = numpy.asarray(function.eval(ru_expr, arguments=up_corrected_args)).reshape(-1)[free_u]
    
    rp_up = numpy.asarray(function.eval(rp_expr, arguments=up_corrected_args)).reshape(-1)[free_p]
    
    du_up_g = function.eval(
        new_dof_sample.bind(new_ns.urel),
        arguments={'u': up_corr_solution['u'] - urel_proj_dofs})
    du_up = numpy.linalg.norm(numpy.asarray(du_up_g / 'm/s'), axis=-1)
    dp_up_g = function.eval(new_dof_sample.bind(new_ns.p), arguments={'p': up_corr_solution['p'] - projected_args['p']})
    dp_up = numpy.abs(numpy.asarray(dp_up_g / 'Pa'))
    
    log.info(
        f'[COUPLED-UP CORRECTION] '
        f'Ru={numpy.linalg.norm(ru_up):.6e}, '
        f'Rp={numpy.linalg.norm(rp_up):.6e}, '
        f'max|du|={du_up.max():.6e} m/s, '
        f'max|dp|={dp_up.max():.6e} Pa')

    # Coupled (u,p) correction while preserving the transferred
    # Newmark relative acceleration
    du_up_dofs = up_corr_solution['u'] - urel_proj_dofs
    
    u0_up_consistent = u0_proj + du_up_dofs
    
    up_hist_args = dict(projected_args)
    up_hist_args['u']  = up_corr_solution['u']
    up_hist_args['p']  = up_corr_solution['p']
    up_hist_args['u0'] = u0_up_consistent

    ru_up_hist = numpy.asarray(function.eval(ru_expr, arguments=up_hist_args)).reshape(-1)[free_u]
    rp_up_hist = numpy.asarray(function.eval(rp_expr, arguments=up_hist_args)).reshape(-1)[free_p]
    arel_proj_g = function.eval(new_dof_sample.bind(new_ns.arel), arguments=projected_args)
    arel_up_hist_g = function.eval(new_dof_sample.bind(new_ns.arel), arguments=up_hist_args)
    
    darel_up_hist = numpy.linalg.norm(numpy.asarray((arel_up_hist_g - arel_proj_g) / 'm/s2'), axis=-1)
    
    log.info(
        f'[COUPLED-UP CONSISTENT-HISTORY] '
        f'Ru={numpy.linalg.norm(ru_up_hist):.6e}, '
        f'Rp={numpy.linalg.norm(rp_up_hist):.6e}, '
        f'max|darel|={darel_up_hist.max():.6e} m/s2, '
        f'RMS|darel|={numpy.sqrt(numpy.mean(darel_up_hist**2)):.6e} m/s2'
    )

    
    # Exact change in relative acceleration caused by the coupled (u,p) correction
    arel_up_g = function.eval(new_dof_sample.bind(new_ns.arel), arguments=up_corrected_args)
    darel_up = numpy.linalg.norm(numpy.asarray((arel_up_g - arel_proj_g) / 'm/s2'), axis=-1)
    
    new_args['u']  = up_corr_solution['u']
    new_args['p']  = up_corr_solution['p']
    new_args['u0'] = u0_up_consistent

    tproj_pres, tproj_visc, tproj_tot = function.eval(
        [
            fluid_traction_located.bind(ns_f.tpres),
            fluid_traction_located.bind(ns_f.tvisc),
            fluid_traction_located.bind(ns_f.traction),
        ],
        arguments=new_args,
    )
    
    _traction_error(
        'pressure POST-PROJ',
        old_pres,
        numpy.asarray(tproj_pres / 'Pa', dtype=float),
    )
    _traction_error(
        'viscous POST-PROJ',
        old_visc,
        numpy.asarray(tproj_visc / 'Pa', dtype=float),
    )
    _traction_error(
        'total POST-PROJ',
        old_tot,
        numpy.asarray(tproj_tot / 'Pa', dtype=float),
    )

    # ------------------------------------------------------------------
    # CHECK: is physical fluid velocity preserved by remeshing?
    #
    # Target:
    #     u_f,target = projection of old physical velocity onto new mesh
    #
    # Actual:
    #     u_f,new = v_m,new + u_rel,new
    # ------------------------------------------------------------------

    uf_target_field = function.replace_arguments(new_ns.urel, [('u', 'ufcheck')])
    uf_target_g, uf_new_g = function.eval([new_dof_sample.bind(uf_target_field), new_dof_sample.bind(new_ns.u)], arguments=dict(new_args, ufcheck=uf_proj_dofs))

    uf_target = numpy.asarray(uf_target_g / 'm/s', dtype=float)
    uf_new = numpy.asarray(uf_new_g / 'm/s', dtype=float)
    duf = numpy.linalg.norm(uf_new - uf_target, axis=-1)
    uf_ref = numpy.sqrt(numpy.mean(numpy.linalg.norm(uf_target, axis=-1)**2))
    duf_rms = numpy.sqrt(numpy.mean(duf**2))

    log.info(
        f'[PHYSICAL VELOCITY TRANSFER] '
        f'max={duf.max():.6e} m/s, '
        f'RMS={duf_rms:.6e} m/s, '
        f'relRMS={duf_rms/max(uf_ref, 1e-30):.6e}')

    vm_g, urel_g, uf_g, vs_g = function.eval(
        [zipped_traction.bind(new_ns.vm), zipped_traction.bind(new_ns.urel),
        zipped_traction.bind(new_ns.u), zipped_traction.bind(new_ns.v)],
        arguments=new_args)

    vm_vs = numpy.linalg.norm(numpy.asarray((vm_g - vs_g) / 'm/s'), axis=-1)
    urel_mag = numpy.linalg.norm(numpy.asarray(urel_g / 'm/s'), axis=-1)
    uf_vs = numpy.linalg.norm(numpy.asarray((uf_g - vs_g) / 'm/s'), axis=-1)
    vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_g / 'm/s'), axis=-1).max(), 1e-30)

    log.info(f'[INTERFACE VELOCITY] ||v_m-v_s|| max={vm_vs.max():.6e} m/s, rel={vm_vs.max()/vs_ref:.6e}')
    log.info(f'[INTERFACE VELOCITY] ||u_rel||   max={urel_mag.max():.6e} m/s, rel={urel_mag.max()/vs_ref:.6e}')
    log.info(f'[INTERFACE VELOCITY] ||u_f-v_s|| max={uf_vs.max():.6e} m/s, rel={uf_vs.max()/vs_ref:.6e}')

    log.info('Fluid solution transferred to new mesh; mesh Newmark history reconstructed consistently.')

    # ------------------------------------------------------------------
    # TRANSFER CHECK: current mesh velocity
    # ------------------------------------------------------------------

    if transfer_via_locate:
        old_vm_check = numpy.asarray(function.eval(old_located.bind(v_field), arguments=args) / 'm/s')
    else:
        old_vm_check = numpy.column_stack((new_vm_x, new_vm_y))

    new_vm_check = numpy.asarray(function.eval(new_dof_sample.bind(new_ns.vm), arguments=new_args) / 'm/s')

    abs_err = numpy.linalg.norm(new_vm_check - old_vm_check, axis=1)
    old_mag = numpy.linalg.norm(old_vm_check, axis=1)

    rms_err = numpy.sqrt(numpy.mean(abs_err**2))
    rms_old = numpy.sqrt(numpy.mean(old_mag**2))
    rel_rms = rms_err / max(rms_old, 1e-30)

    log.info(f'[TRANSFER CHECK] v_mesh: max={abs_err.max():.6e} m/s, RMS={rms_err:.6e} m/s, relRMS={rel_rms:.6e}')

    
    # ------------------------------------------------------------------
    # 13. Verification checks
    # ------------------------------------------------------------------
    from scipy.spatial import cKDTree

    new_fluid_bbezier_check = new_topo['fluid'].boundary['cylinder'].sample('bezier', 3)
    xb_fluid_inner = numpy.asarray(new_fluid_bbezier_check.eval(new_geom) / 'm', dtype=float)
    xb_solid_outer = numpy.asarray(xb_current_m)

    tree = cKDTree(xb_fluid_inner)
    dists, _ = tree.query(xb_solid_outer)
    log.info(f'[REMESH CHECK] Interface gap max : {dists.max():.3e} m')
    log.info(f'[REMESH CHECK] Interface gap mean: {dists.mean():.3e} m')

    dm_max = numpy.abs(new_args['dm']).max()
    log.info(f'[REMESH CHECK] max |dm| after reset: {dm_max:.3e}  (expect ~0)')
    log.info(f'[REMESH CHECK] max |v_m^new| : {numpy.abs(vm_new_dofs_ms).max():.3e} m/s  (need not be 0)')

    # Kinematic interface consistency: the mesh must move with the solid,
    # i.e. v_m = v_s and a_m = a_s on the cylinder.  The bulk interpolation
    # used for the dm Newmark history does NOT guarantee this at the new
    # interface dofs, so we measure it explicitly.  Large values here mean
    # the interface histories should be taken from the solid rather than
    # from bulk interpolation.
    try:
        vm_gamma, am_gamma, vs_gamma, as_gamma = function.eval(
            [zipped_traction.bind(new_ns.vm), zipped_traction.bind(new_ns.am),
             zipped_traction.bind(new_ns.v),  zipped_traction.bind(new_ns.a)],
            arguments=new_args,
        )
        dv = numpy.linalg.norm(numpy.asarray((vm_gamma - vs_gamma) / 'm/s'), axis=-1)
        da = numpy.linalg.norm(numpy.asarray((am_gamma - as_gamma) / 'm/s2'), axis=-1)
        vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_gamma / 'm/s'), axis=-1).max(), 1e-30)
        as_ref = max(numpy.linalg.norm(numpy.asarray(as_gamma / 'm/s2'), axis=-1).max(), 1e-30)
        log.info(f'[REMESH CHECK] ||v_m - v_s||_Γ max: {dv.max():.3e} m/s   '
                 f'(rel {dv.max()/vs_ref:.3e})')
        log.info(f'[REMESH CHECK] ||a_m - a_s||_Γ max: {da.max():.3e} m/s2  '
                 f'(rel {da.max()/as_ref:.3e})')
        if dv.max() > 1e-3 * vs_ref:
            log.info('[REMESH WARNING] interface mesh/solid velocity mismatch exceeds diagnostic tolerance.')
        interface_dv = float(dv.max()); interface_da = float(da.max())
    except Exception as ex:
        log.info(f'[REMESH CHECK] interface kinematics check skipped: {ex}')
        interface_dv = interface_da = float('nan')

    log.info('=== REMESHING COMPLETE ===')

    # ------------------------------------------------------------------
    # 14. Rebuild Bezier samplers and diagnostic quantities
    # ------------------------------------------------------------------
    new_fluid_bezier = new_topo['fluid'].sample('bezier', 3)

    new_x_bz  = function.factor(new_fluid_bezier.bind(ns_f.xm))
    new_u_bz  = function.factor(new_fluid_bezier.bind(new_ns.u))
    new_p_bz  = function.factor(new_fluid_bezier.bind(new_ns.p))
    new_Jfluid_bz = new_fluid_bezier.bind(ns_m.Jm)

    # ------------------------------------------------------------------
    # 15. Updated fluid_state for the NEXT remesh (or for per-step use)
    # ------------------------------------------------------------------
    new_fluid_state = dict(
        topo_fluid = new_topo['fluid'],
        geom_bare  = new_geom_bare,
        x_field    = ns_f.xm,
        v_field    = new_ns.vm,
        a_field    = new_ns.am,
        urel_field = new_ns.urel,
        arel_field = new_ns.arel,
        p_field    = new_ns.p,
        d_field    = new_ns.dm,
        d_name     = 'dm',
        a0dt2_key  = 'am0δt2',

        # Ingredients for the per-timestep interface update
        dm_zipped_interface = zipped_traction,
        d_s_remesh_const_m  = d_s_remesh_const_m,

        # Fixed outer-boundary constraint and interface data
        dm_outer_cons       = dm_outer_cons,
        solid_cyl_gauss     = solid_cyl_gauss,

        fluid_traction_field = ns_f.traction,
        solid_traction_field = new_ns.tsolid,
        reaction_Tdu           = Tdu.copy(),
        reaction_i_u_gamma     = i_u_gamma.copy(),
        reaction_i_d_gamma     = i_d_gamma.copy(),
        reaction_res_fluid_mom = res_fluid_mom,
        reaction_res_solid_lamf = res_solid_lamf,

    )
    return (
        new_ns, new_res, new_cons, new_ucons, new_system,
        new_fluid_bezier,
        new_x_bz, new_u_bz, new_p_bz,
        new_Jfluid_bz,
        new_fluid_state,
        new_args,
        dict(
            interface_gap_max  = float(dists.max()),
            interface_gap_mean = float(dists.mean()),
            dm_max_after_reset = float(dm_max),
            vm_new_max         = float(numpy.abs(vm_new_dofs_ms).max()),
            interface_dv       = interface_dv,
            interface_da       = interface_da,
        ),
    )

def remesh_diagnostics(
    current_t_s,
    remesh_count,
    Jf_old,
    Jf_new,
    old_fluid_bezier,
    new_fluid_bezier,
    xb_solid_m,
    old_x_m,
    new_x_m,
    old_u_vals,
    new_u_vals,
    domain,
    transfer_metrics,
):
    """Produce all diagnostic plots and prints at a remesh event.

    Called immediately after ``remesh_fluid`` returns.  Generates:
    - console summary of mesh quality before/after
    - side-by-side mesh comparison plot
    - side-by-side velocity-magnitude comparison plot
    - zoomed interface-alignment plot
    - transfer error metrics in the log
    """

    from scipy.spatial import cKDTree
    L_plot = float(domain.channel_length / 'm')


    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info(f'--- Remesh #{remesh_count} at t={current_t_s:.6e} s ---')
    log.info(f'  Jf_min  before : {Jf_old.min():.4f}')
    log.info(f'  Jf_min  after  : {Jf_new.min():.4f}')
    log.info(f'  Jf_max  before : {Jf_old.max():.4f}')
    log.info(f'  Jf_max  after  : {Jf_new.max():.4f}')
    log.info(f'  Interface gap max  : {transfer_metrics["interface_gap_max"]:.3e} m')
    log.info(f'  Interface gap mean : {transfer_metrics["interface_gap_mean"]:.3e} m')
    log.info(f'  dm max after reset : {transfer_metrics["dm_max_after_reset"]:.3e}  (expect ~0)')
    log.info(f'  v_m max after transfer : {transfer_metrics["vm_new_max"]:.3e} m/s  (need not be 0)')

    # ------------------------------------------------------------------
    # Plot 1: mesh comparison (old deformed ALE vs new clean mesh)
    # ------------------------------------------------------------------
    # with export.mplfigure(f'remesh_{remesh_count:03d}_mesh_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

    #     ax1 = fig.add_subplot(121, title=f'ALE mesh before remesh\nJf_min={Jf_old.min():.3f}')
    #     export.triplot(ax1, old_x_m, hull=old_fluid_bezier.hull, linewidth=0.25)
    #     closed = numpy.vstack([xb_solid_m, xb_solid_m[:1]])
    #     ax1.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
    #     ax1.set_aspect('equal')
    #     ax1.set_xlim(-L_plot, L_plot)
    #     ax1.set_ylim(-L_plot, L_plot)
    #     ax1.legend(fontsize=6)

    #     ax2 = fig.add_subplot(122, title=f'New fluid mesh after remesh\nJf_min={Jf_new.min():.3f}')
    #     export.triplot(ax2, new_x_m, hull=new_fluid_bezier.hull, linewidth=0.25)
    #     ax2.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
    #     ax2.set_aspect('equal')
    #     ax2.set_xlim(-L_plot, L_plot)
    #     ax2.set_ylim(-L_plot, L_plot)
    #     ax2.legend(fontsize=6)

    #     fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

    # ------------------------------------------------------------------
    # Plot 2: velocity magnitude comparison (old vs new mesh)
    # ------------------------------------------------------------------
    vmin = min(old_u_vals.min(), new_u_vals.min())
    vmax = max(old_u_vals.max(), new_u_vals.max())

    # with export.mplfigure(f'remesh_{remesh_count:03d}_velocity_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

    #     ax1 = fig.add_subplot(121, title='|u| before remesh  [m/s]')
    #     im = ax1.tripcolor(
    #         old_x_m[:, 0], old_x_m[:, 1], old_fluid_bezier.tri,
    #         old_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
    #     )
    #     ax1.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
    #     ax1.set_aspect('equal')
    #     ax1.set_xlim(-L_plot, L_plot)
    #     ax1.set_ylim(-L_plot, L_plot)

    #     ax2 = fig.add_subplot(122, title='|u| after remesh  [m/s]')
    #     im2 = ax2.tripcolor(
    #         new_x_m[:, 0], new_x_m[:, 1], new_fluid_bezier.tri,
    #         new_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
    #     )
    #     ax2.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
    #     ax2.set_aspect('equal')
    #     ax2.set_xlim(-L_plot, L_plot)
    #     ax2.set_ylim(-L_plot, L_plot)

    #     fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', label='|u| [m/s]', shrink=0.6)
    #     fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

 
@dataclass
class Domain:
    '''Parameters for the domain geometry.

    The default values match Table 1 of Turek and Hron [1].'''

    channel_length: Length = Length('0.0006m')
    cylinder_radius: Length = Length('0.00003m')
    
    elemsize: Length = Length('0.00001m')
    coarsening: float = 8.

    def generate_mesh(self):
        'Call gmsh to generate mesh and return topo, geom tuple.'''

        u = Length('m') # reference length for mesh generation

        topo, geom = gmsh(
            Path(__file__).parent/'ellipse_unbounded.geo',
            dimension=2,
            order=2,
            numbers={
                 'channel_length': self.channel_length/u,
                 'cylinder_radius': self.cylinder_radius/u,
                 'elemsize': self.elemsize/u,
                 'coarsening': self.coarsening})

        bezier = topo.sample('bezier', 2)
        bezier_cylinder = topo['fluid'].boundary['cylinder'].sample('bezier', 3)
        #A = topo.points['A'].sample('gauss', 1).eval(geom)

        with export.mplfigure('mesh.jpg', dpi=150) as fig:
            ax = fig.add_subplot(111)
            export.triplot(ax, bezier.eval(geom), hull=bezier.hull)
            export.triplot(ax, bezier_cylinder.eval(geom), hull=bezier_cylinder.tri, linewidth=1, linecolor='b')

            L = self.channel_length/u
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            
 
        return topo, geom * u


@dataclass
class Solid:
    '''Parameters for the solid problem.'''

    density: Density = Density('10kg/L')
    poisson_ratio: float = .45
    shear_modulus: Pressure = Pressure('0.04340015Pa')
    gravity: Acceleration = Acceleration('0m/s2')

    def lame_parameters(self):
        'Return tuple of first and second lame parameter.'

        return 2 * self.shear_modulus * self.poisson_ratio / (1 - 2 * self.poisson_ratio), self.shear_modulus

    def young(self):
        "Return Young's elasticity modulus."

        return 2 * self.shear_modulus * (1 + self.poisson_ratio)


@dataclass
class Fluid:
    '''Parameters for the fluid problem.'''

    density: Density = Density('10kg/L')
    viscosity: Viscosity = Viscosity('0.00559Pa*s')
    velocity: Velocity = Velocity('0.0000931667m/s')

    def reynolds(self, reference_length):
        'Return Reynolds number for given reference length'

        return self.density * self.velocity * reference_length / self.viscosity


@dataclass
class Dynamic:
    '''Parameters relating to time dependence.'''

    timestep: Time = Time('0.1ms') # simulation time step size
    endtime: Time = Time('13s') # total duration of the simulation
    init: Time = Time('.1s') # duration of the ramp-up phase
    window: Time = Time('1s') # sliding window length for time series plots
    gamma: float = .5
    beta: float = .25

    def __post_init__(self):
        self.timeseries = defaultdict(deque(maxlen=round(self.window / self.timestep)).copy)

    def ramp_up(self, t):
        'Return inflow ramp-up scale factor at given time.'

        return .5 - .5 * numpy.cos(numpy.pi * min(t / self.init, 1))

    @property
    def times(self):
        'Return all configured time steps for the simulation.'

        return numpy.arange(1, self.endtime / self.timestep + .5) * self.timestep

    def add_and_plot(self, name, t, v, ax):
        'Add data point and plot time series for past window.'

        d = self.timeseries[name]
        d.append((t, v))
        times, values = numpy.stack(d, axis=1)
        ax.plot(times, values)
        ax.set_ylabel(name)
        ax.grid()
        ax.autoscale(enable=True, axis='x', tight=True)
        vmin, vmax = numpy.quantile(values, [0,1])
        vmean = (vmax + vmin) / 2
        values -= vmean
        icross, = numpy.nonzero(values[1:] * values[:-1] < 0)
        if len(icross) >= 4: # minimum of two up, two down
            tcross = (times[icross] * values[icross+1] - times[icross+1] * values[icross]) / (values[icross+1] - values[icross])
            ax.plot(tcross, [vmean] * len(icross), '+')
            ax.text(tcross[numpy.diff(tcross).argmax():][:2].mean(), vmean,
                s=f'{vmean:+.4f}\n±{(vmax-vmin)/2:.4f}\n↻{(tcross[2:]-tcross[:-2]).mean():.4f}',
                va='center', ha='center', multialignment='right')

    # The Newmark-beta scheme is used for time integration. For our formulation
    # to support both stationary and dynamic simulations, we take displacement
    # (solid) and velocity (fluid) as primary variables, with time derivatives
    # introduced via helper arguments that are updated after every solve.
    #
    # d = d0 + δt u0 + .5 δt^2 aβ, where aβ = (1-2β) a0 + 2β a
    # => δd = δt u0 + δt^2 [ .5 a0 + β δa ]
    # => δa = [ δd / δt^2 - u0 / δt - .5 a0 ] / β
    #
    # u = u0 + δt aγ, where aγ = (1-γ) a0 + γ a
    # => δu = δt [ a0 + γ δa ]
    # => δa = [ δu / δt - a0 ] / γ

    def newmark_defo_args(self, d, d0=0., u0δt=0., a0δt2=0., **args):
        δaδt2 = (d - d0 - u0δt - .5 * a0δt2) / self.beta
        uδt = u0δt + a0δt2 + self.gamma * δaδt2
        aδt2 = a0δt2 + δaδt2
        return dict(args, d=d+uδt+.5*aδt2, d0=d, u0δt=uδt, a0δt2=aδt2)

    def newmark_defo(self, d):
        D = self.newmark_defo_args(d, *[function.replace_arguments(d, [('d', t)]) for t in ('d0', 'u0δt', 'a0δt2')])
        return D['u0δt'] / self.timestep, D['a0δt2'] / self.timestep**2

    def newmark_velo_args(self, u, u0=0., a0δt=0., **args):
        aδt = a0δt + (u - u0 - a0δt) / self.gamma
        return dict(args, u=u+aδt, u0=u, a0δt=aδt)

    def newmark_velo(self, u):
        D = self.newmark_velo_args(u, *[function.replace_arguments(u, [('u', t)]) for t in ('u0', 'a0δt')])
        return D['a0δt'] / self.timestep

    # ------------------------------------------------------------------
    # Generic (named-argument) Newmark helpers.
    #
    # The methods above hardcode the argument names 'd', 'd0', 'u0δt',
    # 'a0δt2' (displacement-type) and 'u', 'u0', 'a0δt' (velocity-type).
    # After remeshing, the fluid mesh displacement field is named 'dm'
    # (to keep it distinct from the persistent solid field 'd'), so we
    # need equivalents that work with arbitrary argument names.
    # ------------------------------------------------------------------

    def newmark_defo_args_named(self, args, name, d0_name, u0dt_name, a0dt2_name):
        'Numeric Newmark predictor step for an arbitrarily-named displacement field.'

        d     = args[name]
        d0    = args.get(d0_name, 0.)
        u0dt  = args.get(u0dt_name, 0.)
        a0dt2 = args.get(a0dt2_name, 0.)
        δaδt2 = (d - d0 - u0dt - .5 * a0dt2) / self.beta
        uδt   = u0dt + a0dt2 + self.gamma * δaδt2
        aδt2  = a0dt2 + δaδt2
        new_args = dict(args)
        new_args[name]       = d + uδt + .5 * aδt2
        new_args[d0_name]    = d
        new_args[u0dt_name]  = uδt
        new_args[a0dt2_name] = aδt2
        return new_args

    def newmark_defo_named(self, d, name, d0_name, u0dt_name, a0dt2_name):
        'Symbolic Newmark velocity/acceleration for an arbitrarily-named displacement field.'

        d0    = function.replace_arguments(d, [(name, d0_name)])
        u0dt  = function.replace_arguments(d, [(name, u0dt_name)])
        a0dt2 = function.replace_arguments(d, [(name, a0dt2_name)])
        δaδt2 = (d - d0 - u0dt - .5 * a0dt2) / self.beta
        uδt   = u0dt + a0dt2 + self.gamma * δaδt2
        aδt2  = a0dt2 + δaδt2
        return uδt / self.timestep, aδt2 / self.timestep**2


def main(domain: Domain = Domain(), solid: Optional[Solid] = Solid(), fluid: Optional[Fluid] = Fluid(), dynamic: Optional[Dynamic] = Dynamic()):
    '''Turek Hron benchmark problem

    This is a monolythic ALE (Arbitrary Lagrangian Eulerian) implementation of
    the fluid-structure interaction benchmark defined in 2006 by [Turek and
    Hron](https://doi.org/10.1007/3-540-34596-5_15). The implementation covers
    all fluid dynamics tests CFD1, CFD2 and CFD3, all structural tests CSM1,
    CSM2 and CSM3, and all interaction tests FSI1, FSI2 and FSI3, as well as
    freely defined variants thereof.'''

    assert solid or fluid, 'nothing to compute'

    if fluid:
        log.info('Re:', fluid.reynolds(domain.cylinder_radius))
        if solid:
            log.info('Ca_e:', fluid.velocity * fluid.viscosity / (solid.shear_modulus *  domain.cylinder_radius))
            #log.info('Ae:', solid.young() / fluid.density / fluid.velocity**2)
            log.info('β:',  solid.density / fluid.density)

    topo, geom = domain.generate_mesh()

    fluid_bezier = topo['fluid'].sample('bezier', 3)              
    bbezier      = topo['solid'].boundary['cylinder'].sample('bezier', 3)
    solid_bezier = topo['solid'].sample('bezier', 3)            


    res = 0.
    cons = {}
    args = {}

    ns = Namespace()
    ns.δ = function.eye(2)
    ns.xref = geom
    ns.define_for('xref', gradient='∇ref', jacobians=('dVref', 'dSref'))

    if solid:

        ns.ρs = solid.density
        ns.λs, ns.μs = solid.lame_parameters()
        ns.g = -solid.gravity * ns.δ[1]

        # Deformation, velocity and acceleration are defined on the entire
        # domain, solving for conservation of momentum on the solid domain and
        # a mesh continuation problem on the fluid domain. While the latter is
        # used only when fluid is enabled, we include it regardless both for
        # simplicity and for testing purposes.
        ns.d = topo.field('d', btype='std', degree=2, shape=(2,)) * domain.cylinder_radius # deformation at the end of the timestep

        ns.cmforce = function.Argument('cm', (2,)) * solid.shear_modulus / domain.cylinder_radius
        ns.cmtest = function.Argument('cmtest', (2,))
        if dynamic:
            ns.v, ns.a = dynamic.newmark_defo(ns.d)
        else:
            ns.a = Acceleration.wrap(function.zeros((2,)))

        # Deformed geometry
        ns.δ_ij = 'δ_ij'

        ns.x_i = 'xref_i + d_i'
        ns.F_ij = '∇ref_j(x_i)' # deformation gradient tensor
        ns.J = numpy.linalg.det(ns.F)
        ns.C_ij = 'F_ki F_kj' # right Cauchy-Green deformation tensor
        ns.Cinv = numpy.linalg.inv(ns.C)
        ns.E_ij = '.5 (C_ij - δ_ij)' # Green-Lagrangian strain tensor
        ns.S_ij = 'μs (δ_ij - Cinv_ij) + λs ln(J) Cinv_ij'
        ns.P_ij     = 'F_ik S_kj'
        ns.σcauchy_ij = 'P_ik F_jk / J'
        ns.Finv         = numpy.linalg.inv(ns.F)
        
        ns.Pmesh_ij      = '2 (F_ij - Finv_ji)'



        # Momentum balance: ρs (a - g) = div P
        ns.dtest = function.replace_arguments(ns.d, 'd:dtest') / (solid.shear_modulus * domain.cylinder_radius**2)
        res += topo['solid'].integral('(∇ref_j(dtest_i) P_ij + dtest_i ρs (a_i - g_i) + dtest_i cmforce_i) dVref' @ ns, degree=4)
        res += topo['solid'].integral('cmtest_i x_i dVref' @ ns, degree=4) / domain.cylinder_radius**3
        
        # In the momentum balance above, the only test and trial dofs involved are those that have support on the solid domain. 
        # The remaining trial dofs will follow from minimizing a mesh energy functional, using the solid deformation as a boundary constraint 
        # so that the continuation problem does not feed back into the physics. 
        # To achieve this within a monolythic setting, we establish a boolean array 'dfluid' to select all dofs that are exclusively supported by the fluid domain, 
        # and use it to restrict the minimization problem to the remaining dofs via the linearize operation.

        sqr = topo['solid'].integral('d_k d_k dVref' @ ns, degree=4) / domain.cylinder_radius**4
        dfluid = numpy.isnan(System(sqr, trial='d').solve_constraints(droptol=1e-9)['d']) # true if dof is not supported by solid domain

        dfluid_mask = dfluid  # True  = fluid-bulk d DOFs  → mesh extension
        if fluid:
            res += Pressure('1Pa') * function.replace_arguments(topo['fluid'].integral('∇ref_j(dtest_i) Pmesh_ij dVref' @ ns, degree=4),
             {'dtest': function.arguments_for(res)['dtest'] * dfluid_mask})

        # Deformation constraints: fixed at exterior boundary and cylinder
        sqr = topo.boundary.integral('d_k d_k dSref' @ ns, degree=4) / domain.cylinder_radius**3
        cons = System(sqr, trial='d').solve_constraints(droptol=1e-9, constrain=cons)

        # Zero initial deformation
        args['d'] = numpy.zeros(function.arguments_for(res)['d'].shape)
        args['cm'] = numpy.zeros(2)

    else: # fully rigid solid

        ns.x = ns.xref
        ns.v = Velocity.wrap(function.zeros((2,)))
        ns.a = Acceleration.wrap(function.zeros((2,)))

    if fluid:

        ns.ρf = fluid.density
        ns.μf = fluid.viscosity

        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

        ns.urel = topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
        if dynamic:
            ns.arel = dynamic.newmark_velo(ns.urel)
            ns.u_i = 'v_i + urel_i'
            ns.DuDt_i = 'a_i + arel_i + ∇_j(u_i) urel_j' # material derivative
        else:
            ns.u = ns.urel
            ns.DuDt_i = '∇_j(u_i) u_j'

        # rate of deformation tensor 
        ns.edot_ij = '.5 (∇_j(u_i) + ∇_i(u_j))'
              
        ns.p = topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius
        ns.σ_ij = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij' # fluid stress tensor

        # Project Posseuille inflow at inlet (exact because of quadratic velocity basis), parallel outflow at outlet, and no-slip conditions for the relative velocity at the remaining boundaries.
        
        ns.epsdot = fluid.velocity / (2 * domain.cylinder_radius)
        ns.uext_i = 'epsdot (δ_i0 xref_0 - δ_i1 xref_1)'
        sqr = topo['fluid'].boundary['cylinder'].integral('urel_i urel_i dSref' @ ns, degree=4) / (2 * domain.cylinder_radius * fluid.velocity**2)
        sqr += topo['fluid'].boundary['outer'].integral('(urel_i - uext_i) (urel_i - uext_i) dSref' @ ns, degree=4) / (2 * domain.cylinder_radius * fluid.velocity**2)
        cons = System(sqr, trial='u').solve_constraints(droptol=1e-9, constrain=cons) # exact projection
        ucons = cons['u'] # save for ramp-up phase

        pcons = numpy.full(function.arguments_for(ns.p)['p'].shape, numpy.nan)
        pcons[0] = 0.0
        cons['p'] = pcons

        # Momentum balance: ρf Du/Dt = ∇σ => ∀v ∫ (v ρf Du/Dt + ∇v:σ) = ∮ v·σ·n = 0
        ns.utest = function.replace_arguments(ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
        res_fluid_mom_old = topo['fluid'].integral('(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns, degree=4)
        res += res_fluid_mom_old

        # Incompressibility: div u = 0
        ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns, degree=4)

        if solid:
            # The action of fluid stress on the solid is imposed weakly by lifting the test functions into the fluid domain, 
            # using the identity ∮ d·σ·n = ∫ ∇(d·σ) = ∫ (∇d:σ + d·∇σ) = ∫ (∇d:σ + ρf d·Du/Dt). 
            # We need the inverse of the dfluid mask to exclude dofs without support on the boundary.

            dsolid = ~dfluid # true if dof is (partially) supported by solid domain
            res_d_fluid_lift = function.replace_arguments(topo['fluid'].integral('(dtest_i ρf DuDt_i + ∇_j(dtest_i) σ_ij) dV' @ ns, degree=4), {'dtest': function.arguments_for(res)['dtest'] * dsolid})
            res_d_fluid_direct = topo['fluid'].boundary['cylinder'].integral('dtest_i σ_ij n_j dS' @ ns, degree=4)

            # Trace map: cylinder fluid-velocity test DOFs -> solid-displacement test DOFs
            cyl_gauss = topo['fluid'].boundary['cylinder'].sample('gauss', 4)
            
            test_unit = Pressure('1Pa') * Length('1m')
            dtest_dimless = ns.dtest * test_unit
            utest_dimless = ns.utest * test_unit
            
            Bd4 = function.eval(cyl_gauss.bind(dtest_dimless.derivative('dtest')))
            Bu4 = function.eval(cyl_gauss.bind(utest_dimless.derivative('utest')))
            
            Bd = Bd4.reshape(Bd4.shape[0] * Bd4.shape[1], -1)
            Bu = Bu4.reshape(Bu4.shape[0] * Bu4.shape[1], -1)
            
            i_d_gamma = numpy.flatnonzero(numpy.linalg.norm(Bd, axis=0) > 1e-12 * numpy.linalg.norm(Bd, axis=0).max())
            i_u_gamma = numpy.flatnonzero(numpy.linalg.norm(Bu, axis=0) > 1e-12 * numpy.linalg.norm(Bu, axis=0).max())
            
            Tdu_old = numpy.linalg.lstsq(Bu[:, i_u_gamma], Bd[:, i_d_gamma], rcond=None)[0]

            res += res_d_fluid_lift

        # Zero initial velocity
        args['u'] = numpy.zeros(function.arguments_for(res)['u'].shape)

        u_bz = function.factor(fluid_bezier.bind(ns.u))
        p_bz = function.factor(fluid_bezier.bind(ns.p)) 


    x_bz = function.factor(fluid_bezier.bind(ns.x))
    x_bbz = function.factor(bbezier.bind(ns.x))

    Jfluid_bz = fluid_bezier.bind(ns.J)

    # Generic "current fluid state" tracking, used by remesh_fluid so that every remesh (first, second, ...) samples the CURRENT fluid fields correctly, 
    # whatever they happen to be named after previous remeshes.

    fluid_state = dict(
        topo_fluid = topo['fluid'],
        geom_bare  = geom / Length('1m'),

        x_field    = ns.x,

        v_field    = ns.v,
        a_field    = ns.a,

        urel_field = ns.urel,
        arel_field = ns.arel,

        p_field    = ns.p,

        d_field    = ns.d,
        d_name     = 'd',
        a0dt2_key  = 'a0δt2',

        dm_zipped_interface = None,
        d_s_remesh_const_m  = None,
    )

    if solid:
        solid_bz = topo['solid'].sample('bezier',3)
        xsolid_bz = function.factor(solid_bezier.bind(ns.x))
        Jsolid_bz = solid_bz.bind(ns.J)
        d_cyl_bz = function.factor(bbezier.bind(ns.d))
        
         
    trial = (['u', 'p'] if fluid else []) + (['d', 'cm'] if solid else [])
    system = System(res, trial=list(trial), test=[t+'test' for t in trial])
    previous_t_s = 0.0
    remesh_count = 0

    # ------------------------------------------------------------------
    # Remeshing control
    # ------------------------------------------------------------------
    # Trigger remeshing when the minimum Jacobian of the fluid mesh drops below this threshold (relative to a perfect element = 1).

    enable_remeshing = False
    remesh_J_threshold = 0.6
    predictor_J_safety = 0.4
    remesh_Jmax_threshold = 2.0

    # Also allow a time-interval-based fallback trigger (set to None to disable).
    # Set to a small value for an early forced remesh to test the procedure before the mesh actually degrades. Once validated, set back to None.

    remesh_interval_s  = None  # first remesh forced at t=0.0001 s for testing

    next_remesh_time_s = remesh_interval_s if remesh_interval_s else float('inf')

    # Track whether we have remeshed (changes which Newmark keys exist)
    has_remeshed = False

    particle_csv = Path('mesh_fine.csv')
    particle_csv.write_text('time_s,time_nd,step,L_particle,B_particle,deformation_index,omega,theta_rad,theta_over_pi\n')
    epsdot_s = float((fluid.velocity / (2 * domain.cylinder_radius)) / '1/s')

    xb_initial = function.eval(x_bbz, arguments=args) / 'm'
    L0, B0, D0, omega0, theta0 = particle_shape_from_boundary(xb_initial)
    
    with particle_csv.open('a') as f:
        f.write(f'{0.0:.12e},{0.0:.12e},{0},{L0:.12e},{B0:.12e},{D0:.12e},{omega0:.12e},{theta0:.12e},{theta0/numpy.pi:.12e}\n')

    for istep, t in enumerate(log.iter.fraction('timestep', dynamic.times) if dynamic else [Time.wrap(float('inf'))], start=1):
        if has_remeshed and 'dm' in args:
            dm_n = args['dm'].copy()
    # ------------------------------------------------------------
    # Time integration predictor/update
    # ------------------------------------------------------------
        if dynamic:
            if solid:
                args = dynamic.newmark_defo_args(**args)

            if fluid:
                args = dynamic.newmark_velo_args(**args)
                cons['u'] = ucons * dynamic.ramp_up(t)

            # After remesh, also advance d_m via the generic named Newmark
            # helper (dm/dm0/vm0δt/am0δt2 — NOT the hardcoded d/d0/u0δt/a0δt2
            # keys, which belong to the persistent solid field).
            if has_remeshed and 'dm' in args:
                args = dynamic.newmark_defo_args_named(
                    args, name='dm', d0_name='dm0',
                    u0dt_name='vm0δt', a0dt2_name='am0δt2')

        # ------------------------------------------------------------
        # Predictor-state diagnostic before Newton
        # ------------------------------------------------------------
        if has_remeshed:

            Jf_pred = function.eval(Jfluid_bz, arguments=args)
            if Jf_pred.min() < predictor_J_safety:

                dm_pred = args['dm'].copy()
                theta = 1.0

                while Jf_pred.min() < predictor_J_safety and theta > 1e-6:

                    theta *= 0.5
                    args['dm'] = dm_n + theta * (dm_pred - dm_n)

                    Jf_pred = function.eval(Jfluid_bz, arguments=args)

                log.info(
                    f'[PREDICTOR DAMPING] theta={theta:.6e}, '
                    f'Jf_min={Jf_pred.min():.6e}'
                )


        # ------------------------------------------------------------
        # Solve nonlinear system
        # ------------------------------------------------------------        
        try:
            args = system.solve(constrain=cons, arguments=args, tol=1e-9)  
            if not has_remeshed and istep <= 10:
                Ru = numpy.asarray(function.eval(res_fluid_mom_old.derivative('utest'), arguments=args), dtype=float).reshape(-1)
                Rlift = numpy.asarray(function.eval(res_d_fluid_lift.derivative('dtest'), arguments=args), dtype=float).reshape(-1)
            
                Rreact = Tdu_old.T @ Ru[i_u_gamma]
                Rlift_gamma = Rlift[i_d_gamma]
            
                log.info(
                    f'[DISCRETE REACTION] step={istep} '
                    f'||lift||={numpy.linalg.norm(Rlift_gamma):.6e}, '
                    f'||reaction||={numpy.linalg.norm(Rreact):.6e}, '
                    f'||reaction-lift||={numpy.linalg.norm(Rreact-Rlift_gamma):.6e}, '
                    f'||reaction+lift||={numpy.linalg.norm(Rreact+Rlift_gamma):.6e}, '
                    f'rel={numpy.linalg.norm(Rreact-Rlift_gamma)/max(numpy.linalg.norm(Rlift_gamma),1e-30):.6e}'
                )

        except Exception:
            log.info('Newton failed. Mesh quality of current prediction / last available state:')

            if fluid:
                Jf = function.eval(Jfluid_bz, arguments=args)
                log.info(f'J fluid min/max: {Jf.min():.6e}, {Jf.max():.6e}')

            if solid:
                Js = function.eval(Jsolid_bz, arguments=args)
                dcyl = function.eval(d_cyl_bz, arguments=args)
                dcyl_mag = numpy.linalg.norm(dcyl / 'm', axis=1)

                log.info(f'J solid min/max: {Js.min():.6e}, {Js.max():.6e}')
                log.info(f'particle/interface displacement max: {dcyl_mag.max():.6e} m')

            raise
        if has_remeshed and istep == fluid_state['remesh_istep'] + 1:

            Ru = numpy.asarray(function.eval(fluid_state['reaction_res_fluid_mom'].derivative('utest'), arguments=args), dtype=float).reshape(-1)
        
            Rlamf = numpy.asarray(function.eval(fluid_state['reaction_res_solid_lamf'].derivative('dtest'), arguments=args), dtype=float).reshape(-1)
        
            i_u = fluid_state['reaction_i_u_gamma']
            i_d = fluid_state['reaction_i_d_gamma']
            Tdu = fluid_state['reaction_Tdu']
        
            Rreaction = Tdu.T @ Ru[i_u]
            Rlamf_gamma = Rlamf[i_d]
        
            diff = Rlamf_gamma - Rreaction
            summ = Rlamf_gamma + Rreaction
        
            log.info(
                f'[LAMF-REACTION] '
                f'||reaction||={numpy.linalg.norm(Rreaction):.6e}, '
                f'||lamf||={numpy.linalg.norm(Rlamf_gamma):.6e}, '
                f'||lamf-reaction||={numpy.linalg.norm(diff):.6e}, '
                f'||lamf+reaction||={numpy.linalg.norm(summ):.6e}, '
                f'rel={numpy.linalg.norm(diff)/max(numpy.linalg.norm(Rreaction),1e-30):.6e}'
            )

        if fluid:
            Jf = function.eval(Jfluid_bz, arguments=args)

            if not numpy.all(numpy.isfinite(Jf)) or Jf.min() <= 0:
                raise RuntimeError(f'Invalid fluid state after Newton: Jf_min={Jf.min():.6e}')

        if solid:
            Js = function.eval(Jsolid_bz, arguments=args)

            if not numpy.all(numpy.isfinite(Js)) or Js.min() <= 0:
                raise RuntimeError(f'Invalid solid state after Newton: Js_min={Js.min():.6e}')

        # ------------------------------------------------------------
        # Time-step size used for cumulative diagnostics
        # ------------------------------------------------------------
        current_t_s = float(t / 's') if dynamic else 0.0
        dt_s = current_t_s - previous_t_s if dynamic else 0.0
        previous_t_s = current_t_s

        # ------------------------------------------------------------
        # Diagnostics after successful solve
        # ------------------------------------------------------------

        if fluid:
            log.info(f'J fluid min/max: {Jf.min():.6e}, {Jf.max():.6e}')

        if solid:
            dcyl = function.eval(d_cyl_bz, arguments=args)
            dcyl_mag = numpy.linalg.norm(dcyl / 'm', axis=1)

            log.info(f'J solid min/max: {Js.min():.6e}, {Js.max():.6e}')
            log.info(f'particle/interface displacement max: {dcyl_mag.max():.6e} m')

        if has_remeshed and istep % 100 == 0:
            zipped = fluid_state['dm_zipped_interface']

            dm_g, ds_g, vm_g, vs_g, am_g, as_g = function.eval(
                [zipped.bind(ns.dm), zipped.bind(ns.d),
                zipped.bind(ns.vm), zipped.bind(ns.v),
                zipped.bind(ns.am), zipped.bind(ns.a)],
                arguments=args)

            dd = numpy.linalg.norm(
                numpy.asarray(dm_g / 'm')
                - (numpy.asarray(ds_g / 'm') - fluid_state['d_s_remesh_const_m']),
                axis=-1)

            dv = numpy.linalg.norm(
                numpy.asarray((vm_g - vs_g) / 'm/s'),
                axis=-1)

            da = numpy.linalg.norm(
                numpy.asarray((am_g - as_g) / 'm/s2'),
                axis=-1)

            R_m = float(domain.cylinder_radius / 'm')
            vs_ref = max(
                numpy.linalg.norm(numpy.asarray(vs_g / 'm/s'), axis=-1).max(),
                1e-30)

            as_ref = max(
                numpy.linalg.norm(numpy.asarray(as_g / 'm/s2'), axis=-1).max(),
                1e-30)

            log.info(f'[INTERFACE STEP] ||dm-(ds-ds_r)|| max={dd.max():.6e} m, /R={dd.max()/R_m:.6e}')
            log.info(f'[INTERFACE STEP] ||vm-vs|| max={dv.max():.6e} m/s, rel={dv.max()/vs_ref:.6e}')
            log.info(f'[INTERFACE STEP] ||am-as|| max={da.max():.6e} m/s2, rel={da.max()/as_ref:.6e}')

            urel_gamma = function.eval(fluid_state['dm_zipped_interface'].bind(ns.urel), arguments=args)
            urel_gamma = numpy.asarray(urel_gamma / 'm/s', dtype=float)
            urel_err = numpy.linalg.norm(urel_gamma, axis=-1)
            
            log.info(f'[INTERFACE STEP] ||urel|| max={urel_err.max():.6e} m/s')


        # ------------------------------------------------------------
        # Remeshing trigger — checked after every successful solve
        # ------------------------------------------------------------

        if fluid and solid and enable_remeshing:
            need_remesh = False

            # Quality-based trigger: minimum Jacobian below threshold
            if Jf.min() < remesh_J_threshold:
                log.info(f'Remesh triggered by mesh quality: Jf_min={Jf.min():.4f} < {remesh_J_threshold}')
                need_remesh = True
                
            if Jf.max() > remesh_Jmax_threshold:
                log.info(f'Remesh triggered by mesh stretching: Jf_max={Jf.max():.4f} > {remesh_Jmax_threshold}')
                need_remesh = True

            # Time-interval-based trigger (optional fallback)
            if remesh_interval_s and current_t_s >= next_remesh_time_s - 0.5 * dt_s:
                log.info(
                    f'Remesh triggered by time interval at t={current_t_s:.6f} s')
                
                need_remesh = True

            if need_remesh:
                remesh_count += 1
                if remesh_count == 1:
                    Ru_old = numpy.asarray(
                        function.eval(res_fluid_mom_old.derivative('utest'), arguments=args),
                        dtype=float).reshape(-1)
                
                    Rreact_old = Tdu_old.T @ Ru_old[i_u_gamma]  
                    fluid_state['Rreact_old_diag'] = Rreact_old.copy()
                
                    log.info(
                        f'[REACTION BEFORE REMESH] '
                        f'||R||={numpy.linalg.norm(Rreact_old):.6e}')

                # Capture old-mesh state for diagnostics BEFORE remeshing.
                # x_bz/u_bz/p_bz already carry consistent definitions
                # (spatial position, TOTAL physical velocity, pressure) both before and after any remesh, so no special-casing is needed.
                 
                xb_current_for_remesh = function.eval(x_bbz, arguments=args) / 'm'
                Jf_old_for_diag       = Jf.copy()
                old_fb_diag  = fluid_bezier   # keep reference to old sampler
                old_x_diag   = function.eval(x_bz, arguments=args) / 'm'
                old_u_diag   = numpy.linalg.norm(function.eval(u_bz, arguments=args) / 'm/s', axis=1)
               

                (
                    ns, res, cons, ucons, system,
                    fluid_bezier,
                    x_bz, u_bz, p_bz,
                    Jfluid_bz,
                    fluid_state,
                    args,
                    transfer_metrics,
                ) = remesh_fluid(
                    current_t_s,
                    xb_current_for_remesh,
                    domain,
                    ns,
                    solid,
                    fluid,
                    dynamic,
                    args,
                    topo,
                    fluid_state,
                    remesh_count,
                )
                
                fluid_state['remesh_istep'] = istep

                has_remeshed = True

                # Update remesh schedule
                if remesh_interval_s:
                    next_remesh_time_s = current_t_s + remesh_interval_s

                # Re-evaluate Jf on the fresh mesh
                Jf = function.eval(Jfluid_bz, arguments=args)

                # Capture new-mesh state for diagnostics using the freshly
                # rebuilt x_bz/u_bz/p_bz (already correct on the new mesh).
                new_x_diag  = function.eval(x_bz, arguments=args) / 'm'
                new_u_diag  = numpy.linalg.norm(function.eval(u_bz, arguments=args) / 'm/s', axis=1)
                

                remesh_diagnostics(
                    current_t_s   = current_t_s,
                    remesh_count  = remesh_count,
                    Jf_old        = Jf_old_for_diag,
                    Jf_new        = Jf,
                    old_fluid_bezier = old_fb_diag,
                    new_fluid_bezier = fluid_bezier,
                    xb_solid_m       = xb_current_for_remesh,
                    old_x_m          = old_x_diag,
                    new_x_m          = new_x_diag,
                    old_u_vals       = old_u_diag,
                    new_u_vals       = new_u_diag,
                    domain           = domain,
                    transfer_metrics = transfer_metrics,
                )

        xb_current = function.eval(x_bbz, arguments=args) / 'm'
        L_particle_m, B_particle_m, deformation_index, aspect_ratio, orientation_rad = particle_shape_from_boundary(xb_current)
        
        time_nd = epsdot_s * current_t_s
        theta_over_pi = orientation_rad 
        
        with particle_csv.open('a') as f:
            f.write(
                f'{current_t_s:.12e},{time_nd:.12e},{istep},'
                f'{L_particle_m:.12e},{B_particle_m:.12e},{deformation_index:.12e},'
                f'{aspect_ratio:.12e},{orientation_rad:.12e},{theta_over_pi:.12e}\n'
            )
        
        # ------------------------------------------------------------
        # Save deformed mesh every 100 timesteps
        # ------------------------------------------------------------
        # if istep % 1000 == 0:
        #     x, xb = function.eval([x_bz, x_bbz], arguments=args)

        #     if solid:
        #         xs = function.eval(xsolid_bz, arguments=args)

        #     with export.mplfigure(str('mesh.jpg'), dpi=200) as fig:
        #         ax = fig.add_subplot(111, title=f'Deformed mesh at t={t:.3s}', ylabel='[m]')

        #         # Fluid mesh
        #         if fluid:
        #             export.triplot(ax, x / 'm', hull=fluid_bezier.hull, linewidth=0.25)

        #         # Solid/particle mesh
        #         if solid:
        #             export.triplot(ax, xs / 'm', hull=solid_bezier.hull,linewidth=0.25)

        #         # Particle boundary
        #         export.triplot(ax, xb / 'm', hull=bbezier.tri, linewidth=1.0, linecolor='b')

        #         ax.set_aspect('equal')

        #         L = domain.channel_length / 'm'
        #         ax.set_xlim(-L, L)
        #         ax.set_ylim(-L, L)

        # ------------------------------------------------------------
        # Save velocity plot every 100 timesteps
        # ------------------------------------------------------------
        # if fluid and istep % 1000 == 0:

        #     u = function.eval(u_bz, arguments=args)
        #     velmag = numpy.linalg.norm(u / 'm/s', axis=1)

        #     with export.mplfigure(str('solution.jpg'), dpi=150) as fig:
        #         ax = fig.add_subplot(111, ylabel='[m]', title=f'Velocity magnitude at t={t:.3s}')
        #         im = ax.tripcolor(*(x / 'm').T, fluid_bezier.tri, velmag, shading='gouraud')

        #         fig.colorbar(im, orientation='horizontal', label='velocity [m/s]')

        #         export.triplot(ax, xb / 'm', hull=bbezier.tri, linewidth=1)

        #         ax.set_aspect('equal')

        #         L = domain.channel_length / 'm'
        #         ax.set_xlim(-L, L)
        #         ax.set_ylim(-L, L)


if __name__ == '__main__':
          cli.choose(main)


