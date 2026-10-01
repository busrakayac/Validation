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

    deformation_index = (
        (L_particle_m - B_particle_m) / (L_particle_m + B_particle_m)
        if (L_particle_m + B_particle_m) > 0 else numpy.nan
    )
    aspect_ratio = B_particle_m / L_particle_m

    return L_particle_m, B_particle_m, deformation_index, aspect_ratio, orientation_rad

def _dot2(a, b):
    """Contract two length-2 vectors.
    nutils.SI.Quantity does not implement .sum() or numpy.einsum, so an explicit component sum is the only way to contract dimensional arrays."""
    return a[0] * b[0] + a[1] * b[1]


def _snap_small(v, scale, rel=1e-12):
    """Snap |v| below rel*scale to exactly 0.
    Boundary coordinates that should be exactly zero often carry floating-point residues (e.g. 3.7e-21). Written out at %.16e these become literals that some Gmsh kernels treat as degenerate."""
    return 0.0 if abs(v) < rel * scale else v


def write_fluid_remesh_geo(boundary_xy_m, domain, filename, bezier_edges=None):
    """Write a fluid-only Gmsh .geo around the current deformed particle.

    The outer cross-slot geometry is identical to the original.  
    The particle boundary is described by the *current spatial* coordinates ``boundary_xy_m`` so that the new reference configuration of the fluid mesh coincides with the current deformed state.  
    After loading this mesh the fluid displacement field ``d`` must therefore be reset to zero (see ``remesh_fluid``)."""

    boundary_xy_m = numpy.asarray(boundary_xy_m, dtype=float)

    L = float(domain.channel_length / 'm')
    W = float(domain.channel_length / 'm')
    R = float(domain.cylinder_radius / 'm')

    # In write_fluid_remesh_geo(), replace W and the current mesh sizes:
    h_particle = float(domain.elemsize / 'm')
    h_far = h_particle * domain.coarsening
    r_grow = 8.0 * R

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

        # Distance-based refinement around the particle
        f.write('Field[1] = Distance;\n')
        f.write(f'Field[1].CurvesList = {{{line_string}}};\n')
        f.write('\nField[2] = Threshold;\n')
        f.write('Field[2].InField = 1;\n')
        f.write('Field[2].DistMin = 0;\n')
        f.write(f'Field[2].DistMax = {r_grow:.16e};\n')
        f.write(f'Field[2].SizeMin = {h_particle:.16e};\n')
        f.write(f'Field[2].SizeMax = {h_far:.16e};\n')
        f.write('\nBackground Field = 2;\n')


def remesh_fluid(current_t_s, xb_current_m, domain, ns, solid, fluid, dynamic, args, topo, fluid_state, remesh_index):
    """Remesh the fluid domain using the ALE/FSI remeshing architecture.

    Architecture:
    - d_s : solid displacement on topo['solid'] (ns.d); the solid topology and material reference configuration are preserved across fluid remeshing.
    - d_m : fluid mesh displacement on new_topo['fluid'] (ns.dm); the fresh fluid mesh becomes the new ALE reference configuration, so d_m = 0 at the remesh instant.
    - The mesh Newmark state is reconstructed on the fresh mesh so that the reset displacement is consistent with the transferred current mesh velocity and acceleration and their temporal history.
    - The physical fluid velocity is reconstructed as u_f = v_m + u_rel.
    - Pressure and fluid velocity-related fields are transferred from the old fluid mesh to the fresh fluid mesh at the same current physical positions.
    - Gradients and fluid stresses are evaluated with respect to the current ALE geometry x_m = X_f^new + d_m.
    - FSI traction coupling uses topology.locate + sample.zip to couple the persistent solid test functions to the fresh fluid stress field, since the solid and remeshed fluid no longer share a finite-element topology.
    - At the remesh instant, the reconstructed mesh state is synchronized with the solid interface so that v_m = v_s and a_m = a_s as transfer-consistency conditions.
    - fluid_state stores the current fluid topology and fields generically, so subsequent remeshes sample the latest fluid mesh and solution rather than the original mesh."""

    log.info(f'=== REMESHING FLUID at t={current_t_s:.6f} s ===')

    # ------------------------------------------------------------------
    # 1. Sample the OLD fluid solution generically via fluid_state
    # ------------------------------------------------------------------
    old_topo_fluid = fluid_state['topo_fluid']
    x_field        = fluid_state['x_field']
    v_field        = fluid_state['v_field']
    a_field        = fluid_state['a_field']
    urel_field     = fluid_state['urel_field']
    arel_field     = fluid_state['arel_field']
    p_field        = fluid_state['p_field']
    d_field        = fluid_state['d_field']
    d_name         = fluid_state['d_name']

    a0dt2_key      = fluid_state['a0dt2_key']

    old_sample = old_topo_fluid.sample('gauss', 4)

    a0dt2_expr = function.replace_arguments(d_field, [(d_name, a0dt2_key)])

    # urel's Newmark history is always named 'u0'/'a0δt' since the relative velocity field is always (re)created with argument name 'u'.
    u0_expr    = function.replace_arguments(urel_field, [('u', 'u0')])
    a0dt_urel_expr = function.replace_arguments(urel_field, [('u', 'a0δt')])

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

    old_uf = old_v + old_urel   # physical fluid velocity u_f = v_m + u_rel

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

    # ------------------------------------------------------------------
    # 2. Record the current SOLID boundary state for the P_Γ interface mapping.  
    
    # This uses a Gauss sample so it can also serve as the quadrature rule for the FSI traction coupling and
    # for the per-timestep d_m,Γ constraint, both built via zip+locate.
    # ------------------------------------------------------------------

    solid_cyl_boundary  = topo['solid'].boundary['cylinder']
    solid_cyl_gauss     = solid_cyl_boundary.sample('gauss', 4)
    solid_cyl_pts_now_m = numpy.asarray(function.eval(solid_cyl_gauss.bind(ns.x), arguments=args) / 'm', dtype=float)
    d_s_remesh_const_m  = numpy.asarray(function.eval(solid_cyl_gauss.bind(ns.d), arguments=args) / 'm', dtype=float)

    # Old mesh/solid interface velocity BEFORE remeshing
    if fluid_state.get('dm_zipped_interface') is not None:
        old_zipped = fluid_state['dm_zipped_interface']
        vm_old_g, vs_old_g = function.eval(
            [old_zipped.bind(v_field), old_zipped.bind(ns.v)],
            arguments=args)
    else:
        old_interface_located = old_topo_fluid.boundary['cylinder'].locate(
            x_field / Length('1m'),
            solid_cyl_pts_now_m,
            tol=float(domain.cylinder_radius / 'm') * 1e-4,
            arguments=args,
            skip_missing=False)
        
        vm_old_g = function.eval(old_interface_located.bind(v_field), arguments=args)
        vs_old_g = function.eval(solid_cyl_gauss.bind(ns.v), arguments=args)
    dv_old = numpy.linalg.norm(numpy.asarray((vm_old_g - vs_old_g) / 'm/s'), axis=-1)
    vs_old_ref = max(numpy.linalg.norm(numpy.asarray(vs_old_g / 'm/s'), axis=-1).max(), 1e-30)

    log.info(
        f'[INTERFACE BEFORE REMESH] ||v_m-v_s|| '
        f'max={dv_old.max():.6e} m/s, '
        f'rel={dv_old.max()/vs_old_ref:.6e}')

    # Extract the solid's quadratic (P2) boundary edges as (P0, Pmid, P1) triples, in element order.  
    # A bezier-3 sample yields exactly these three points per element.  
    # These are converted to quadratic Bezier arcs in the new .geo so the fresh fluid boundary reproduces the solid interface. 

    solid_cyl_bez3 = solid_cyl_boundary.sample('bezier', 3)
    solid_cyl_bez3_pts_m = numpy.asarray(function.eval(solid_cyl_bez3.bind(ns.x), arguments=args) / 'm', dtype=float)
    solid_p2_edges = [solid_cyl_bez3_pts_m[solid_cyl_bez3.getindex(ielem)] for ielem in range(solid_cyl_bez3.nelems)]


    # Verify the element order actually traces a closed loop: the end point of each edge must coincide with the start point of the next, 
    # and the last must close back onto the first.  Element ordering happens to be traversing for the present topology, 
    # but that is an assumption, not a guarantee — if it breaks, the Bezier arcs would be stitched in the
    # wrong order and the fresh boundary would be garbage.  If the check fails we reorder greedily by nearest-neighbour chaining.

    def _edges_form_closed_loop(edges, tol):
        for e_this, e_next in zip(edges, edges[1:] + edges[:1]):
            if numpy.linalg.norm(e_this[2] - e_next[0]) > tol:
                return False
        return True

    _loop_tol = float(domain.cylinder_radius / 'm') * 1e-6
    if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
        log.info('[REMESH] solid P2 edges are not in traversing order; '
                 'reordering by endpoint chaining.')
        remaining = list(solid_p2_edges)
        ordered   = [remaining.pop(0)]
        while remaining:
            tail = ordered[-1][2]
            j = min(range(len(remaining)),
                    key=lambda k: min(numpy.linalg.norm(remaining[k][0] - tail),
                                      numpy.linalg.norm(remaining[k][2] - tail)))
            e = remaining.pop(j)
            if (numpy.linalg.norm(e[2] - tail) < numpy.linalg.norm(e[0] - tail)): e = e[::-1]          # flip edge so it continues the chain
            ordered.append(e)
        solid_p2_edges = ordered
        if not _edges_form_closed_loop(solid_p2_edges, _loop_tol):
            raise RuntimeError(
                'solid cylinder boundary edges do not form a closed loop; '
                'cannot build a consistent fresh fluid boundary.')
        log.info('[REMESH] edge reordering succeeded; closed loop verified.')
    else:
        log.info('[REMESH] solid P2 edge connectivity verified (closed loop).')


    # ------------------------------------------------------------------
    # 3. Generate new fluid mesh, in its own gmsh "space" 
    # so that Nutils' Topology.locate / Sample.zip can combine it with the persistent solid topology (which lives in the default space 'X').
    # ------------------------------------------------------------------

    # Use the P2 edge start points, in element order, as the .geo boundary vertices.  
    # Element order already traverses the closed loop, so no angular sorting is needed.

    xb_ordered = numpy.array([e[0] for e in solid_p2_edges], dtype=float)
    geo_file   = Path(f'remesh_fluid_{current_t_s:.6f}s.geo')
    write_fluid_remesh_geo(xb_ordered, domain, geo_file,
                           bezier_edges=solid_p2_edges)

    fluid_space_name = f'Xfluid{remesh_index}'
    msh_file = geo_file.with_suffix('.msh')

    subprocess.run(
        ['gmsh', str(geo_file), '-2', '-order', '2',
        '-format', 'msh2', '-o', str(msh_file)],
        check=True)

    mesh = meshio.read(msh_file)
    cylinder_id = int(mesh.field_data['cylinder'][0])

    # Gmsh line3 connectivity: endpoint, endpoint, quadratic midpoint.
    fluid_edge_nodes = []
    for cells, physical in zip(
            mesh.cells, mesh.cell_data['gmsh:physical']):
        if cells.type == 'line3':
            fluid_edge_nodes.extend(
                nodes for nodes, group in zip(cells.data, physical)
                if group == cylinder_id)

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
                max(numpy.linalg.norm(p0 - solid_edge[0]),
                    numpy.linalg.norm(p1 - solid_edge[2])),
                max(numpy.linalg.norm(p1 - solid_edge[0]),
                    numpy.linalg.norm(p0 - solid_edge[2])))

        j = min(unused, key=endpoint_error)
        error = endpoint_error(j)
        if error > endpoint_tol:
            raise RuntimeError(
                f'Cannot match fluid edge to solid edge: '
                f'endpoint error={error:.3e} m')

        unused.remove(j)
        midpoint_node = fluid_edge_nodes[j][2]
        mesh.points[midpoint_node, :2] = solid_edge[1]

    meshio.write(msh_file, mesh, file_format='gmsh22', binary=False)

    new_topo, new_geom_bare = gmsh(
        msh_file, space=fluid_space_name)
    # Compare each persistent solid P2 edge with its fresh fluid boundary edge.
    fluid_bz3 = new_topo['fluid'].boundary['cylinder'].sample('bezier', 3)
    fluid_pts = numpy.asarray(
        function.eval(fluid_bz3.bind(new_geom_bare)), dtype=float)
    fluid_edges = [
        fluid_pts[fluid_bz3.getindex(i)]
        for i in range(fluid_bz3.nelems)
    ]

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
        # Match by both endpoints; allow opposite edge orientation.
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
        midpoint_errors.append(
            numpy.linalg.norm(solid_edge[1] - fluid_edges[j][1]))

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

    # ------------------------------------------------------------------
    # 4. Define new fluid fields on new_topo['fluid']
    # new_ns: unknown fields
    # ------------------------------------------------------------------
    new_ns = ns   # reuse namespace object; overwrite fluid-specific entries

    dm_basis = new_topo['fluid'].basis('std', degree=2)
    new_ns.dm = function.field('dm', dm_basis, shape=(2,)) * domain.cylinder_radius
    if dynamic:
        new_ns.vm, new_ns.am = dynamic.newmark_defo_named(new_ns.dm, name='dm', d0_name='dm0', u0dt_name='vm0δt', a0dt2_name='am0δt2')
    else:
        new_ns.vm = Velocity.wrap(function.zeros((2,)))
        new_ns.am = Acceleration.wrap(function.zeros((2,)))

    new_ns.urel = new_topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
    if dynamic:
        new_ns.arel = dynamic.newmark_velo(new_ns.urel)
        new_ns.u_i  = 'vm_i + urel_i'
    else:
        new_ns.u_i  = 'urel_i'

    new_ns.p = (new_topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius)

    new_ns.utest = function.replace_arguments(new_ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
    new_ns.ptest = function.replace_arguments(new_ns.p,    'p:ptest') / fluid.viscosity / fluid.velocity**2

    # ------------------------------------------------------------------
    # 5. ns_f: gradients/measures on the MOVING ALE geometry
    # fluid physics on the current ALE geometry 
    # ------------------------------------------------------------------
    ns_f        = Namespace()
    ns_f.δ      = new_ns.δ
    ns_f.ρf     = new_ns.ρf
    ns_f.μf     = new_ns.μf
    ns_f.xfnew  = new_geom
    ns_f.dm     = new_ns.dm
    ns_f.xm_i   = 'xfnew_i + dm_i'

    ns_f.define_for('xm', gradient='∇', normal='n', jacobians=('dV', 'dS'))

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

    ns_f.traction_i = 'σ_ij n_j'   # fluid stress vector, for the FSI coupling
    new_ns.tsolid_i = 'σcauchy_ij n_j'

    ns_f.utest   = new_ns.utest
    ns_f.ptest   = new_ns.ptest
    ns_f.dtest   = new_ns.dtest

    # ns_m: mesh-extension namespace, ∇ref w.r.t. the epoch reference new_geom

    ns_m        = Namespace()
    ns_m.δ      = new_ns.δ
    ns_m.xfnew  = new_geom

    ns_m.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))

    ns_m.dm     = new_ns.dm
    ns_m.xm_i   = 'xfnew_i + dm_i'
    ns_m.Fm_ij  = '∇ref_j(xm_i)'
    ns_m.Jm     = numpy.linalg.det(ns_m.Fm)
    ns_m.Fminv  = numpy.linalg.inv(ns_m.Fm)
    ns_m.Pmesh_ij = '2 (Fm_ij - Fminv_ji)'
    ns_m.dmtest = function.replace_arguments(new_ns.dm, 'dm:dmtest') / (domain.cylinder_radius**2 * Pressure('1Pa'))

    # ------------------------------------------------------------------
    # 6. Build the new residual
    # ------------------------------------------------------------------
    new_res = 0.

    # Solid momentum — unchanged, persistent topology/field
    new_res += topo['solid'].integral('(∇ref_j(dtest_i) P_ij + dtest_i ρs (a_i - g_i) + dtest_i cmforce_i) dVref' @ new_ns, degree=4)
    new_res += topo['solid'].integral('cmtest_i x_i dVref' @ new_ns, degree=4) / domain.cylinder_radius**3

    # Mesh extension for d_m (elasticity-type smoothing, epoch reference new_geom)
    new_res += Pressure('1Pa') * new_topo['fluid'].integral('∇ref_j(dmtest_i) Pmesh_ij dVref' @ ns_m, degree=4)

    # Fluid momentum and incompressibility, gradients w.r.t. moving x_m
    new_res += new_topo['fluid'].integral('(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns_f, degree=4)
    new_res += new_topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns_f, degree=4)

    # ------------------------------------------------------------------
    # FSI coupling: fluid traction acting on the solid boundary.
    #
    # After remeshing, fluid and solid no longer share a common finite-element topology, so the original "volume lift" identity
    # ∮ d·σ·n dΓ = ∫ (∇d:σ + d·ρDuDt) dV (valid only when d and u share dofs on the SAME mesh) no longer applies.  
    # Instead we impose the coupling directly as a boundary traction term, evaluated at matching physical points via
    # Topology.locate, combined via Sample.zip so that the solid's own Gauss quadrature provides the (correct, persistent) weights:
    # ∮_cylinder dtest_i (σ_ij n_j) dΓ

    locate_tol_m = float(domain.cylinder_radius / 'm') * 1e-4

    try:
        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(
            new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False)
    except Exception:
        log.info(
            f'[REMESH] locate failed at tol={locate_tol_m:.3e} m; '
            f'retrying with a looser tolerance.')
        
        locate_tol_m *= 100.

        fluid_traction_located = new_topo['fluid'].boundary['cylinder'].locate(new_geom_bare, solid_cyl_pts_now_m, tol=locate_tol_m, skip_missing=False)
        
    log.info(f'[REMESH] FSI traction coupling located at tol={locate_tol_m:.3e} m')

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

    

    # Diagnostic trace-space map between solid and fresh-fluid interface DOFs
    Bf4 = function.eval(zipped_traction.bind((new_ns.dm / Length('1m')).derivative('dm')))
    Bf = Bf4.reshape(Bf4.shape[0] * Bf4.shape[1], -1)

    dm_norm = numpy.linalg.norm(Bf, axis=0)

    i_dm_gamma = numpy.flatnonzero(dm_norm > 1e-12 * dm_norm.max())
    i_dm_gamma_scalar = numpy.unique(i_dm_gamma // 2)
    i_dm_expected = numpy.sort((2 * i_dm_gamma_scalar[:, None] + numpy.arange(2)).ravel())

    if not numpy.array_equal(numpy.sort(i_dm_gamma), i_dm_expected):
        raise RuntimeError('Unexpected dm interface DOF ordering.')

    log.info(
        f'[INTERFACE TRACE BASIS] '
        f'scalar P2 trace dofs={len(i_dm_gamma_scalar)}'
    )

    new_res += zipped_traction.integral(_dot2(new_ns.dtest, ns_f.traction) * new_ns.dS)

    # Interface target 
    d_s_remesh_coeffs = numpy.array(args['d'], copy=True)

    d_s_remesh_field = function.replace_arguments(new_ns.d, {'d': d_s_remesh_coeffs})
    dm_interface_target_expr = (new_ns.d - d_s_remesh_field)

    lam_basis = dm_basis[i_dm_gamma_scalar]
    new_ns.lam = (function.field('lam', lam_basis, shape=(2,)) * solid.shear_modulus)
    Bl4 = function.eval(zipped_traction.bind((new_ns.lam / solid.shear_modulus).derivative('lam')))
    Bl = Bl4.reshape(Bl4.shape[0] * Bl4.shape[1], -1)
    Ctrace = Bl.T @ Bf[:, i_dm_gamma]

    log.info(
        f'[INTERFACE TRACE BASIS] '
        f'lambda dofs={Bl.shape[1]}, '
        f'rank={numpy.linalg.matrix_rank(Ctrace)}/{len(i_dm_gamma)}')
    

    new_ns.lamtest = function.replace_arguments(new_ns.lam, 'lam:lamtest') / (solid.shear_modulus * domain.cylinder_radius**2)

    # Reaction term: +∫ λ·δd_m  ONLY.
    # The multiplier must NOT push back on the solid (no −∫ λ·δd_s term).
    # The mesh-extension problem is artificial numerical machinery; its reaction is not a physical force.  
    # The genuine fluid force on the solid is already supplied by the traction term above.  
    # The coupling is therefore one-directional in the kinematic sense:
    
    # d_s  ——> interface constraint ——>  d_m
    new_res += zipped_traction.integral(_dot2(ns_m.dmtest, new_ns.lam) * new_ns.dS)

    # Constraint term: ∫ δλ·(d_m − (d_s − d_s^remesh))
    new_res += zipped_traction.integral(_dot2(new_ns.lamtest, new_ns.dm - dm_interface_target_expr) * new_ns.dS)

    # ------------------------------------------------------------------
    # 7. Rebuild constraints
    #
    # d_m is fixed to zero only on the OUTER fluid boundary.  
    # The cylinder (interface) boundary is intentionally NOT constrained here — its value is d_m,Γ = P_Γ(d_s - d_s^remesh), 
    # which must be refreshed every timestep as the solid keeps moving.  
    # ------------------------------------------------------------------
    
    new_cons = {}

    sqr_ds = topo.boundary.integral('d_k d_k dSref' @ new_ns, degree=4) / domain.cylinder_radius**3
    new_cons = System(sqr_ds, trial='d').solve_constraints(droptol=1e-9, constrain=new_cons)

    # Outer (stationary) fluid boundary: dm = 0.  This constraint is FIXED for the whole epoch, 
    # so we keep it separately as dm_outer_cons and rebuild the full dm constraint from it every timestep.  
    # Nutils treats finite entries as already-constrained and skips them, giving d_m,Γ^{n+1} = d_m,Γ^n while the solid keeps moving.

    sqr_dm_outer = new_topo['fluid'].boundary['outer'].integral('dm_k dm_k dSref' @ ns_m, degree=4) / domain.cylinder_radius**3

    dm_outer_only = System(sqr_dm_outer, trial='dm').solve_constraints(droptol=1e-9)
    dm_outer_cons = dm_outer_only['dm'].copy()

    # The interface value of d_m is NOT constrained here: it is imposed monolithically by the Lagrange-multiplier residual built above, 
    # so Newton can update d_s and d_m,Γ together.  
    # Only the stationary outer fluid boundary is a Dirichlet constraint.

    new_cons = dict(new_cons)
    new_cons['dm'] = dm_outer_cons.copy()

    # u (urel): no-slip + inflow on fluid boundary

    ns_bc = Namespace()
    ns_bc.urel  = new_ns.urel
    ns_bc.xfnew = new_geom

    ns_bc.define_for('xfnew', gradient='∇ref', jacobians=('dVref', 'dSref'))

    ns_bc.δ = new_ns.δ
    ns_bc.epsdot = fluid.velocity / (2 * domain.cylinder_radius)
    ns_bc.uext_i = 'epsdot (δ_i0 xfnew_0 - δ_i1 xfnew_1)'
    sqr_u = new_topo['fluid'].boundary['cylinder'].integral('urel_i urel_i dSref' @ ns_bc, degree=4) / (2 * domain.cylinder_radius * fluid.velocity**2)
    sqr_u += new_topo['fluid'].boundary['outer'].integral('(urel_i - uext_i) (urel_i - uext_i) dSref' @ ns_bc, degree=4) / (2 * domain.cylinder_radius * fluid.velocity**2)


    new_cons = System(sqr_u, trial='u').solve_constraints(droptol=1e-9, constrain=new_cons)
    new_ucons = new_cons['u'].copy()
    pcons = numpy.full(function.arguments_for(new_ns.p)['p'].shape, numpy.nan)
    pcons[0] = 0.0
    new_cons['p'] = pcons

    # ------------------------------------------------------------------
    # 8. Build new Newton system
    # ------------------------------------------------------------------
    new_system = System(new_res, trial=['d', 'cm', 'dm', 'u', 'p', 'lam'], test=['dtest', 'cmtest', 'dmtest', 'utest', 'ptest', 'lamtest'])

    # ------------------------------------------------------------------
    # 9. Transfer solution to new mesh (physical coordinates, physical velocity, full Newmark history for both d_m and u_rel)
    # ------------------------------------------------------------------
    # Gauss quadrature is the natural sample for an L2 / least-squares projection (bezier sampling is a visualisation/interpolation device).
     
    new_dof_sample = new_topo['fluid'].sample('gauss', 4)
    new_dof_x_m    = numpy.asarray(new_dof_sample.eval(new_geom) / 'm', dtype=float)

    # Field transfer: evaluate the OLD fields directly at the NEW sample points by locating those points in the old physical ALE mesh.  
    # This is exact (uses the old FE basis) and, unlike LinearNDInterpolator with fill_value=0, does not silently zero points that fall outside
    # the convex hull of the old Gauss cloud — which happens precisely near the cylinder, where accuracy matters most.
    # Independent of locate_tol_m, which the FSI-coupling block above may have loosened on retry; that must not silently affect field transfer.

    transfer_tol_m = float(domain.cylinder_radius / 'm') * 1e-4
    transfer_via_locate = False
    try:
        # locate must be performed in the OLD mesh's *physical* (deformed ALE) coordinates, because new_dof_x_m are physical positions.
        # x_field is exactly that (= X_old + d_m_old); strip its SI unit and supply args so the displacement part evaluates.

        old_located = old_topo_fluid.locate(x_field / Length('1m'), new_dof_x_m, tol=transfer_tol_m, arguments=args, skip_missing=False)
        
        (t_uf, t_vm, t_am, t_urel, t_arel, t_p, t_a0dt2, t_u0, t_a0dt_urel) = function.eval([
                old_located.bind(v_field + urel_field),
                old_located.bind(v_field),
                old_located.bind(a_field),
                old_located.bind(urel_field),
                old_located.bind(arel_field),
                old_located.bind(p_field),
                old_located.bind(a0dt2_expr),
                old_located.bind(u0_expr),
                old_located.bind(a0dt_urel_expr)], arguments=args)


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

        log.info('[REMESH] field transfer via Topology.locate (exact FE evaluation)')
    except Exception as ex:
        log.info(f'[REMESH] locate-based transfer unavailable ({ex}); '
                 f'falling back to scattered interpolation.')

    if not transfer_via_locate:

        def _I(vals):
            return LinearNDInterpolator(old_x_m, vals, fill_value=0.0)(new_dof_x_m)

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


    beta, gamma, dt = dynamic.beta, dynamic.gamma, float(dynamic.timestep / 's')

    # ------------------------------------------------------------------
    # 10. Project all interpolated fields onto new dof arrays via L2 solve
    # ------------------------------------------------------------------
    R_scale   = float(domain.cylinder_radius / 'm')
    u_scale   = float(fluid.velocity / 'm/s')
    p_scale_v = float(fluid.viscosity * fluid.velocity / domain.cylinder_radius / 'Pa')

    def _project_vector(target_x, target_y, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=2, shape=(2,))
        ns_p.ftarget = new_dof_sample.asfunction(numpy.stack([target_x / scale, target_y / scale], axis=1).astype(float))

        sqr = new_dof_sample.integral(((ns_p.fproj - ns_p.ftarget) ** 2).sum(-1))

        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']

    # Synchronize current mesh state at remeshing

    def _project_mesh_state(tx, ty, interface_target):
        ns_h = Namespace()
        ns_h.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
        ns_h.htarget = new_dof_sample.asfunction(numpy.stack([tx / R_scale, ty / R_scale], axis=1).astype(float)) * domain.cylinder_radius

        sqr_gamma = zipped_traction.integral(_dot2(ns_h.h - interface_target, ns_h.h - interface_target) * new_ns.dS) / domain.cylinder_radius**3

        hcons = System(sqr_gamma, trial='h').solve_constraints(
            droptol=1e-9,
            constrain={'h': dm_outer_cons.copy()},
            arguments=args,
        )

        sqr_bulk = new_dof_sample.integral((((ns_h.h - ns_h.htarget) / domain.cylinder_radius)**2).sum(-1))

        return System(sqr_bulk, trial='h').solve(constrain=hcons, arguments={})['h']
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
    
            sqr_gamma = zipped_traction.integral(
                _dot2(ns_h.h - interface_target, ns_h.h - interface_target) * new_ns.dS
            ) / domain.cylinder_radius**3
    
            hcons = System(sqr_gamma, trial='h').solve_constraints(
                droptol=1e-9, constrain={'h': dm_outer_cons.copy()}, arguments=args)
    
            res_h = Pressure('1Pa') * new_topo['fluid'].integral(
                '∇ref_j(htest_i) Ph_ij dVref' @ ns_h, degree=4)
    
            return System(res_h, trial='h', test='htest').solve(
                constrain=hcons, arguments={'h': numpy.nan_to_num(hcons['h'], nan=0.)}, tol=1e-12)['h']

    uf_proj_dofs = _project_vector(new_uf_x, new_uf_y, u_scale)

    V_mesh = _extend_mesh_state(new_ns.v * dynamic.timestep)

    urel_new_dofs = (
        uf_proj_dofs - V_mesh * R_scale / (dt * u_scale)
    )

    # Gao's ramp is complete at the first remesh (0.201 s > 0.1 s).
    u_fixed = numpy.isfinite(new_ucons)
    urel_new_dofs[u_fixed] = new_ucons[u_fixed]

    a0dt_new = _project_vector(
        new_a0dt_urel_x, new_a0dt_urel_y, u_scale)
    A_rel = _project_vector(
        new_arel_x * dt, new_arel_y * dt, u_scale)

    u0_new = (
        urel_new_dofs
        - gamma * A_rel
        - (1 - gamma) * a0dt_new
    )
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

    log.info(
        f'[CURRENT STATE] V error = '
        f'{numpy.linalg.norm(vm_new_dofs - V_mesh):.6e}'
    )

    log.info(
        f'[CURRENT STATE] A error = '
        f'{numpy.linalg.norm(am_new_dofs - A_mesh):.6e}'
    )

    def _project_scalar(target, scale):
        ns_p = Namespace()
        ns_p.fproj = new_topo['fluid'].field('fproj', btype='std', degree=1)
        ns_p.ftarget = new_dof_sample.asfunction((target / scale).astype(float))
        sqr = new_dof_sample.integral((ns_p.fproj - ns_p.ftarget) ** 2)
        return System(sqr, trial='fproj').solve(constrain={}, arguments={})['fproj']


    p_proj_dofs  = _project_scalar(new_p_Pa, p_scale_v)
    log.info(f'[PRESSURE GAUGE] p_proj[0]={p_proj_dofs[0]:.6e}, physical={p_proj_dofs[0]*p_scale_v:.6e} Pa')

    V_mesh_direct = _project_vector(new_vm_x * dt, new_vm_y * dt, R_scale)
    A_mesh_direct = _project_vector(new_am_x * dt**2, new_am_y * dt**2, R_scale)

    # Relative-acceleration history check

    arel_new_dofs = a0dt_new + (urel_new_dofs - u0_new - a0dt_new) / gamma
    arel_err = numpy.linalg.norm((arel_new_dofs - A_rel) * u_scale / dt)
    log.info(f'[AREL HISTORY CHECK] error={arel_err:.6e} m/s2')

    vm_target = numpy.column_stack((new_vm_x, new_vm_y))
    am_target = numpy.column_stack((new_am_x, new_am_y))

    def check_vm_transfer(label, coeffs):
        val = function.eval(new_dof_sample.bind(new_ns.dm / dynamic.timestep), arguments={'dm': coeffs})
        val = numpy.asarray(val / 'm/s', dtype=float)
        err = numpy.linalg.norm(val - vm_target, axis=1)
        rms = numpy.sqrt(numpy.mean(err**2))
        ref = numpy.sqrt(numpy.mean(numpy.linalg.norm(vm_target, axis=1)**2))
        log.info(
            f'[VM TRANSFER] {label}: max={err.max():.6e} m/s, '
            f'RMS={rms:.6e} m/s, relRMS={rms/max(ref,1e-30):.6e}'
        )

    def check_am_transfer(label, coeffs):
        val = function.eval(new_dof_sample.bind(new_ns.dm / dynamic.timestep**2), arguments={'dm': coeffs})
        val = numpy.asarray(val / 'm/s2', dtype=float)
        err = numpy.linalg.norm(val - am_target, axis=1)
        rms = numpy.sqrt(numpy.mean(err**2))
        ref = numpy.sqrt(numpy.mean(numpy.linalg.norm(am_target, axis=1)**2))
        log.info(
            f'[AM TRANSFER] {label}: max={err.max():.6e} m/s2, '
            f'RMS={rms:.6e} m/s2, relRMS={rms/max(ref,1e-30):.6e}'
        )

    check_vm_transfer('direct L2 projection', V_mesh_direct)
    check_am_transfer('direct L2 projection', A_mesh_direct)

    check_vm_transfer('synchronized current state', V_mesh)
    check_am_transfer('synchronized current state', A_mesh)

    # Direct projection with stationary outer-boundary correction
    outer_mask = numpy.isfinite(dm_outer_cons)

    V_mesh_direct_outer = V_mesh_direct.copy()
    A_mesh_direct_outer = A_mesh_direct.copy()

    V_mesh_direct_outer[outer_mask] = 0.
    A_mesh_direct_outer[outer_mask] = 0.

    check_vm_transfer('direct L2 projection + outer BC', V_mesh_direct_outer)
    check_am_transfer('direct L2 projection + outer BC', A_mesh_direct_outer)

    # ------------------------------------------------------------------
    # 11. Check synchronized interface state at remesh    
    # ------------------------------------------------------------------

    ns_check = Namespace()

    ns_check.h = function.replace_arguments(new_ns.dm, [('dm', 'h')])
    vm_gamma, vs_gamma = function.eval(
        [zipped_traction.bind(ns_check.h / dynamic.timestep),
        zipped_traction.bind(new_ns.v)],
        arguments=dict(args, h=vm_new_dofs))

    am_gamma, as_gamma = function.eval(
        [zipped_traction.bind(ns_check.h / dynamic.timestep**2),
        zipped_traction.bind(new_ns.a)],
        arguments=dict(args, h=am_new_dofs))
    
    dv = numpy.linalg.norm(numpy.asarray((vm_gamma - vs_gamma) / 'm/s'), axis=-1)
    da = numpy.linalg.norm(numpy.asarray((am_gamma - as_gamma) / 'm/s2'), axis=-1)

    vs_ref = max(numpy.linalg.norm(numpy.asarray(vs_gamma / 'm/s'), axis=-1).max(), 1e-30)
    as_ref = max(numpy.linalg.norm(numpy.asarray(as_gamma / 'm/s2'), axis=-1).max(), 1e-30)

    log.info(f'[INTERFACE AT REMESH] ||a_m-a_s|| max={da.max():.6e} m/s2, rel={da.max()/as_ref:.6e}')
    log.info(f'[INTERFACE AT REMESH] ||v_m-v_s|| max={dv.max():.6e} m/s, rel={dv.max()/vs_ref:.6e}')

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

    # Acceleration field immediately after remesh
    am_restart = numpy.asarray(
        function.eval(new_dof_sample.bind(new_ns.am), arguments=new_args) / 'm/s2',
        dtype=float,
    )

    # Old acceleration evaluated at exactly the same new-mesh sample points
    am_target = numpy.column_stack((new_am_x, new_am_y))

    # Spatial error introduced by remeshing/projection
    am_transfer_error = am_restart - am_target

    errmag = numpy.linalg.norm(am_transfer_error, axis=1)
    ierr = numpy.argmax(errmag)

    log.info(
        f'[AM SEED] max|a_new-a_old|={errmag[ierr]:.6e} m/s2 at '
        f'x=({new_dof_x_m[ierr,0]:.6e}, {new_dof_x_m[ierr,1]:.6e}) m'
    )
    
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

        DuDt_field = ns_f.DuDt,

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

        # Acceleration seed diagnostic
        transfer_sample       = new_dof_sample,
        transfer_x0_m         = new_dof_x_m.copy(),
        am_restart_ms2        = am_restart.copy(),
        am_transfer_error_ms2 = am_transfer_error.copy(),
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
    with export.mplfigure(f'remesh_{remesh_count:03d}_mesh_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

        ax1 = fig.add_subplot(121, title=f'ALE mesh before remesh\nJf_min={Jf_old.min():.3f}')
        export.triplot(ax1, old_x_m, hull=old_fluid_bezier.hull, linewidth=0.25)
        closed = numpy.vstack([xb_solid_m, xb_solid_m[:1]])
        ax1.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
        ax1.set_aspect('equal')
        ax1.set_xlim(-L_plot, L_plot)
        ax1.set_ylim(-L_plot, L_plot)
        ax1.legend(fontsize=6)

        ax2 = fig.add_subplot(122, title=f'New fluid mesh after remesh\nJf_min={Jf_new.min():.3f}')
        export.triplot(ax2, new_x_m, hull=new_fluid_bezier.hull, linewidth=0.25)
        ax2.plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1.5, label='solid boundary')
        ax2.set_aspect('equal')
        ax2.set_xlim(-L_plot, L_plot)
        ax2.set_ylim(-L_plot, L_plot)
        ax2.legend(fontsize=6)

        fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

    # ------------------------------------------------------------------
    # Plot 2: velocity magnitude comparison (old vs new mesh)
    # ------------------------------------------------------------------
    vmin = min(old_u_vals.min(), new_u_vals.min())
    vmax = max(old_u_vals.max(), new_u_vals.max())

    with export.mplfigure(f'remesh_{remesh_count:03d}_velocity_t{current_t_s:.6f}s.jpg', dpi=200) as fig:

        ax1 = fig.add_subplot(121, title='|u| before remesh  [m/s]')
        im = ax1.tripcolor(
            old_x_m[:, 0], old_x_m[:, 1], old_fluid_bezier.tri,
            old_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
        )
        ax1.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
        ax1.set_aspect('equal')
        ax1.set_xlim(-L_plot, L_plot)
        ax1.set_ylim(-L_plot, L_plot)

        ax2 = fig.add_subplot(122, title='|u| after remesh  [m/s]')
        im2 = ax2.tripcolor(
            new_x_m[:, 0], new_x_m[:, 1], new_fluid_bezier.tri,
            new_u_vals, shading='gouraud', vmin=vmin, vmax=vmax,
        )
        ax2.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=1)
        ax2.set_aspect('equal')
        ax2.set_xlim(-L_plot, L_plot)
        ax2.set_ylim(-L_plot, L_plot)

        fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', label='|u| [m/s]', shrink=0.6)
        fig.suptitle(f'Remesh #{remesh_count}  t={current_t_s:.4e} s', fontsize=9)

 
@dataclass
class Domain:
    '''Parameters for the domain geometry.

    The default values match Table 1 of Turek and Hron [1].'''

    channel_length: Length = Length('0.0006m')
    cylinder_radius: Length = Length('0.00003m')
    
    elemsize: Length = Length('0.00002m')
    coarsening: float = 2.

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

    timestep: Time = Time('0.5ms') # simulation time step size
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
        res += topo['fluid'].integral('(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns, degree=4)

        # Incompressibility: div u = 0
        ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns, degree=4)

        if solid:
            # The action of fluid stress on the solid is imposed weakly by lifting the test functions into the fluid domain, 
            # using the identity ∮ d·σ·n = ∫ ∇(d·σ) = ∫ (∇d:σ + d·∇σ) = ∫ (∇d:σ + ρf d·Du/Dt). 
            # We need the inverse of the dfluid mask to exclude dofs without support on the boundary.

            dsolid = ~dfluid # true if dof is (partially) supported by solid domain
            res += function.replace_arguments(topo['fluid'].integral('(dtest_i ρf DuDt_i + ∇_j(dtest_i) σ_ij) dV' @ ns, degree=4), {'dtest': function.arguments_for(res)['dtest'] * dsolid})


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

    enable_remeshing = True
    remesh_J_threshold = 0.6
    predictor_J_safety = 0.4

    # Also allow a time-interval-based fallback trigger (set to None to disable).
    # Set to a small value for an early forced remesh to test the procedure before the mesh actually degrades. Once validated, set back to None.

    remesh_interval_s  = None  # first remesh forced at t=0.0001 s for testing

    next_remesh_time_s = remesh_interval_s if remesh_interval_s else float('inf')

    # Track whether we have remeshed (changes which Newmark keys exist)
    has_remeshed = False

    particle_csv = Path('D_remesh_06.csv')
    particle_csv.write_text('time,step,L_particle,B_particle,deformation_index\n')

    for istep, t in enumerate(log.iter.fraction('timestep', dynamic.times) if dynamic else [Time.wrap(float('inf'))], start=1):

        if has_remeshed and 'dm' in args:
            dm_n = args['dm']
            dm0 = args['dm0']
            V0 = args['vm0δt']
            A0 = args['am0δt2']
            δA = (dm_n - dm0 - V0 - 0.5*A0) / dynamic.beta
            Vn = V0 + A0 + dynamic.gamma*δA
            An = A0 + δA

            dm_before_solve = dm_n.copy()
            Vn_before_solve = Vn.copy()
            An_before_solve = An.copy()

            def _Jf_with_dm(dm_test):
                test_args = dict(args)
                test_args['dm'] = dm_test
                return function.eval(Jfluid_bz, arguments=test_args)

            J_cur = _Jf_with_dm(dm_n)
            J_vel = _Jf_with_dm(dm_n + Vn)
            J_acc = _Jf_with_dm(dm_n + 0.5*An)
            J_full = _Jf_with_dm(dm_n + Vn + 0.5*An)
            J_lin = _Jf_with_dm(dm_n + (dm_n - dm0))

            norm_An = numpy.linalg.norm(An)
            norm_A0 = numpy.linalg.norm(A0)
            acc_cos = numpy.sum(An*A0) / max(norm_An*norm_A0, 1e-30)
            acc_amp_ratio = norm_An / max(norm_A0, 1e-30)
            R_m = float(domain.cylinder_radius / 'm')
            max_V_inc = numpy.linalg.norm(Vn, axis=1).max() * R_m
            max_A_inc = 0.5 * numpy.linalg.norm(An, axis=1).max() * R_m

            log.info(f'[MESH PREDICTOR PARTS] Jcur={J_cur.min():.6e}, Jvel={J_vel.min():.6e}, Jacc={J_acc.min():.6e}, Jfull={J_full.min():.6e}, Jlinear={J_lin.min():.6e}')
            log.info(f'[MESH PREDICTOR HISTORY] cos(A_n,A_nm1)={acc_cos:.6e}, ||A_n||/||A_nm1||={acc_amp_ratio:.6e}, max|dt*v_m|={max_V_inc:.6e} m, max|0.5*dt^2*a_m|={max_A_inc:.6e} m')

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
                
            if has_remeshed and istep % 20 == 0:

                R_m = float(domain.cylinder_radius / 'm')
                dt_ = float(dynamic.timestep / 's')

                e_d0 = numpy.linalg.norm(args['dm0'] - dm_n, axis=1).max() * R_m
                e_v0 = numpy.linalg.norm(args['vm0δt'] - Vn, axis=1).max() * R_m / dt_
                e_a0 = numpy.linalg.norm(args['am0δt2'] - An, axis=1).max() * R_m / dt_**2

                log.info(
                    f'[NEWMARK SHIFT CHECK] '
                    f'dm0-dm_n={e_d0:.6e} m, '
                    f'vm0-vm_n={e_v0:.6e} m/s, '
                    f'am0-am_n={e_a0:.6e} m/s2'
                )

        # ------------------------------------------------------------
        # Predictor-state diagnostic before Newton
        # ------------------------------------------------------------
        if has_remeshed:

            Jf_pred = function.eval(Jfluid_bz, arguments=args)
            Js_pred = function.eval(Jsolid_bz, arguments=args)

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

            log.info(
                f'[PREDICTOR] '
                f'J fluid min/max: {Jf_pred.min():.6e}, {Jf_pred.max():.6e} | '
                f'J solid min/max: {Js_pred.min():.6e}, {Js_pred.max():.6e}')

        # ------------------------------------------------------------
        # Solve nonlinear system
        # ------------------------------------------------------------

        check_newton_jump = has_remeshed and istep <= fluid_state['remesh_istep'] + 2

        if check_newton_jump:
            before_newton = {
                name: args[name].copy()
                for name in ('dm', 'd', 'cm', 'lam')}

        # Residual of each equation at the predictor, before Newton.
        if has_remeshed and istep == fluid_state['remesh_istep'] + 1:
            for trial_name, test_name in (
                ('d', 'dtest'),       # solid momentum
                ('cm', 'cmtest'),     # solid centre
                ('dm', 'dmtest'),     # mesh equilibrium
                ('u', 'utest'),       # fluid momentum
                ('p', 'ptest'),       # fluid incompressibility
                ('lam', 'lamtest'),   # interface displacement
            ):
                values = numpy.asarray(
                    function.eval(
                        res.derivative(test_name),
                        arguments=args,
                    ),
                    dtype=float,
                ).reshape(-1)

                # Dirichlet equations are not solved by Newton.
                if trial_name in cons:
                    free = ~numpy.isfinite(cons[trial_name]).reshape(-1)
                    values = values[free]

                log.info(
                    f'[PRE-NEWTON RESIDUAL] {trial_name}: '
                    f'L2={numpy.linalg.norm(values):.6e}, '
                    f'max={numpy.max(numpy.abs(values)):.6e}'
                )
        try:
            args = system.solve(constrain=cons, arguments=args, tol=1e-9)

            if check_newton_jump:
                R_m = float(domain.cylinder_radius / 'm')
                dt_s = float(dynamic.timestep / 's')

                for name in ('dm', 'd', 'cm', 'lam'):
                    change = args[name] - before_newton[name]
                    magnitude = numpy.linalg.norm(change.reshape(-1, change.shape[-1]), axis=1).max() if name in ('dm', 'd', 'lam') else numpy.abs(change).max()

                    if name in ('dm', 'd'):
                        displacement_m = magnitude * R_m
                        implied_acceleration = (
                            displacement_m / (dynamic.beta * dt_s**2)
                        )
                        log.info(
                            f'[NEWTON CHANGE] {name}: '
                            f'max={displacement_m:.6e} m, '
                            f'implied acceleration={implied_acceleration:.6e} m/s2')
                    else:
                        log.info(
                            f'[NEWTON CHANGE] {name}: '
                            f'max coefficient change={magnitude:.6e}')            

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

        if fluid:
            Jf_check = function.eval(Jfluid_bz, arguments=args)

            if not numpy.all(numpy.isfinite(Jf_check)) or Jf_check.min() <= 0:
                raise RuntimeError(f'Invalid fluid state after Newton: Jf_min={Jf_check.min():.6e}')

        if solid:
            Js_check = function.eval(Jsolid_bz, arguments=args)

            if not numpy.all(numpy.isfinite(Js_check)) or Js_check.min() <= 0:
                raise RuntimeError(f'Invalid solid state after Newton: Js_min={Js_check.min():.6e}')

        # Put the new MESH STEP CONSISTENCY diagnostic here.
        if has_remeshed and 'dm' in args:
            
            Δdm_solve = args['dm'] - dm_before_solve
            Δdm_vel   = Vn_before_solve

            err_vel  = Δdm_solve - Δdm_vel

            R_m = float(domain.cylinder_radius / 'm')

            max_solve = numpy.linalg.norm(Δdm_solve, axis=1).max() * R_m
            max_vel   = numpy.linalg.norm(Δdm_vel, axis=1).max() * R_m
            max_acc   = 0.5 * numpy.linalg.norm(An_before_solve, axis=1).max() * R_m
            max_err_vel  = numpy.linalg.norm(err_vel, axis=1).max() * R_m

            rel_err_vel = max_err_vel / max(max_solve, 1e-30)

            log.info(f'[MESH STEP CONSISTENCY] max|dm_np1-dm_n|={max_solve:.6e} m')
            log.info(f'[MESH STEP CONSISTENCY] max|dt*vm_n|={max_vel:.6e} m, max|0.5*dt^2*am_n|={max_acc:.6e} m')
            log.info(f'[MESH STEP CONSISTENCY] ||Δdm-dt*vm_n|| max={max_err_vel:.6e} m, rel={rel_err_vel:.6e}')

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
            Jf = function.eval(Jfluid_bz, arguments=args)
            log.info(f'J fluid min/max: {Jf.min():.6e}, {Jf.max():.6e}')

        if solid:
            Js = function.eval(Jsolid_bz, arguments=args)
            dcyl = function.eval(d_cyl_bz, arguments=args)
            dcyl_mag = numpy.linalg.norm(dcyl / 'm', axis=1)

            log.info(f'J solid min/max: {Js.min():.6e}, {Js.max():.6e}')
            log.info(f'particle/interface displacement max: {dcyl_mag.max():.6e} m')

        if has_remeshed:
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

            # --------------------------------------------------------------
            # ADD ACCELERATION SEED DIAGNOSTIC HERE
            # --------------------------------------------------------------

            age = istep - fluid_state['remesh_istep']

            if 1 <= age <= 10:

                sample = fluid_state['transfer_sample']

                am_now = numpy.asarray(
                    function.eval(
                        sample.bind(fluid_state['a_field']),
                        arguments=args
                    ) / 'm/s2',
                    dtype=float,
                )

                am0  = fluid_state['am_restart_ms2']
                seed = fluid_state['am_transfer_error_ms2']

                dam = am_now - am0

                seed_norm = numpy.linalg.norm(seed)
                dam_norm  = numpy.linalg.norm(dam)

                cos_seed = numpy.sum(seed * dam) / max(
                    seed_norm * dam_norm, 1e-30)

                amplification = dam_norm / max(seed_norm, 1e-30)

                seedmag = numpy.linalg.norm(seed, axis=1)
                dammag  = numpy.linalg.norm(dam, axis=1)

                i_seed = numpy.argmax(seedmag)
                i_dam  = numpy.argmax(dammag)

                x0 = fluid_state['transfer_x0_m']

                dx_peak = numpy.linalg.norm(
                    x0[i_seed] - x0[i_dam])

                dam_prev = fluid_state['dam_previous']

                if dam_prev is None:
                    cos_prev = numpy.nan
                else:
                    cos_prev = numpy.sum(dam * dam_prev) / max(
                        numpy.linalg.norm(dam)
                        * numpy.linalg.norm(dam_prev),
                        1e-30)

                fluid_state['dam_previous'] = dam.copy()

                log.info(
                    f'[AM SEED TEST] post-remesh step={age}, '
                    f'||Δam||/||seed||={amplification:.6e}, '
                    f'cos(seed,Δam)={cos_seed:.6e}, '
                    f'cos(Δam_n,Δam_nm1)={cos_prev:.6e}, '
                    f'peak-distance={dx_peak:.6e} m'
                )

            

            if istep % 20 == 0:

                x_am = numpy.asarray(function.eval(x_bz, arguments=args) / 'm')

                am_g, arel_g, DuDt_g = function.eval([fluid_bezier.bind(fluid_state['a_field']),
                        fluid_bezier.bind(fluid_state['arel_field']),
                        fluid_bezier.bind(fluid_state['DuDt_field'])], arguments=args)

                am_g   = numpy.asarray(am_g / 'm/s2')
                arel_g = numpy.asarray(arel_g / 'm/s2')
                DuDt_g = numpy.asarray(DuDt_g / 'm/s2')

                amag = numpy.linalg.norm(am_g, axis=-1)
                rmag = numpy.linalg.norm(arel_g, axis=-1)

                at_g = am_g + arel_g
                tmag = numpy.linalg.norm(at_g, axis=-1)
                dmag = numpy.linalg.norm(DuDt_g, axis=-1)

                ia = numpy.argmax(amag)

                cos_ar = numpy.dot(am_g[ia], arel_g[ia]) / max(amag[ia] * rmag[ia], 1e-30)

                log.info(
                    f'[MESH ACC LOCATION] max|am|={amag[ia]:.6e} m/s2 at '
                    f'x=({x_am[ia,0]:.6e}, {x_am[ia,1]:.6e}) m')

                log.info(
                    f'[FLUID ACC DECOMP] at max|am|: '
                    f'|am|={amag[ia]:.6e}, '
                    f'|arel|={rmag[ia]:.6e}, '
                    f'|am+arel|={tmag[ia]:.6e}, '
                    f'|DuDt|={dmag[ia]:.6e} m/s2, '
                    f'cos(am,arel)={cos_ar:.6e}')

                # --------------------------------------------------------------
                # Direct Newmark reconstruction check
                # --------------------------------------------------------------

                beta  = dynamic.beta
                gamma = dynamic.gamma

                # Direct Newmark reconstruction check
                δA_chk = (args['dm'] - args['dm0'] - args['vm0δt'] - .5*args['am0δt2']) / dynamic.beta
                V_chk = args['vm0δt'] + args['am0δt2'] + dynamic.gamma*δA_chk
                A_chk = args['am0δt2'] + δA_chk
                Arel_chk = args['a0δt'] + (args['u'] - args['u0'] - args['a0δt']) / dynamic.gamma

                dm_template = fluid_state['d_field']
                urel_template = fluid_state['urel_field']

                vm_chk_field = function.replace_arguments(dm_template, [('dm', 'vmcheck')]) / dynamic.timestep
                am_chk_field = function.replace_arguments(dm_template, [('dm', 'amcheck')]) / dynamic.timestep**2
                arel_chk_field = function.replace_arguments(urel_template, [('u', 'arelcheck')]) / dynamic.timestep

                check_args = dict(args, vmcheck=V_chk, amcheck=A_chk, arelcheck=Arel_chk)

                vm_sym, am_sym, arel_sym, vm_chk, am_chk, arel_chk = function.eval([
                    fluid_bezier.bind(fluid_state['v_field']), fluid_bezier.bind(fluid_state['a_field']),
                    fluid_bezier.bind(fluid_state['arel_field']), fluid_bezier.bind(vm_chk_field),
                    fluid_bezier.bind(am_chk_field), fluid_bezier.bind(arel_chk_field)], arguments=check_args)

                vm_sym, vm_chk = numpy.asarray(vm_sym/'m/s'), numpy.asarray(vm_chk/'m/s')
                am_sym, am_chk = numpy.asarray(am_sym/'m/s2'), numpy.asarray(am_chk/'m/s2')
                arel_sym, arel_chk = numpy.asarray(arel_sym/'m/s2'), numpy.asarray(arel_chk/'m/s2')

                ev = numpy.linalg.norm(vm_sym - vm_chk, axis=-1)
                ea = numpy.linalg.norm(am_sym - am_chk, axis=-1)
                er = numpy.linalg.norm(arel_sym - arel_chk, axis=-1)

                vref = max(numpy.linalg.norm(vm_sym, axis=-1).max(), 1e-30)
                aref = max(numpy.linalg.norm(am_sym, axis=-1).max(), 1e-30)
                rref = max(numpy.linalg.norm(arel_sym, axis=-1).max(), 1e-30)

                log.info(f'[NEWMARK RECON CHECK] vm err={ev.max():.6e} m/s, rel={ev.max()/vref:.6e} | am err={ea.max():.6e} m/s2, rel={ea.max()/aref:.6e} | arel err={er.max():.6e} m/s2, rel={er.max()/rref:.6e}')
        # ------------------------------------------------------------
        # Remeshing trigger — checked after every successful solve
        # ------------------------------------------------------------

        if fluid and solid and enable_remeshing:
            need_remesh = False

            # Quality-based trigger: minimum Jacobian below threshold
            if Jf.min() < remesh_J_threshold:
                log.info(
                    f'Remesh triggered by mesh quality: '
                    f'Jf_min={Jf.min():.4f} < {remesh_J_threshold}')
                
                need_remesh = True

            # Time-interval-based trigger (optional fallback)
            if remesh_interval_s and current_t_s >= next_remesh_time_s - 0.5 * dt_s:
                log.info(
                    f'Remesh triggered by time interval at t={current_t_s:.6f} s')
                
                need_remesh = True

            if need_remesh:
                remesh_count += 1

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
                fluid_state['dam_previous'] = None

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

        with particle_csv.open('a') as f:
            f.write(f'{current_t_s:.12e},{istep},{L_particle_m:.12e},{B_particle_m:.12e},{deformation_index:.12e}\n')
        x, xb = function.eval([x_bz, x_bbz], arguments=args)
        # ------------------------------------------------------------
        # Save deformed mesh every 100 timesteps
        # ------------------------------------------------------------
        if istep % 100 == 0:

            if solid:
                xs = function.eval(xsolid_bz, arguments=args)

            with export.mplfigure(str('mesh.jpg'), dpi=200) as fig:
                ax = fig.add_subplot(111, title=f'Deformed mesh at t={t:.3s}', ylabel='[m]')

                # Fluid mesh
                if fluid:
                    export.triplot(ax, x / 'm', hull=fluid_bezier.hull, linewidth=0.25)

                # Solid/particle mesh
                if solid:
                    export.triplot(ax, xs / 'm', hull=solid_bezier.hull,linewidth=0.25)

                # Particle boundary
                export.triplot(ax, xb / 'm', hull=bbezier.tri, linewidth=1.0, linecolor='b')

                ax.set_aspect('equal')

                L = domain.channel_length / 'm'
                ax.set_xlim(-L, L)
                ax.set_ylim(-L, L)

        # ------------------------------------------------------------
        # Save velocity plot every 100 timesteps
        # ------------------------------------------------------------
        if fluid and istep % 100 == 0:

            u = function.eval(u_bz, arguments=args)
            velmag = numpy.linalg.norm(u / 'm/s', axis=1)

            with export.mplfigure(str('solution.jpg'), dpi=150) as fig:
                ax = fig.add_subplot(111, ylabel='[m]', title=f'Velocity magnitude at t={t:.3s}')
                im = ax.tripcolor(*(x / 'm').T, fluid_bezier.tri, velmag, shading='gouraud')

                fig.colorbar(im, orientation='horizontal', label='velocity [m/s]')

                export.triplot(ax, xb / 'm', hull=bbezier.tri, linewidth=1)

                ax.set_aspect('equal')

                L = domain.channel_length / 'm'
                ax.set_xlim(-L, L)
                ax.set_ylim(-L, L)


if __name__ == '__main__':
          cli.choose(main)


