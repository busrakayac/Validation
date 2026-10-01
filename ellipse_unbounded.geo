// ============================================================
// Gao et al. (2013) validation
// 2D circular neo-Hookean particle in planar extensional flow
//
// Fluid domain:
//      x,y in [-L,L]
//
// Particle:
//      centered at (0,0)
//      radius = cylinder_radius
//
// Outer velocity BC will be imposed in Nutils:
//      ux = epsdot*x
//      uy = -epsdot*y
// ============================================================

SetFactory("OpenCASCADE");

Mesh.MshFileVersion = 2.2;


// ============================================================
// 1. INPUT PARAMETERS
// ============================================================

// These values are overwritten by Nutils through gmsh(... numbers={...})
//
// Recommended first validation:
// cylinder_radius = 30e-6 m
// channel_length  = 240e-6 m
// elemsize        = 2e-6 m

If (!Exists(channel_length))
    channel_length = 240e-6;
EndIf

If (!Exists(cylinder_radius))
    cylinder_radius = 30e-6;
EndIf

If (!Exists(elemsize))
    elemsize = 2e-6;
EndIf

If (!Exists(coarsening))
    coarsening = 4;
EndIf

L = channel_length;
R = cylinder_radius;

Lambda = 0.8;
theta0 = Pi/4;

a0 = R/Sqrt(Lambda);
b0 = R*Sqrt(Lambda);

ct = Cos(theta0);
st = Sin(theta0);


// ============================================================
// 2. OUTER SQUARE
// ============================================================
//
//             top
//        4 ----------- 3
//        |             |
//   left |             | right
//        |             |
//        1 ----------- 2
//            bottom
//

Point(1) = {-L, -L, 0};
Point(2) = { L, -L, 0};
Point(3) = { L,  L, 0};
Point(4) = {-L,  L, 0};

Line(1) = {1, 2};   // bottom
Line(2) = {2, 3};   // right
Line(3) = {3, 4};   // top
Line(4) = {4, 1};   // left

Curve Loop(100) = {1, 2, 3, 4};


// ============================================================
// 3. ELLIPTICAL PARTICLE
// ============================================================
//
// Initial aspect ratio:
//      Lambda = b0/a0 = 0.8
//
// Initial orientation:
//      theta0 = Pi/4
//
// Particle area is kept equal to that of the original
// circle of radius R:
//      a0*b0 = R^2
// ============================================================

// Center
Point(10) = {0, 0, 0};

// Principal-axis points of rotated ellipse
//
// Major-axis unit vector:
//      e_a = (cos(theta0), sin(theta0))
//
// Minor-axis unit vector:
//      e_b = (-sin(theta0), cos(theta0))

// + major axis
Point(11) = {
    a0*ct,
    a0*st,
    0
};

// + minor axis
Point(12) = {
    -b0*st,
     b0*ct,
    0
};

// - major axis
Point(13) = {
    -a0*ct,
    -a0*st,
    0
};

// - minor axis
Point(14) = {
     b0*st,
    -b0*ct,
    0
};


// Elliptical arcs
//
// Syntax:
// Ellipse(tag) = {start, center, major-axis point, end};

Ellipse(11) = {11, 10, 11, 12};
Ellipse(12) = {12, 10, 11, 13};
Ellipse(13) = {13, 10, 11, 14};
Ellipse(14) = {14, 10, 11, 11};

Curve Loop(200) = {11, 12, 13, 14};


// ============================================================
// 4. SOLID AND FLUID SURFACES
// ============================================================

// Solid particle
Plane Surface(300) = {200};

// Fluid = square minus particle
Plane Surface(400) = {100, 200};


// ============================================================
// 5. PHYSICAL GROUPS
// ============================================================

// Domains used in Python:
// topo['solid']
// topo['fluid']

Physical Surface("solid") = {300};
Physical Surface("fluid") = {400};


// Particle FSI interface
//
// This gives:
// topo['fluid'].boundary['cylinder']
// topo['solid'].boundary['cylinder']

Physical Curve("cylinder") = {11, 12, 13, 14};


// Individual outer boundaries.
// These are useful for debugging and plotting.

Physical Curve("bottom") = {1};
Physical Curve("right")  = {2};
Physical Curve("top")    = {3};
Physical Curve("left")   = {4};


// All external boundaries together.
// This is the one we will use for the Gao velocity BC:
//
// topo['fluid'].boundary['outer']

Physical Curve("outer") = {1, 2, 3, 4};


// ============================================================
// 6. MESH
// ============================================================

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;


// ------------------------------------------------------------
// Fine mesh around particle
// ------------------------------------------------------------

lc_particle = elemsize;
lc_far = coarsening * elemsize;

// Distance from particle interface
Field[1] = Distance;
Field[1].CurvesList = {11, 12, 13, 14};

// Gradually increase mesh size away from particle
Field[2] = Threshold;
Field[2].InField = 1;

Field[2].SizeMin = lc_particle;
Field[2].SizeMax = lc_far;

// Keep finest resolution near particle
Field[2].DistMin = 0;

// Reach far-field resolution at approximately four radii
Field[2].DistMax = 3.0*a0;

Background Field = 2;


// ============================================================
// 7. MESH QUALITY SETTINGS
// ============================================================

Mesh.Algorithm = 6;
Mesh.Smoothing = 30;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
