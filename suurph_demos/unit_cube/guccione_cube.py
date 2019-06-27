import matplotlib.pyplot as plt
import numpy as np
from fenics import *

from guccionematerial import GuccioneMaterial


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4


# Setup the mesh and the function space for the solutions
mesh = UnitCubeMesh(4,4,4)
V = VectorFunctionSpace(mesh, "Lagrange", 2)


# Define functions
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration



# Mark boundary subdomains
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
right.mark(boundary_markers, 2)

# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

# Define Dirichlet boundary (x = 0 or x = 1)
clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, left)
bcs = [bc]

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
#C = F.T*F                   # the right Cauchy-Green tensor
#E = 0.5*(C - I)             # the Green-Lagrange strain tensor

F = variable(F)

# Tissue microstructure
f0 = as_vector([ 1.0, 0.0, 0.0 ])
s0 = as_vector([ 0.0, 1.0, 0.0 ])
n0 = as_vector([ 0.0, 0.0, 1.0 ])
        
"""
Define the material model, either by defining the strain energy directly,
or using the guccione_material class (recommended). You can use the default 
material parameters. 
"""      

mat = GuccioneMaterial(e1=f0,e2=s0,e3=n0,kappa=1e3)
psi = mat.strain_energy(F)

P = diff(psi,F) # first Piola-Kirchhoff stress tensor



"""
# Material parameters (Lame parameters)
mu    = 4.0  
lmbda = 20.0  

# The strain energy for the neo-Hookean model:
E = variable(E)
psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)

S = diff(psi,E) #second Piola-Kirchhoff stress
P = F*S         #First Piola-Kirchhoff stress
"""
p_right = Constant(0.0) #the pressure load (zero for now)

# Definition of the weak form:
N = FacetNormal(mesh)
Gext = p_right * inner(v, cofac(F)*N) * ds(2) #ds(2) = left boundary
R = inner(P,grad(v))*dx + Gext



# The middle point on the right boundary
point0 = np.array([1.0,0.5,0.5])


# Step-wise loading (for plotting and convergence)
load_steps = 6
target_load = 10.0
loads = np.linspace(0,target_load,load_steps)

d0 = np.zeros(3)                #displacement at point0
disp = np.zeros(load_steps) #array to store displacement for all steps

for step in range(load_steps):
    # Stretch is a negative pressure
    p_right.assign(-loads[step])
    
    #solve the problem:
    solve(R == 0, u, bcs)
    
    #evaluate displacement at point defined above
    u.eval(d0,point0)
    disp[step] = d0[0]

    
#displacement on x-axis, load on y-axis
plt.figure(1)
plt.plot(disp,loads)
plt.xlabel('Displacement of point (1.0,0.5,0.5)')
plt.ylabel('Applied pressure load')

# Plot the deformed cube and the original
plt.figure(2)
u_int = dolfin.interpolate(u, dolfin.VectorFunctionSpace(mesh, "CG", 1))
deformed_mesh = dolfin.Mesh(mesh)
dolfin.ALE.move(deformed_mesh, u_int)
dolfin.plot(mesh, alpha=0.5, edgecolor='k')#, title="original")
dolfin.plot(deformed_mesh, edgecolor='g', alpha=0.7)#, title='Contracting cube')
ax = plt.gca()
ax.view_init(elev=2, azim=-92)

plt.show()
