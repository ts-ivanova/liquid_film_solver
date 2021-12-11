# liquid_film_solver
A solver for an integral boundary layer model 
for liquid films on moving substrates 
developed for a project during RM2021:
"Physics-based and Data-driven Modelling of Nonlinear Waves in Liquid Films" 
by T. Ivanova, supervised by Prof. M. Mendez and by F. Pino

This project solves a simplified version of the 3D integral model
for a liquid film on a moving substrate.
The unknowns of this integral model are
the liquid film height h,
the flow rate along the streamwise direction qx,
and the flow rate along the spanwise direction qz
(check RM2021 report at VKI).

The ```main_solver_liquid_film.py``` script computes the solution of this integral model. The numerical schemes, fluxes and sources computations are all separated in different scripts for easier navigation. Tools for saving and post-processing are also developed during this project.


The 3D integral model is dimensionless.
It is derived during RM2021 using a common approach
in boundary layer theory.
The starting point is the Navier-Stokes equations
for a divergence-free Newtonian fluid.
Re-formulating the equations in their dimensionless version
using proper reference quantities
leads to the possibility to neglect terms of higher order
with respect to a liquid film parameter
defined as the ratio between the wall-normal scale
and the streamwise scale.
What is obtained is a dimensionless system
in a first-order long-wave approximation.
This approach allows to preserve the nonlinearities of the system.
The final step of the derivation of the 3D integral model is
the integration along the wall-normal direction.
This yields a system with three unknowns which depends only
on two spatial coordinates.
Therefore, a three-dimensional problem is converted
to a two-dimensional configuration.
This is an important advantage
because it reduces significantly the computational
costs of the liquid film simulations.


This system accounts for all physical forces and the equations are:

<img src="images_for_readme/system.png" width="600">

Flux matrix elements:

<img src="images_for_readme/fluxes.png" width="600">

To discretise this system, the 2D formulation of the finite volume method is applied:

<img src="images_for_readme/fv.png" width="500">

The numerical fluxes are approximated by a blended scheme between high order and low order schemes:

<img src="images_for_readme/flux_approx_bl.png" width="500">


What is currently not completely implemented
in this version of the solver are
pressure gradients (because they affect a narrow region of the domain),
interface shear stresses,
and the most challenging - surface tension terms
(the third derivatives of the height h).

This is the working simplified version of the 3D model that the solver currently can investigate:

<img src="images_for_readme/implemented.png" width="600">

Without surface tension, pressure gradient and interface shear stress, the system reads:

<img src="images_for_readme/solved_system_simplified.png" width="600">
