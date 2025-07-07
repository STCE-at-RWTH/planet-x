### A Pluto.jl notebook ###
# v0.20.8

#> [frontmatter]
#> title = "Applying Bressan's Method to 2-D Shockwaves"
#> date = "2024-04-16"
#> 
#>     [[frontmatter.author]]
#>     name = "Alexander Fleming"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e93237ca-27d8-415f-a4dd-99eb4f530f69
begin
	using DifferentiationInterface
	import ForwardDiff
	import Mooncake
	using LaTeXStrings
	using LinearAlgebra
	using Plots
	using PlutoUI
	using ShockwaveProperties
	using ShockwaveProperties.BilligShockParametrization
	using Tullio
	using Unitful
end;

# ╔═╡ 2af7b8a2-b5d7-4957-84c2-815d52b0ebd3
PlutoUI.TableOfContents()

# ╔═╡ 00f60ab4-aa32-4f0d-bf83-c175de05066e
let Minf=4.0
	α = 0.:0.01:4
	sf = x -> shock_front(x, Minf, 1.0)
	Γ = reduce(hcat, sf.(α))
	p = plot(Γ[1,:], Γ[2,:], label=L"\Gamma(M_\infty, \bar y)", xlims=(-2.5, 1.0), ylims=(0, 3.1), aspect_ratio=:equal, legend=:topright, dpi=600)
	title!(p, "Shock Front")
	xlabel!(p, L"\bar{x}")
	ylabel!(p, L"\bar{y}")
	plot!(p, -1:0.001:1, x->sqrt(1-x^2), type=:shape, fill=true, color=:black, label=false)
	for a = 0.:0.2:4
		n = shock_normal(a, Minf, 1.)
		pt = shock_front(a, Minf, 1.)
		plot!(p, [pt[1], pt[1]+0.25*n[1]], [pt[2], pt[2]+0.25*n[2]], label=(a == 0. ? L"\mathbf{\hat{n}}(M_\infty, \bar y)" : false), color=:green, arrow=true)
	end
	for a = 0.:0.25:4
		plot!(p, [-2.5, -2.0], [a, a], color=:blue, arrow=true, label=(a == 0. ? L"v_\infty^0" : false))
	end
	scatter!(p, [sf(1.75)[1]], [1.75], label=false)
	annotate!(p, sf(1.75)[1]+0.6, 1.7, Plots.text(L"F_n(u_L) = F_n(u_R)", 10))
	#savefig(p, "../../fluid-dynamics-conference-2024/abstract-submission/problem_setting.pdf")
	p
end

# ╔═╡ ca06cf02-5844-4487-ab22-5e6bf7547378
md"""
# Problem Setting

We are interested in the sensitivity of the shock wave position and the conditions behind the shock wave to changes in the free-stream Mach number $M_\infty$. We will use generalized tangent vectors (defined by Bressan) to do this.

## Assumptions

- Steady-state supersonic ($M_\infty > 1$) flow *to the right* impacts a cylinder in the plane. (Note: the free stream velocity is always represented as `[M, 0]`) )
- Shock wave will develop and move away from the cylinder.
- We will work with flow density ``\rho``, flow momentum density ``\rho\vec{v}``, and flow total energy density ``\rho E = \rho\left(e+\frac{\vec{v}\cdot\vec{v}}{2}\right)``
- We will neglect heat generation/addition ``\dot q`` and body forces ``\vec f``
``\renewcommand{\divv}{\mathrm{div}\,}``

## Notation
- The difference in a quantity over a shock wave is taken left-to-right: ``[u] = u_L - u_R``.
- Products between scalars and non-scalars are simply ``(ab)_i = ab_i``.
- ``a^\intercal`` swaps the first and last dimensions of ``a``.
- Products between larger objects are understood to be contractions along the last dimension of ``a`` and the first dimension of ``b``. 
```math
	(ab)_{ijk} = a_{ij\ell}b_{\ell k}
```
- ``\nabla_\Gamma F`` is the jacobian matrix of ``F`` w.r.t. each component of ``\Gamma``, i.e. ``(\nabla_\Gamma F)_{ijk}=\partial_{\Gamma_k}F_{ij}``  
- ``\divv A = \partial_{x_k}A_{ik}``. (Rows of ``\divv A`` are ``\divv A_{i\star}``)
- ``(a \otimes b)_{ij} = a_ib_j``
- Unit vectors are boldface and have little hats.
"""

# ╔═╡ 6d4f0e92-2fd8-4d91-bb50-8a5b18dde745
md"""
In this scenario, the Euler equations for inviscid and compressible flow are:
```math
\begin{align}
	0 &=\partial_t\rho + \divv \rho\vec v\\
	0 &=\partial_t(\rho\vec v) + \divv\left(\rho\vec{v}\otimes\vec{v}+P \mathcal{I}\right)\\
	0 &=\partial_t(\rho E) + \divv\left(\rho E + P\right)\vec v
\end{align}
```
Set ``\vec{u}`` to ``(\rho, \rho v_1, \rho v_2, \rho E)`` to rewrite the system as
```math
	\partial_t\vec{u} + \divv F(\vec{u}) = 0
```
In this case, ``\vec u \in \mathbb R^4`` and ``F\colon \mathbb R^4 \rightarrow\mathbb R^{4 \times 2}``. 
"""

# ╔═╡ 6e91e9ed-473d-4496-b589-85ac5c954877
md"""
# Rankine-Hugoniot Conditions
If we have ``\mathbf{\hat n}`` normal to the shock wave, then
```math
	s\left[\vec u\right] = \left[F(\vec u)\,\mathbf{\hat n}\right]
```
In the stationary case, the shock speed ``s`` is zero. Total mass, momentum, and energy must be conserved across the shock wave. Define ``F_n`` as
```math
F_n(\vec u) = F(\vec u)\,\mathbf{\hat n}
```
"""

# ╔═╡ a1789327-d3a8-4c6c-8fc0-64e8d81cc98d
function F(u::ConservedProps; gas::CaloricallyPerfectGas=DRY_AIR)
	v = u.ρv / u.ρ # velocity
	P = pressure(u; gas=gas) 
	return vcat(u.ρv', (u.ρv .* v' + I*P), (v .* (u.ρE + P))') # stack row vectors
end

# ╔═╡ 9f358218-6eea-4d50-8374-f1c7c3b65e2d
function F(u; gas::CaloricallyPerfectGas=DRY_AIR)
	ρv = @view u[2:end-1]
	v = ρv/u[1]
	P = ustrip(pressure(internal_energy_density(u[1], ρv, u[end]); gas=gas))
	return vcat(ρv', ρv .* v' + I*P, (v .* (u[end] + P))')
end

# ╔═╡ 417f01cc-dae8-4876-96a9-13f2b6b4ed77
function ∂F∂u(u; gas::CaloricallyPerfectGas = DRY_AIR)
	n = length(u)
	_, F_back = pullback(u->F(u; gas=gas), u)
	seeds = vcat([hcat(1.0*I[1:n, k], zeros(n)) for k∈1:n],
				 [hcat(zeros(n), 1.0*I[1:n, k]) for k∈1:n])
	∂F = map(seeds) do F̄
		F_back(F̄)[1]
	end
	out = stack([reduce(hcat, ∂F[1:n])', reduce(hcat, ∂F[(n+1):end])'])
	return permutedims(out, (1, 3, 2))
end

# ╔═╡ 246c97ad-1087-484a-90b0-bee0c09c61ae
F_n(u, n̂; gas::CaloricallyPerfectGas=DRY_AIR) = F(u) * n̂

# ╔═╡ 78adce9e-5253-45d1-9e28-87012a74262e
md"""
##  Solving Post-Shock Conditions

Set ``v_n = \vec{v}\cdot\mathbf{\hat{n}}`` and ``v_t=\vec{v}-v_n\mathbf{\hat{n}}``. From either the Rankine-Hugoniot conditions or control-volume analysis of an oblique shock wave passing through an appropriately drawn control volume, we can derive:
```math
\begin{align}
	0 &= \rho_L v_{n,L} - \rho_R v_{n,R}\\
	0 &= v_{t, L}-v_{t,R}\\
	0 &= \rho_L \vec v_L\cdot\vec u_L + P_L -\rho_R \vec v_R\cdot\vec v_R - P_R\\
	0 &= \rho_L v_{n,L}E_L + P_Lv_{n,L} - \rho_Rv_{n,R}E_R - P_Rv_{n,R} 
\end{align}
```

The equations of state for an ideal gas (``PV = RT`` and ``de = c_v(T)dT`` or ``dh= c_p(T)dT``) add the final two equations to fully determine the above system.
"""

# ╔═╡ c97a2cdd-7ff7-4e86-a026-1fb4f9287628
md"""
## Closing the System
"""

# ╔═╡ 0ed965e6-7662-417b-b750-eccdc4ba2184
md"""
From the defintions of ``E = e + \frac{\vec{v}\cdot\vec{v}}{2}``, ``e = c_VT = \frac{R}{\gamma-1}T``, and the ideal gas equation ``PV= MRT \implies P = \rho RT``, we have
```math
\begin{align}
P &= \rho (\gamma-1) e\\
P &= \rho (\gamma-1)\left(E - \frac{\vec v \cdot \vec v}{2}\right)\\
P &= (\gamma-1)\left(\rho E - \frac{\rho \vec v \cdot \rho\vec v}{2\rho}\right)\\
P &= (\gamma-1)\left(u_4 - \frac{\vec u_{2:3}\cdot\vec u_{2:3}}{2u_1}\right)
\end{align}
```
"""

# ╔═╡ a505438f-a1c8-451b-a04f-c7efdd5a94a6
md"""
## Computing Initial Conditions
"""

# ╔═╡ 39f6d764-f16b-4dd5-9c26-f41688915ca5
function u0_jump(s_L, x, y; gas::CaloricallyPerfectGas = DRY_AIR)
	uL = conserved_state_vector(s_L; gas=gas)
	n = shock_normal(y, s_L[2], 1.0)
	t = shock_tangent(y, s_L[2], 1.0)
	uR = conserved_state_behind(uL, n, t; gas=gas)
	return (uL, uR)
end

# ╔═╡ 8f28cd61-82a7-41af-b3d0-1e48e29d1430
function u0(s_L, x, y; gas::CaloricallyPerfectGas = DRY_AIR)
	shock_x, shock_y = shock_front(y, s_L[2], 1.0)
	uL, uR = u0_jump(s_L, x, y; gas=gas)
	if x<=shock_x
		return uL
	end
	return uR
end

# ╔═╡ f727b337-bfa0-4d45-b593-6ad5607930ae
let M1=4.0, M2 = M1 + 0.5
	α = 0.:0.01:4
	sf = (x, M, ) -> shock_front(x, M, 1.)
	Γ1 = reduce(hcat, Base.Fix2(sf, M1).(α))
	Γ2 = reduce(hcat, Base.Fix2(sf, M2).(α))
	p = plot(Γ1[1,:], Γ1[2,:], label=L"\Gamma(M_\infty, \bar y)", 
		xlims=(-1, 0), ylims=(2,3), 
		aspect_ratio=:equal, legend=:outertopright, dpi=600)
	plot!(p, Γ2[1,:], Γ2[2,:], label=L"\Gamma(M_\infty+\varepsilon, \bar y)")
	title!(p, "Perturbation of Shock Wave by "*L"\varepsilon")
	xlabel!(p, L"\bar{x}")
	ylabel!(p, L"\bar{y}")
	arrow_size=0.03
	Plots.GR.setarrowsize(0.5)
	for a = 2.:0.05:3
		n = shock_normal(a, M1, 1.)
		pt = shock_front(a, M1, 1.)
		arrows = hcat(pt, pt + arrow_size*n)
		plot!(p, arrows[1,:], arrows[2,:], label=false, color=:green, arrow=true)
		n = shock_normal(a, M2, 1.)
		pt = shock_front(a, M2, 1.)
		arrows = hcat(pt, pt + arrow_size*n)
		plot!(p, arrows[1,:], arrows[2,:], label=false, color=:green, arrow=true)
	end
	savefig(p, "gamer_zone.pdf")
	p
end

# ╔═╡ 8a4e3e3e-9074-49bf-840d-3888579d5ba2
md"""
# A First-Order Approximation

In their paper, Bressan et. al. create a class of objects called generalized tangent vectors. A single element in the space of generalized tangent vectors ``T_u``, corresponds to the perturbation 

```math
	u^\varepsilon = u + \varepsilon v + \sum_{\xi_k < 0}\left(u_R(x_k) - u_L(x_k)\right)\mathcal X_{[x_k+\varepsilon\xi_k, x_k]} - \sum_{\xi_k > 0}\left(u_R(x_k) - u_L(x_k)\right)\mathcal X_{[x_k, x_k+\varepsilon\xi_k]}
```
"""

# ╔═╡ f1bb19a5-87a8-44c8-89f5-af15351940eb
md"""
## Direct Derivation
"""

# ╔═╡ 87b1111f-b427-4893-9b09-9e0c75cbfefa
md"""
We assume that ``\varepsilon`` pushes the shockwave closer to the body (which only fixes a sign in our derivation, the logic would remain the same otherwise).
```math
u^\varepsilon(x, y) = u_L\cdot\mathcal X_{(-\infty, \Gamma_1^\varepsilon(y)]}(x) + u_R(x,y)\cdot\mathcal X_{(\Gamma_1^\varepsilon(y), \infty)}(x)
```

If we (naïvely) try to compute
```math
	\Delta u(x, y) =  \lim_{\varepsilon\to 0^+} \frac{1}{\varepsilon}\left(u^\varepsilon(x, y) - u(x, y)\right)
```
we have
```math
	\Delta u(x,y) = \lim_{\varepsilon\to0^+}\frac{1}{\varepsilon}\left[
\begin{split}(u_L(x, y, M_\infty+\varepsilon)-u_L(x, y, M_\infty))\cdot\mathcal X_{(-\infty, \Gamma_1(y)]}(x) \\ 
+ (u_L(x, y, M_\infty+\varepsilon) - u_R(x,y,M_\infty))\cdot\mathcal X_{(\Gamma_1(y), \Gamma_1^\varepsilon(y)]}(x)\\ 
+ (u_R(x,y,M_\infty+\varepsilon)-u_R(x, y, M_\infty))\cdot\mathcal X_{(\Gamma_1(y),\infty)}(x)
\end{split}\right]
```

The first and last terms of the sum in the limit resolve neatly. However, as ``\varepsilon\to0``, the second term is zero everywhere except for an infinitesemally small width ``[\Gamma_1(y), \Gamma_1^\varepsilon(y)]``, which we can identify with the components ``\xi_i`` in a generalized tangent vector.

Therefore, independent of the choice of shock parametrization, we have
```math
v(x,y) = \partial_Mu_L(x,y,M_\infty)\cdot\mathcal X_{(-\infty, \Gamma_1(y)]}(x) + \partial_M u_R(x, y, M_\infty)\cdot\mathcal X_{(\Gamma_1(y), \infty)}(x)
```
"""

# ╔═╡ bc3da011-3f0f-4e44-82d0-bab6b5f1f2b3
md"""
## Taylor Expansion of the Shock

We would like to be able to expand the shock position ``\Gamma`` like so:
```math
\Gamma(\alpha, M+\varepsilon) = \Gamma(\alpha, M) + \varepsilon\partial_M\Gamma(\alpha, M)
```

Our calculus relies on the assumption that the shock position _and_ the shock normal can be expanded in ``\varepsilon``. We can verify this by fixing the shock position parameter ``\alpha``, and checking that ``\left|\Gamma_1^\varepsilon(\alpha, M_\infty) - \Gamma_1(\alpha, M_\infty+\varepsilon)\right| \sim \mathcal O(\varepsilon^2)``.

In the plots below, we let ``y`` range over ``[0, 3]`` and choose ``dy<<ε``. 
"""

# ╔═╡ f46bb338-cd91-4ca1-a7b4-030576b5e9bc


# ╔═╡ b40f51ca-a47d-42f2-8d7f-013b78448b19
md"""
The approximation error is ``\mathcal{O}(\varepsilon^2)``. Our choice to use the Taylor expansion of ``\Gamma`` is safe.
"""

# ╔═╡ ff20dbe2-3d14-4af9-9f68-f5b9d4f410fa
md"""
## Taylor Expansion of the Flux

Adjusting the Mach number ``M_\infty`` will change the conditions behind the shock wave as well as shift the position of the shock wave and the shock normals. We can perform a first-order Taylor expansion in ``M``, starting with ``\Gamma`` and working outward through ``\left[F_n\left(\vec u\right)\right] = 0``.

In general, the shock front depends on the free stream state (``M_\infty, T_\infty, \rho_\infty``) and the size of the blunt body (``R``). For this derivation, we'll focus on a perturbation in ``M_\infty``, but the same method should work for a general perturbation vector, with extra bookkeeping for the indices in the parameter vectors. The shock wave itself is parametrized by a value ``\alpha``. 
We abbreviate the following algebra by removing arguments when any of ``\vec u``, ``\Gamma``, or ``\mathbf{\hat n}`` are evaluated at ``(M_\infty, \alpha)``.
"""

# ╔═╡ 1368d709-7f30-40fd-ab5d-20ae0c69da7a
md"""
```math
	\Gamma(M_\infty + \varepsilon, \alpha) = \Gamma(M_\infty, \alpha) + \varepsilon\partial_M\Gamma(M_\infty,\alpha)+\mathcal O (\varepsilon^2)
```

Now feed the previous expansion into the Rankine-Hugoniot condition at the shockwave, being careful to only discard ``\mathcal O(\varepsilon^2)`` terms...

```math
\begin{align}
0 &= F\left(\vec u_L(M_\infty+\varepsilon, \Gamma + \varepsilon\partial_M\Gamma)\right)\mathbf{\hat n}-\ldots\\
0 &= F\left(\vec u_L(M_\infty+\varepsilon, \Gamma)+\varepsilon\left(\nabla_\Gamma\vec u_L(M_\infty+\varepsilon, \Gamma)\right)\partial_M\Gamma\right)\mathbf{\hat n}-\ldots\\
0 &= F\left(\vec u_L+\varepsilon\partial_M\vec u_L+\varepsilon\left(\nabla_\Gamma\vec u_L(M_\infty+\varepsilon, \Gamma)\right)\partial_M\Gamma\right)\mathbf{\hat n}-\ldots
\end{align}
```

Expanding ``\varepsilon\nabla_\Gamma\vec u_L(M_\infty+\varepsilon, \Gamma)\cdot\partial_M\Gamma`` will create an ``\mathcal O\left(\varepsilon^2\right)`` term.
"""

# ╔═╡ bd18ef3d-36b7-482a-809d-e7f843f9867b
md"""
```math
\begin{align}
0 &= F\left(\vec u_L+\varepsilon\left(\partial_M\vec u_L+\left(\nabla_\Gamma\vec u_L\right)\partial_M\Gamma\right)\right)\mathbf{\hat n}(M_\infty+\varepsilon, \beta)-\ldots\\
0 &= F\left(\vec u_L+\varepsilon\left(\partial_M\vec u_L+\left(\nabla_\Gamma\vec u_L\right)\partial_M\Gamma\right)\right)\left(\mathbf{\hat n}+\varepsilon\partial_M\mathbf{\hat n}\right)-\ldots\\
0 &= \left(F\left(\vec u_L\right) + \varepsilon\nabla_u F\left(\vec u_L\right)\left(\partial_M\vec u_L+\left(\nabla_\Gamma\vec u_L\right)\partial_M\Gamma\right)\right)\left(\mathbf{\hat n} + \varepsilon\partial_M\mathbf{\hat n}\right)-\ldots\\
0 &= F\left(\vec u_L\right)\mathbf{\hat n} + \varepsilon F\left(\vec u_L\right)\partial_M\mathbf{\hat n} +\left(\varepsilon\nabla_u F\left(\vec u_L\right)\left(\partial_M\vec u_L+\left(\nabla_\Gamma\vec u_L\right)\partial_M\Gamma\right)\right)\mathbf{\hat n}-\ldots
\end{align}
```

The final line above passes a sanity check: If ``\vec u\in\mathbb R^4``, each term eventually reduces to the correct dimensions. If we gather the terms that only differ by ``L`` or ``R``, we will have the following:
"""

# ╔═╡ 9d719826-a9f3-4914-86e0-5b62ffdd2908
md"""
```math
0 = \begin{aligned}
& \left[F\left(\vec u\right)\mathbf{\hat n}\right] + \varepsilon\left[F\left(\vec u\right)\right]\partial_M\mathbf{\hat n}\, + \\ &\varepsilon\left[\nabla_u F(\vec u)\,\partial_M\vec u\right]\mathbf{\hat n} + \varepsilon\left[\nabla_uF(\vec u)\,\nabla_\Gamma\vec u\,\partial_M\Gamma\right]\mathbf{\hat n}
\end{aligned}
```

Since the jump conditions ``\left[F_n(\vec u)\right] = 0`` are satisfied, we can discard the first term.
"""

# ╔═╡ 036ce73c-9b2b-4b78-80be-bdc5a7900a78
md"""
```math
0 = \left[F\left(\vec u\right)\right]\partial\mathbf{\hat n} + \left[\nabla_u F(\vec u)\,\partial_M\vec u\right]\mathbf{\hat n} + \left[\nabla_u F(\vec u)\,\nabla_\Gamma\vec u\,\partial_M\Gamma\right]\mathbf{\hat n} + \mathcal O(\varepsilon^2)
```

The idea would be to compute ``∂M\vec{u}_R`` via solving the above equation (up to ``\mathcal O(\varepsilon^2)``), but the equation is too difficult to invert. Instead, we'll substitute the expansion using generalized tangent vectors into the flux and verify that ``[F_n^\varepsilon(u)]`` converges with ``\mathcal O(\varepsilon^2)``.
"""

# ╔═╡ ea1d89cb-b00a-4bbb-aaab-147d827bdca5
md"""
We want to demonstrate, for a shock wave originally located at ``x_k``:
```math
\begin{align}
0 &= \left[F_n^ε(u^\varepsilon, x^ε_k)\right]\\
0 &= F_n^ε(u^\varepsilon_L, x_k^\varepsilon)-F_n^\varepsilon(u^\varepsilon_R, x_k^\varepsilon)\\
\end{align}
```
We need to carefully compute ``u_L^\varepsilon`` and ``u_R^\varepsilon``:
```math
\begin{align}
0 &= 
\begin{aligned}&F_n^\varepsilon(u_L(x_k)+\varepsilon v_L(x_k))\,-\\
&F_n^\varepsilon(u_R(x_k)+ \varepsilon v_R(x_k))\\
\end{aligned}
\end{align}
```
"""

# ╔═╡ ffe618e9-da93-4f1b-9ebb-6e1cf246c201
md"""
# Verifying the Approximation
The expansion outlined above does not depend on the particular properties of the parametrisation for the shock front ``\Gamma``. We will use Billig's formulae for shock fronts around blunt, round-nosed bodies in 2- and 3- dimensions. 
"""

# ╔═╡ 28bbb112-89a8-4f75-83f4-622b31da11b7
free_stream = PrimitiveProps(1.225, [4.0, 0.0], 300.)

# ╔═╡ 641c0e4e-55ce-46dc-97fc-aa3422c4f8c7
fs = state_to_vector(free_stream);

# ╔═╡ 2d411de5-2421-4bfe-90ed-afaed98f1dfa
# define ξ
function ∂Γ∂M(α, M)
	# use forward mode to make the plots (marginally) faster
	J = ForwardDiff.derivative(M) do Minf
		shock_front(α, Minf, 1.0)
	end
	return J
end

# ╔═╡ 8d0f4afe-4a32-4011-add3-b7adac642d5c
function v_jump(s_L, x, y; gas::CaloricallyPerfectGas = DRY_AIR)
	_, u_back = pullback(u0_jump, s_L, x, y) 
	n=length(s_L)
	J_L = mapreduce(vcat, (1.0*I[1:n, k] for k∈1:n)) do ū
		u_back((ū, zeros(4)))[1]'
	end
	J_R = mapreduce(vcat, (1.0*I[1:n, k] for k∈1:n)) do ū
		u_back((zeros(4),ū))[1]'
	end
	# we only need ∂u∂M
	return (J_L[:, 2:end-1], J_R[:, 2:end-1])
end

# ╔═╡ c484959a-bac1-44f5-bade-bc09e03ecc81
function ∂u0∂s(s_L, x, y; gas::CaloricallyPerfectGas=DRY_AIR)
	u, u_back = pullback(s_L, x, y) do a1, a2, a3
		u0(a1, a2, a3; gas=gas)
	end
	n = length(s_L)
	# accumulate the jacobian wrt. u_L
	J_M = mapreduce(vcat, (1.0*I[1:n, k] for k∈1:n)) do ū
		return u_back(vcat(ū))[1]'
	end
	return J_M
end

# ╔═╡ bce1447a-8ff0-479f-af0a-c61c5d0750fe
∂u0∂M(s_L, x, y; gas::CaloricallyPerfectGas=DRY_AIR) = ∂u0∂s(s_L, x, y; gas=gas)[:, 2:end-1]

# ╔═╡ 420b9b5c-8bc1-45aa-847d-2296ed75dfcd
function uε(s_L, x, y, ε) 
	Γ = shock_front(y, s_L[2], 1.0)[1]
	u_L, u_R = u0_jump(s_L, x, y; gas=DRY_AIR)
	v_L, v_R = v_jump(s_L, x, y; gas=DRY_AIR)
	
	ξ = ∂Γ∂M(y, s_L[2])[1]
	jump = (u_L - u_R) * (ξ < 0 ? -1 : 1)
	u = x < Γ ? u_L : u_R
	εv = (x < Γ ? v_L : v_R) * [ε, 0.0]

	return u + εv + jump * (Γ < x < Γ+ε*ξ || Γ +ε*ξ < x < Γ)
end

# ╔═╡ a1afc6e7-bdae-44bc-8381-5d55619dd671
# returns uε(s, x, y, ε) - u(s+ε, x, y)
function compare_expansion_to_actual(s_L, x, y, ε)
	n = length(s_L)
	s_Lε = s_L + ε*I[1:n, 2]
	# close over states
	err = (x, y, ε) -> begin
		theo = u0(s_Lε, x, y)
		return uε(s_L, x, y, ε) - theo
	end
	return permutedims( stack(@. err(x', y, ε) ), (2, 3, 1))
end

# ╔═╡ 0ede910a-5b77-460d-9175-d59bcb723e93
md"""
ε = $(@bind dM confirm(Slider(0.005:0.005:0.5, default=0.15, show_value=true)))
"""

# ╔═╡ d21322f9-2b8b-4afb-bc7f-8e667715e2e1
begin
	ε = dM
	x = -1.6:0.0025:-1.25
	y = 0.0:0.0005:1.0	
	Γ0 = mapreduce(vcat, y) do yk
		shock_front(yk, fs[2], 1.0)'
	end
	Γε = mapreduce(vcat, y) do yk
		(shock_front(yk, fs[2], 1.0) + ε * ∂Γ∂M(yk, fs[2]))'
	end
end;

# ╔═╡ a57c7ae0-b92f-4537-9d53-1bdc5dd34eb5
err = compare_expansion_to_actual(fs, x, y, ε);

# ╔═╡ 232ad815-0cf8-4c79-af16-303f04ca3b51
heatmap_plot(idx) = begin
		p = heatmap(x, y, err[:, :, idx],
			fill=true,
			xlims=(first(x), last(x)), ylims=(first(y), last(y)), dpi=600,
			title="Error in $([L"\rho", L"\rho v_x", L"\rho v_y", L"\rho E"][idx])")
	plot!(p, Γ0[:, 1], Γ0[:, 2], color=:blue, label=L"Γ^0", lw=2, ls=:dash)
	plot!(p, Γε[:, 1], Γε[:, 2], color=:red, label=L"Γ^\varepsilon", lw=2, ls=:dash)
end

# ╔═╡ 1d60de48-05de-4798-8d07-7d70dc3f6ea8
let 
	p = plot(reshape([heatmap_plot(i) for i=1:4], (2,2))...)
	# savefig(p, "../../fluid-dynamics-conference-2024/abstract-submission/2d_error_plots.pdf")
	p
end

# ╔═╡ d2aa5a4e-c08f-4e11-b4f4-c481c358ff0a
md"""
 - The difference between the actual position of the shock wave and its estimated position is shown by the black band of large error. As the Mach number increases, this band gets narrower, since the relative displacement of the shock wave decreases.
 - The remaining error is expansion error from the jump computations.
 - The plot for error in total internal energy density (``\rho E``) appears similar.
 - The plots for ``M_x`` and ``M_y`` have larger error further away from ``y=0`` due to the expansion error introduced by the larger value of ``M\cdot\hat t``.
"""

# ╔═╡ bdb5cdfe-ab17-4a8a-b4e0-7368a92491cb
# compute L1 norm of f using trapezoidal quadrature
function norm_L1(f_data, dx::Number, dy::Number)
	dA = dx*dy
	# corner terms (map abs first)
	res = sum(abs, (f_data[1,1], f_data[end, end], 
					f_data[1, end], f_data[end, 1]))/4 * dA
	@views begin
		# edge terms
		res += sum(abs, f_data[2:end-1, 1]) + sum(abs, f_data[2:end-1, end]) / 2 * dA
		res += sum(abs, f_data[1, 2:end-1]) + sum(abs, f_data[end, 2:end-1]) / 2 * dA
		# bulk terms
		res += sum(abs, f_data[2:end-1, 2:end-1]) * dA
	end
	return res
end

# ╔═╡ 9c96676e-9202-4d90-95b6-fd4b92d4eb56
norm_L1(f_data, dt::Number) = @views 0.5 * dt * sum(abs.(f_data[1:end-1]) + abs.(f_data[2:end]))

# ╔═╡ c7c00c0f-75a7-417e-a007-0c774f121ba5
err_data = begin
	εs = [0.1, 0.01, 0.001, 0.0001, 1.0e-5]
	mapreduce(vcat, εs) do ε
		err = compare_expansion_to_actual(fs, x, y, ε);
		vec(mapslices(data -> norm_L1(data, step(x), step(y)), err, dims=(1,2)))'
	end
end;

# ╔═╡ 2f4998a9-6442-42cc-b63d-5c99a4cc9102
let
	titles = [L"\|\rho^\varepsilon - \rho\|", L"\|M_x^\varepsilon - M_x\|", L"\|M_y^\varepsilon - M_y\|", L"\|(\rho E)^\varepsilon - \rho E\|"]
	subplots = [
		plot(εs, [1000. .* εs.^2, err_data[:,i]], 
			title=titles[i], labels=[L"\mathcal{O}(\varepsilon^2)" "Computed"], marker=[:plus :x], minorticks=true, grid=true,
			yscale=:log10, xscale=:log10, legend=(i==1 ? :topleft : false), xflip=true)
		for i=1:4
	]
	xlabel!.(subplots[3:4], L"\varepsilon")
	ylabel!(subplots[1], L"\|\cdot\|_{L^1}")
	ylabel!(subplots[3], L"\|\cdot\|_{L^1}")
	p = plot(subplots...)
	savefig(p, "../../Euler2D.jl/gfx/convergence_plots.pdf")
	p
end

# ╔═╡ bad68a26-5784-4d05-8d36-c83d3cc4ac57
function uε_jump(s_L, α, ε)
	# choice of x shouldn't matter
	u_L, u_R = u0_jump(s_L, 0.0, α; gas=DRY_AIR)
	v_L, v_R = v_jump(s_L, 0.0, α; gas=DRY_AIR)
	ξ = ∂Γ∂M(α, s_L[2])[1]
	
	jump = (u_L - u_R) * (ξ < 0 ? -1 : 1)
	new_R = u_R + v_R*[ε, 0.0]
	new_L = new_R + jump
	
	return (new_L, new_R)
end

# ╔═╡ 0875ebbc-7cd1-4fe1-9999-ce6bacab7756
alleps = [1*10^(-n) for n=1.:7.]

# ╔═╡ 0352c09e-7c1b-48e2-847b-943990628c8b
function ∂n̂∂M(α, M)
	∂n̂ = ForwardDiff.derivative(M) do Minf
		shock_normal(α, Minf, 1.0)
	end
	return ∂n̂
end

# ╔═╡ be87856e-320d-4611-8ad0-c255b96e7827
let α = 0.0:0.005:3.0, εs = [0.5, 0.1, 0.01, 0.001, 0.0001]
	Γ = shock_front.(α, 4.0, 1.0)
	n = shock_normal.(α, 4.0, 1.0)
	shock_displacement = map(εs) do ε
		Γ1 = shock_front.(α, 4.0+ε, 1.0)
		Γ2 = Γ + ε .* ∂Γ∂M.(α, 4.0)
		dists = norm.(Γ2 - Γ1)
		return norm_L1(dists, step(α))
	end
	normal_displacements = map(εs) do ε
		n1 = shock_normal.(α, 4.0+ε, 1.0)
		n2 = n + ε .* ∂n̂∂M.(α, 4.0)
		dists = norm.(n2 - n1)
		return norm_L1(dists, step(α))
	end
	p = scatter(εs, shock_displacement, xscale=:log10, yscale=:log10, title=L"L^1"*" Error in Shockwave Displacement", label=L"\Gamma^ε(α, M) - \Gamma(α, M+ε)", ylabel=L"\|\cdot\|_{L^1}", xlabel=L"\varepsilon", grid=true, minorticks=true, minorgrid=true, legend=:bottomleft, dpi=300)
	xflip!(p)
	scatter!(p, εs , normal_displacements, label=L"\mathbf{\hat{n}}^\varepsilon(α, M) - \mathbf{\hat{n}}(α, M+ε)")
	plot!(p, εs, hcat(0.1 .* εs, 0.1 .* εs.^2), labels=[L"\mathcal{O}(\varepsilon)" L"\mathcal{O}(\varepsilon^2)"], ls=:dashdot)
	p
end

# ╔═╡ be7c85a0-064f-4e57-bd50-84d29c6ce724
function flux_difference(s_L, α, ε)
	# Γε = shock_front(α, s_L[2], 1.0) + ε * ∂Γ∂M(α, s_L[2])
	n̂ε = shock_normal(α, s_L[2], 1.0)  + ε * ∂n̂∂M(α, s_L[2])
	(uε_L, uε_R) = uε_jump(s_L, α, ε)
	flux = (F(uε_L) - F(uε_R)) * n̂ε
	return flux
end

# ╔═╡ b36900f0-07bc-4c74-9498-2a4a32a2621f
flux_convergence_data = let
	alpha = 0:0.0005:1
	integrator = Base.Fix2(norm_L1, step(alpha))
	data = mapreduce(vcat, alleps) do ε
		err = map(integrator, mapreduce(vcat, alpha) do α
			flux_difference(fs, α, ε)'	
		end |> eachcol)
		err'
	end
end;

# ╔═╡ 00498c7c-7b03-42c7-92a9-310e07fd05a2
let
	p = plot(alleps, 10000*alleps, ls=:dashdot, yscale=:log10, xscale=:log10, legend=:topleft, label=L"O(\varepsilon)", title="Error in the R-H Condition", xlabel=L"\varepsilon", ylabel=L"E")
	scatter!(p, alleps, flux_convergence_data, labels=[L"(F_n(u))_1" L"(F_n(u))_2" L"(F_n(u))_3" L"(F_n(u))_4"])
	p
end

# ╔═╡ 035e78c5-568c-437a-ab9d-1b9c8ea265f6
md"""
Note: we compute the error from the flux expansion above via:
```math
E = \int_0^1\left|\left[F^ε(u, x_k^ε\right]\mathbf{\hat n}^ε(α)\right|\,d\alpha
```
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShockwaveProperties = "77d2bf28-a3e9-4b9c-9fcf-b85f74cc8a50"
Tullio = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"
UnitfulChainRules = "f31437dd-25a7-4345-875f-756556e6935d"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ForwardDiff = "~0.10.36"
LaTeXStrings = "~1.3.1"
Plots = "~1.40.1"
PlutoUI = "~0.7.58"
ShockwaveProperties = "~0.1.3"
Tullio = "~0.3.7"
Unitful = "~1.19.0"
UnitfulChainRules = "~0.1.2"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "8fccf0338ac5308da4c2eaf81238fcfb9982e62e"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "a975ae558af61a2a48720a6271661bf2621e0f4e"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.3"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "eea7b3a1964b4de269bb380462a9da604be7fcdb"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "7ffa4049937aeba2e5e1242274dc052b0362157a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.14"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "98fc192b4e4b938775ecd276ce88f539bcec358e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.14+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "80d268b2f4e396edc5ea004d1e0f569231c71e9e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.34"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "f0e861832695dbb70e710606a7d18b7f81acec92"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.3.1"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4b5ad6a4ffa91a00050a964492bc4f86bb48cea0"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.35+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "809ba625a00c605f8d00cd2a9ae19ce34fc24d68"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.13"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShockwaveProperties]]
deps = ["ChainRulesCore", "LinearAlgebra", "Unitful", "UnitfulChainRules"]
git-tree-sha1 = "b5c1b1cc410447f01e3a16ccd174af56a39d2310"
uuid = "77d2bf28-a3e9-4b9c-9fcf-b85f74cc8a50"
version = "0.1.11"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.Tullio]]
deps = ["DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "972698b132b9df8791ae74aa547268e977b55f68"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.8"

    [deps.Tullio.extensions]
    TullioCUDAExt = "CUDA"
    TullioChainRulesCoreExt = "ChainRulesCore"
    TullioFillArraysExt = "FillArrays"
    TullioTrackerExt = "Tracker"

    [deps.Tullio.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "352edac1ad17e018186881b051960bfca78a075a"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulChainRules]]
deps = ["ChainRulesCore", "LinearAlgebra", "Unitful"]
git-tree-sha1 = "24a349fa6f4fbcdc12830c5640baa6eae532f83f"
uuid = "f31437dd-25a7-4345-875f-756556e6935d"
version = "0.1.2"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8462a20f0fd85b4ef4a1b7310d33e7475d2bb14f"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.77"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "002748401f7b520273e2b506f61cab95d4701ccf"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.48+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "c950ae0a3577aec97bfccf3381f66666bc416729"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.8.1+0"
"""

# ╔═╡ Cell order:
# ╠═e93237ca-27d8-415f-a4dd-99eb4f530f69
# ╠═2af7b8a2-b5d7-4957-84c2-815d52b0ebd3
# ╠═00f60ab4-aa32-4f0d-bf83-c175de05066e
# ╟─ca06cf02-5844-4487-ab22-5e6bf7547378
# ╠═6d4f0e92-2fd8-4d91-bb50-8a5b18dde745
# ╟─6e91e9ed-473d-4496-b589-85ac5c954877
# ╠═a1789327-d3a8-4c6c-8fc0-64e8d81cc98d
# ╠═9f358218-6eea-4d50-8374-f1c7c3b65e2d
# ╠═417f01cc-dae8-4876-96a9-13f2b6b4ed77
# ╠═246c97ad-1087-484a-90b0-bee0c09c61ae
# ╟─78adce9e-5253-45d1-9e28-87012a74262e
# ╟─c97a2cdd-7ff7-4e86-a026-1fb4f9287628
# ╠═0ed965e6-7662-417b-b750-eccdc4ba2184
# ╟─a505438f-a1c8-451b-a04f-c7efdd5a94a6
# ╠═39f6d764-f16b-4dd5-9c26-f41688915ca5
# ╠═8f28cd61-82a7-41af-b3d0-1e48e29d1430
# ╠═f727b337-bfa0-4d45-b593-6ad5607930ae
# ╟─8a4e3e3e-9074-49bf-840d-3888579d5ba2
# ╟─f1bb19a5-87a8-44c8-89f5-af15351940eb
# ╟─87b1111f-b427-4893-9b09-9e0c75cbfefa
# ╟─bc3da011-3f0f-4e44-82d0-bab6b5f1f2b3
# ╠═be87856e-320d-4611-8ad0-c255b96e7827
# ╠═f46bb338-cd91-4ca1-a7b4-030576b5e9bc
# ╟─b40f51ca-a47d-42f2-8d7f-013b78448b19
# ╟─ff20dbe2-3d14-4af9-9f68-f5b9d4f410fa
# ╟─1368d709-7f30-40fd-ab5d-20ae0c69da7a
# ╟─bd18ef3d-36b7-482a-809d-e7f843f9867b
# ╟─9d719826-a9f3-4914-86e0-5b62ffdd2908
# ╟─036ce73c-9b2b-4b78-80be-bdc5a7900a78
# ╟─ea1d89cb-b00a-4bbb-aaab-147d827bdca5
# ╠═be7c85a0-064f-4e57-bd50-84d29c6ce724
# ╟─ffe618e9-da93-4f1b-9ebb-6e1cf246c201
# ╠═28bbb112-89a8-4f75-83f4-622b31da11b7
# ╠═641c0e4e-55ce-46dc-97fc-aa3422c4f8c7
# ╠═2d411de5-2421-4bfe-90ed-afaed98f1dfa
# ╠═8d0f4afe-4a32-4011-add3-b7adac642d5c
# ╠═c484959a-bac1-44f5-bade-bc09e03ecc81
# ╠═bce1447a-8ff0-479f-af0a-c61c5d0750fe
# ╠═420b9b5c-8bc1-45aa-847d-2296ed75dfcd
# ╠═a1afc6e7-bdae-44bc-8381-5d55619dd671
# ╠═d21322f9-2b8b-4afb-bc7f-8e667715e2e1
# ╠═232ad815-0cf8-4c79-af16-303f04ca3b51
# ╠═1d60de48-05de-4798-8d07-7d70dc3f6ea8
# ╠═a57c7ae0-b92f-4537-9d53-1bdc5dd34eb5
# ╟─0ede910a-5b77-460d-9175-d59bcb723e93
# ╟─d2aa5a4e-c08f-4e11-b4f4-c481c358ff0a
# ╟─bdb5cdfe-ab17-4a8a-b4e0-7368a92491cb
# ╟─9c96676e-9202-4d90-95b6-fd4b92d4eb56
# ╠═2f4998a9-6442-42cc-b63d-5c99a4cc9102
# ╠═c7c00c0f-75a7-417e-a007-0c774f121ba5
# ╠═bad68a26-5784-4d05-8d36-c83d3cc4ac57
# ╠═0875ebbc-7cd1-4fe1-9999-ce6bacab7756
# ╠═0352c09e-7c1b-48e2-847b-943990628c8b
# ╠═b36900f0-07bc-4c74-9498-2a4a32a2621f
# ╟─00498c7c-7b03-42c7-92a9-310e07fd05a2
# ╟─035e78c5-568c-437a-ab9d-1b9c8ea265f6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
