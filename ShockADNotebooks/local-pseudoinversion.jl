### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> title = "Local Pseudoinversion"
#> date = "2025-07-18"
#> description = "A local pseudoinversion technique for solutions to the Euler Equations."
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

# ╔═╡ 6f1542ea-a747-11ef-2466-fd7f67d1ef2c
begin # public packages
    using Accessors
    using DifferentiationInterface
    using ForwardDiff
	using Graphs
    using Interpolations: linear_interpolation
    using LaTeXStrings
    using LinearAlgebra
	using MetaGraphsNext
    import Mooncake
	using OhMyThreads
    using Plots
    using PlutoUI
    using Printf
    using StaticArrays
    using Unitful
end

# ╔═╡ 2e9dafda-a95c-4277-9a7c-bc80d97792f0
begin #STCE Packages
	using Euler2D
    using Euler2D: TangentQuadCell
	using PlanePolygons
	using ShockwaveProperties
end

# ╔═╡ 3aea1823-86ff-4b8b-9a4c-c87a4fbf4dff
using Tullio

# ╔═╡ e5036dd3-9070-4521-9d7d-e0293b967d78
using ShockwaveProperties: BilligShockParametrization

# ╔═╡ d8bfb40f-f304-41b0-9543-a4b10e95d182
using PlanePolygons: _poly_image

# ╔═╡ e4b54bd3-5fa9-4291-b2a4-6b10c494ce34
# get some internal bindings from Euler2D
using Euler2D: _dirs_dim, select_middle

# ╔═╡ f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
PlutoUI.TableOfContents()

# ╔═╡ e7001548-a4b4-4709-b10c-0633a11bd624
md"""
# Local Pseudoinversion

This notebook describes the current state of my work on something I've dubbed "local pseudoinversion", which can be applied to a solution to the Euler equations in order to compute the shock shift component of a generalized tangent vector to the solution.

It may also be applicable to other hyperbolic partial differential equations, but I cannot make any guarantees at this time.
"""

# ╔═╡ c87b546e-8796-44bf-868c-b2d3ad340aa1
md"""
## Setup
Declare ``u(x, 0; p) = u_0`` and provide a useful scale to to nondimensionalize the Euler equations.

The parameters (taken from a previously-done simulation) are:
 - ``ρ_\inf=0.662\frac{\mathrm{kg}}{\mathrm{m}^3}``
 - ``M_\inf=4.0``
 - ``T_\inf=220\mathrm{K}``

"""

# ╔═╡ 4267b459-7eb7-4678-8f06-7b9deab1f830
const ambient_primitives = SVector(0.662, 4.0, 220.0)

# ╔═╡ 2716b9b5-07fd-4175-a83e-22be3810e4b3
md"""
We can set up `u0` to always return the conserved quantities computed from the ambient primitives; this also yields a nondimensionalization scale.
"""

# ╔═╡ afc11d27-1958-49ba-adfa-237ba7bbd186
function u0(x, p)
    # ρ, M, T -> ρ, ρv, ρE
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

# ╔═╡ 0df888bd-003e-4b49-9c2a-c28a7ccc33d2
const ambient = u0((-Inf, 0.0), ambient_primitives);

# ╔═╡ 3fe1be0d-148a-43f2-b0a5-bb177d1c041d
sim_scale = EulerEqnsScaling(
    1.0u"m",
    ShockwaveProperties.density(ambient),
    speed_of_sound(ambient, DRY_AIR),
)

# ╔═╡ 136ab703-ae33-4e46-a883-0ed159360361
const ambient_u = nondimensionalize(ambient, sim_scale)

# ╔═╡ a9d31e2d-fc4b-4fa5-9015-eb2ac2a3df5d
const ambient_u̇ = ForwardDiff.jacobian(ambient_primitives) do prim
    nondimensionalize(u0((-Inf, 0.0), prim), sim_scale)
end

# ╔═╡ d832aeb4-42d6-4b72-88ee-4cdd702a4f48
md"""
Load up a data file. This contains a forward-mode computation on a fine grid allowed to run to $T=15$.
"""

# ╔═╡ 90bf50cf-7254-4de8-b860-938430e121a9
const sim_with_ad = 
	#load_cell_sim("../data/tangent_last_tstep.celltape");
	#load_cell_sim("../data/probe_obstacle_tangent_very_long_time_selected_tsteps.celltape");
	load_cell_sim("../data/60691549/11/bow_shock_t20.celltape")

# ╔═╡ d29aa465-fcba-4210-9809-c92e6bf604bf
const FINAL_TSTEP_IDX = n_tsteps(sim_with_ad)

# ╔═╡ 09efacdf-2e99-4e74-b00b-b021955a3220
const NCELLS_X = sim_with_ad.ncells[1]

# ╔═╡ 7422611b-596a-48e1-9e24-0e8d8bd1eba8
const NCELLS_Y = sim_with_ad.ncells[2]

# ╔═╡ fffcb684-9b58-43d7-850a-532c609c5389
const boundary_conditions = (ExtrapolateToPhantom(), StrongWall(), ExtrapolateToPhantom(), ExtrapolateToPhantom(), StrongWall());

# ╔═╡ 33e635b3-7c63-4b91-a1f2-49da93307f29
md"""
We also know that this simulation was done with a blunt probe with front radius ``0.75`` located at the origin.

Unfortunately, propagating sensitivities to boundary conditions requires more a more advanced finite volume solver than Euler2D. 
"""

# ╔═╡ 0cedbab2-ade9-4e8d-aa3c-a0c9cc28eaaa
body = CircularObstacle(SVector(0.0, 0.0), 0.75)

# ╔═╡ 8bd1c644-1690-46cf-ac80-60654fc6d8c0
md"""
## Pressure Field Sensitivities

The Euler2D code supports forwards-mode algorithmic differentiation; the tangent values ``\dot u(t, x)`` are computed by the handwritten tangent to the finite volume update rule (here in the x-direction):

```math
\dot{u}_{i,j}^{n+1} = \dot{u}_{i,j}^{n, (1)} + \frac{\Delta t}{\Delta x}\left(\frac{\partial\mathcal{F}_{i-\frac{1}{2}}(u^n)}{\partial u}\dot{u}^{n} - \frac{\partial\mathcal{F}_{i+\frac{1}{2}}(u^n)}{\partial u}\dot{u}^{n}\right)
```

Here we use ``\dot u_{i,j}`` to denote the Jacobian matrix ``\frac{\partial u_{i,j}}{\partial p_k}``, which is seeded at ``t=0``.

Previous work indicates that this is an appropriate update rule for the tangent; a proof that the tangent of the discretization is the same as the discretization of the tangent is not available.
"""

# ╔═╡ 893ec2c8-88e8-4d72-aab7-88a1efa30b47
function dPdp(sim::CellBasedEulerSim{T,C}, n) where {T,C<:Euler2D.TangentQuadCell}
    _, u_cells = nth_step(sim, n)
    res = Array{Union{T,Missing}}(missing, (3, grid_size(sim)...))
    for i ∈ eachindex(IndexCartesian(), sim.cell_ids)
        sim.cell_ids[i] == 0 && continue
        cell = u_cells[sim.cell_ids[i]]
        dP = ForwardDiff.gradient(cell.u) do u
            Euler2D._pressure(u, DRY_AIR)
        end
        res[:, i] = dP' * cell.u̇
    end
    return res
end

# ╔═╡ d14c3b81-0f19-4207-8e67-13c09fd7636a
md"""
With the tangents available, it is possible to compute the dependence of field quantities (pressure, temperature, magnitude of the mach number) on the input paramaters ``p``.

Computing the full gradient ``\partial_{p_k} P`` is a bit finicky, but ultimately works out to be repeated Jacobian-vector products over the pressure field.
"""

# ╔═╡ 2e3b9675-4b66-4623-b0c4-01acdf4e158c
@bind n_pplot Slider(1:n_tsteps(sim_with_ad); default = 2, show_value = true)

# ╔═╡ f6147284-02ec-42dd-9c2f-a1a7534ae9fa
pfield = map(pressure_field(sim_with_ad, n_pplot, DRY_AIR)) do val
    isnothing(val) ? missing : val
end;

# ╔═╡ cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
pressure_tangent = dPdp(sim_with_ad, n_pplot);

# ╔═╡ d5db89be-7526-4e6d-9dec-441f09606a04
let n = n_pplot
    pplot = heatmap(
        pfield';
        xlims = (0, NCELLS_X),
        ylims = (0, NCELLS_Y),
        aspect_ratio = :equal,
        title = L"P",
    )
    cbar_limits = (:auto, (-2, 10), :auto)
    titles = [
        L"\frac{\partial P}{\partial \rho_\inf}",
        L"\frac{\partial P}{\partial M_\inf}",
        L"\frac{\partial P}{\partial T_\inf}",
    ]
    dpplot = [
        heatmap(
            (@view(pressure_tangent[i, :, :]))';
            xlims = (0, NCELLS_X),
            ylims = (0, NCELLS_Y),
            aspect_ratio = :equal,
            clims = cbar_limits[i],
            title = titles[i],
        ) for i = 1:3
    ]
    plots = reshape(vcat(pplot, dpplot), (2, 2))
    xlabel!.(plots, L"i")
    ylabel!.(plots, L"j")
    p = plot(plots...; size = (800, 800), dpi = 1000)
	savefig(p, "../gfx/pressure_derivatives_n$n.pdf")
	p
end

# ╔═╡ 0536f540-fa7b-48ed-8746-cd7cbce16555
let
pplot = heatmap(
        pfield';
        xlims = (0, NCELLS_X),
        ylims = (0, NCELLS_Y),
	size = (600, 450),
        aspect_ratio = :equal,
        title = L"P(\vec x)",
    )
	savefig(pplot, "../gfx/pressure_plot.pdf")
	pplot
end

# ╔═╡ db1c858e-a483-40ef-aabc-2aa58de73e92
md"""
There's some interesting numerical effects happening here with ``\partial_{M_\infty}P``, but no time to invstigate them!
"""

# ╔═╡ 4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
@bind n2 Slider(1:n_tsteps(sim_with_ad), default = 2, show_value = true)

# ╔═╡ bcdd4862-ac68-4392-94e2-30b1456d411a
let n = n2
    dPdM = dPdp(sim_with_ad, n)
    title = L"\partial P(\vec x) / \partial M_\inf"
    p = heatmap(
        (@view(dPdM[2, :, :]))';
        xlims = (0, NCELLS_X),
        ylims = (0, NCELLS_Y),
        aspect_ratio = :equal,
        clims = (-10, 10),
        title = title,
        size = (600, 450),
    )
	#savefig(p, "../gfx/single_dMdP_n$(n).pdf")
    p
end

# ╔═╡ 44ff921b-09d0-42a4-8852-e911212924f9
md"""
## Shock Sensor
Implementation of the technique proposed in _Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows_.
"""

# ╔═╡ 4f8b4b5d-58de-4197-a676-4090912225a1
md"""
---
"""

# ╔═╡ 6e4d2f60-3c40-4a2b-be2b-8c4cc40fb911
function pad_pfield_from_bcs(pfield)
	padded = similar(pfield, eltype(pfield), size(pfield).+2)
	padded[2:end-1, 2:end-1] = pfield
	@views padded[1, 2:end-1] = padded[2, 2:end-1]
	@views padded[end, 2:end-1] = padded[end-1, 2:end-1]
	@views padded[:, 1] = padded[:, 2]
	@views padded[:, end] = padded[:, end-1]
	return padded
end

# ╔═╡ 706146ae-3dbf-4b78-9fcc-e0832aeebb28
_diff_op(T) = SVector{3,T}(-one(T), zero(T), one(T))

# ╔═╡ 9b6ab300-6434-4a96-96be-87e30e35111f
_avg_op(T) = SVector{3,T}(one(T), 2 * one(T), one(T))

# ╔═╡ df4a4b15-b581-4e3c-a363-c1ac5a5a2999
function _convolve_tullio(mat, op)
	return @tullio out[i,j] := mat[i+k+1,j+l+1]*op[k+2,l+2]
end

# ╔═╡ 21cbdeec-3438-4809-b058-d23ebafc9ee2
# function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
#     Gy = _avg_op(T) * _diff_op(T)'
#     Gx = _diff_op(T) * _avg_op(T)'
#     new_size = size(matrix) .- 2
#     outX = similar(matrix, new_size)
#     outY = similar(matrix, new_size)
#     for i ∈ eachindex(IndexCartesian(), outX, outY)
#         view_range = i:(i+CartesianIndex(2, 2))
#         outX[i] = Gx ⋅ @view(matrix[view_range])
#         outY[i] = Gy ⋅ @view(matrix[view_range])
#     end
#     return outX, outY
# end
function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
    Gy = _avg_op(T) * _diff_op(T)'
    Gx = _diff_op(T) * _avg_op(T)'
    new_size = size(matrix) .- 2
    outX = _convolve_tullio(matrix, Gx)
	outY = _convolve_tullio(matrix, Gy)
    return outX, outY
end

# ╔═╡ 90ff1023-103a-4342-b521-e229157001fc
function discretize_gradient_direction(θ)
    if -π / 8 ≤ θ < π / 8
        return 0
    elseif π / 8 ≤ θ < 3 * π / 8
        return π / 4
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return π / 2
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return 3 * π / 4
    elseif 7 * π / 8 ≤ θ
        return π
    elseif -3 * π / 8 ≤ θ < -π / 8
        return -π / 4
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return -π / 2
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return -3π / 4
    elseif θ < -7 * π / 8
        return -π
    end
end

# ╔═╡ 5c0be95f-3c4a-4062-afeb-3c1681cae549
function gradient_grid_direction(θ)
	@assert -π ≤ θ ≤ π
    if -π / 8 ≤ θ < π / 8
        return CartesianIndex(1, 0)
    elseif π / 8 ≤ θ < 3 * π / 8
        return CartesianIndex(1, 1)
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return CartesianIndex(0, 1)
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return CartesianIndex(-1, 1)
    elseif 7 * π / 8 ≤ θ
        return CartesianIndex(-1, 0)
    elseif -3 * π / 8 ≤ θ < -π / 8
        return CartesianIndex(1, -1)
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return CartesianIndex(0, -1)
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return CartesianIndex(-1, -1)
    elseif θ < -7 * π / 8
        return CartesianIndex(-1, 0)
    end
end

# ╔═╡ 88889293-9afc-4540-a2b9-f30afb62b1de
function mark_edge_candidate(dP2_view, Gx, Gy)
    grid_theta = gradient_grid_direction(atan(Gy, Gx))
    idx = CartesianIndex(2, 2)
    return dP2_view[idx+grid_theta] < dP2_view[idx] &&
           dP2_view[idx-grid_theta] < dP2_view[idx]
end

# ╔═╡ bc0c6a41-adc8-4d18-9574-645704f54b72
md"""
---
"""

# ╔═╡ 4a5086bc-5c36-4e71-9d3a-8f77f48a90f9
md"""
The implemented shock sensor has some issues (we need to set ``TOL=0.7``), but produces a reasonable result on this test data. We can clearly see the shock front that forms in front of the body, and the other "shocks" are likely numerical effects from the terribly-done rasterization of the body.

``TOL`` is much more relaxed here than in the original paper, but the sensor was originally tested on data generated using a MUSCL implementation.
"""

# ╔═╡ 747f0b67-546e-4222-abc8-2007daa3f658
@bind rh_err Slider(0.0:0.01:2.5, show_value = true, default = 0.7)

# ╔═╡ 2cb33e16-d735-4e60-82a5-aa22da0288fb
@bind smoothness_err Slider(0.000:0.005:0.2, show_value = true, default = 0.1)

# ╔═╡ 9ac61e80-7d6a-40e8-8254-ee306f8248c3
let
	pf = map(p -> isnothing(p) ? 0.0 : p, pressure_field(sim_with_ad, FINAL_TSTEP_IDX, DRY_AIR))
	a, b = convolve_sobel(pf)
	x, y = cell_centers(sim_with_ad)
	th = discretize_gradient_direction.(atan.(b, a)).*(4/π)
	dP2 = a.^2 .+ b.^2
	p1 = heatmap(x[2:end-1], y[2:end-1], th', aspect_ratio=:equal, ylims=(0., 2.), xlims=(-1.5, 0.5))
	edge_candidates = Array{Bool,2}(undef, size(dP2) .- 2)
    window_size = CartesianIndex(2, 2)
    for i ∈ eachindex(IndexCartesian(), edge_candidates)
        edge_candidates[i] = mark_edge_candidate(
            @view(dP2[i:i+window_size]),
            a[i+CartesianIndex(1, 1)],
            b[i+CartesianIndex(1, 1)],
        )
    end
	p2 = heatmap(x[3:end-2], y[3:end-2], edge_candidates', aspect_ratio=:equal, ylims=(0., 2.), xlims=(-1.5, 0.5), title="All Edge Candidates")
	#plot(p1, p2)
	p2
end

# ╔═╡ 9c601619-aaf1-4f3f-b2e2-10422d8ac640
function shock_cells(sim, n, shock_field)
    sort(
        reduce(
            vcat,
            filter(!isnothing, map(enumerate(eachcol(shock_field))) do (j, col)
                i = findfirst(col)
                isnothing(i) && return nothing
                id = @view(sim.cell_ids[2:end-1, 2:end-1])[i, j]
                return sim.cells[n][id]
            end),
        );
        lt = (a, b) -> a.center[2] < b.center[2],
    )
end

# ╔═╡ e0a934d6-3962-46d5-b172-fb970a537cc0
function shock_points(sim::CellBasedEulerSim{T,C}, n, shock_field) where {T,C}
    sp = shock_cells(sim, n, shock_field)
    res = Matrix{T}(undef, (length(sp), 2))
    for i ∈ eachindex(sp)
        res[i, :] = sp[i].center
    end
    #sort!(sp; lt=(a, b) -> a[2] < b[2])
    return res
end

# ╔═╡ 92044a9f-2078-48d1-8181-34be87b03c4c
md"""
## Choosing a New Mesh

We know that there is a relationship between the shape and position of the bow shock and the simulation parameters. We will take advantage of the fact that the flow is in a steady state (``\partial_t u(x) = 0``) to find an implicit relationship between the position of points on the shock and the flow data.

We will take advantage of the fact that, for any control volume ``C``:
```math
\int_{C}\frac{\partial u}{\partial t}\,dx = 0 = \oint_{\partial C}\mathcal F(u)\cdot\hat n\,dS
```
and draw a set of control volumes that have corners on the shock front.
"""

# ╔═╡ 93043797-66d1-44c3-b8bb-e17deac70cfa
md"""
If we take a set of points along the ``y``-axis, we can create cells that have vertices on any of the:
 - computational domain boundaries
 - bow shock
 - blunt body

In fact, we only need three cells: one in front of the shock, one immediately behind the shock, and its eastern neighbor.

We would also like to avoid numerical noise that can develop in the gradient ``\nabla_x u(t, x)`` near the shock. To do this, we will move the points on the _right_ of the shock some distance away from the extracted shock boundary, similar to the approximation done in Hüser's thesis.
"""

# ╔═╡ 766b440b-0001-4037-8959-c0b7f04d999e
const num_coarse_cells_pos_y = 16;

# ╔═╡ 7468fbf2-aa57-4505-934c-baa4dcb646fc
const cell_width_at_shock = 0.075;

# ╔═╡ af0732ba-b280-4dd0-8bed-2170a2a4d6f8
const minimum_cell_width_scaling = 0.75;

# ╔═╡ 76cb5757-140e-429f-8190-2e5ae888d3a4
const APPLY_SHOCK_BAND_CORRECTION = true;

# ╔═╡ 4d202323-e1a9-4b24-b98e-7d17a6cc144f
struct CoarseQuadCell{T,NS,NTAN}
    id::Int
    pts::SVector{8,T}
    u::SVector{4,T}
    u̇::SMatrix{4,NS,T,NTAN}
    du_dpts::SMatrix{4,8,T,32}
end

# ╔═╡ 2041d463-5382-4c38-bf52-23d22820ac59
function make_polys(y1, y2, xs1, xs2)
    L1 = length(xs1)
    L2 = length(xs2)
    L = min(L1, L2)
    polys = [
        SClosedPolygon(
            Point(xs1[i+1], y1),
            Point(xs1[i], y1),
            Point(xs2[i], y2),
            Point(xs2[i+1], y2),
        ) for i ∈ 1:(L-1)
    ]
    if L1 > L2
        # more points below
        @reset polys[end].pts[1][1] = xs1[end]
    elseif L2 > L1
        # more points above
        @reset polys[end].pts[4][1] = xs2[end]
    end
    return polys
end

# ╔═╡ 9a239bb8-9140-43ad-9b1b-b85c5e78b40a
sanity_check = Dict([
	1=>CoarseQuadCell(
		1,
		SVector(
			-1.1203125,0.0,
			-1.1953125,0.0,
			-1.1921875,0.1248046875,
			-1.1171875,0.1248046875,
		),
		zero(SVector{4,Float64}),
        zero(SMatrix{4,3,Float64,12}),
        zero(SMatrix{4,8,Float64,32}),
	)
])

# ╔═╡ e98df040-ac22-4184-95dd-a8635ab72af6
md"""
---
"""

# ╔═╡ 95947312-342f-44b3-90ca-bd8ad8204e18
begin
    function cell_boundary_polygon(cell::Euler2D.QuadCell)
        c = cell.center
        dx, dy = cell.extent / 2
        return SClosedPolygon(
            c + SVector(dx, -dy),
            c + SVector(-dx, -dy),
            c + SVector(-dx, dy),
            c + SVector(dx, dy),
        )
    end
    function cell_boundary_polygon(cell::CoarseQuadCell)
        return cell.pts
    end
end

# ╔═╡ d790b1ba-180e-4de2-bed2-56155a8b8dff
tcell1 = CoarseQuadCell(1, @SVector([-1.3, 1.5, -1.4, 1.5, -1.4, 1.75, -1.3, 1.75]), zero(SVector{4, Float64}), zero(SMatrix{4, 3, Float64, 12}), zero(SMatrix{4, 8, Float64, 32}))

# ╔═╡ a1dc855f-0b24-4373-ba00-946719d38d95
md"""
---
"""

# ╔═╡ 6aaf12a0-c2d9-48ab-9e13-94039cf95258
md"""
Numerical viscosity or numerical dissipation may have affected things more strongly than we would like. Brief inspection reveals that the relative error immediately in _front_ of the shock is around 10%.
"""

# ╔═╡ fd73e7b8-5887-4fee-9d8c-c8df45e54d11
# we need this to match the syntax for ϕ_hll
# but we might want to choose other flux functions later...
# for example, an exact Riemann solver?
begin
    ϕ = Euler2D.ϕ_hll
end

# ╔═╡ da01e1fa-876e-4bdd-8f2e-e8df524fe7cb
const x_select = hcat(
    	SVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    	SVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
	);

# ╔═╡ cdbc037e-d542-47be-b78e-b396c5c3d8df
md"""
## Figures
"""

# ╔═╡ 4cb96329-f3f4-4874-9ec3-63665abc8d22
ε_scale = 2.5/100

# ╔═╡ 3ee5edb5-e648-489f-9c17-10c3912e0fcd
test_perturbations = diagm(ε_scale.*ambient_primitives)

# ╔═╡ 9e9cc14d-bca7-4bea-832e-f4cecb556e90
md"""
Perturbing the mach number yields the expected results; perturbing the density / incoming flow temperature yields nothing! Wow!
"""

# ╔═╡ 1f174634-812d-4704-8e82-36935e8f0cb5
md"""
## Debug GFX and Information
"""

# ╔═╡ 2c0133c2-35d5-4cfd-b19d-dfb08712479f
@bind poly_skewx Slider(range(0, 0.125; length = 101), show_value = true)

# ╔═╡ 0ef2267a-2116-469c-9b50-9b85be4ccc45
@bind poly_skewy Slider(range(0, 0.125; length = 101), show_value = true)

# ╔═╡ 3750e0d5-5ca3-447a-bbf5-add15a4a4e9b
@bind test_cx Slider(range(0.1, 0.9; length = 101), show_value = true)

# ╔═╡ a1d7888c-34a2-425b-a8e4-2329b6400901
@bind test_cy Slider(range(0.1, 0.9; length = 101), show_value = true)

# ╔═╡ 68650137-22f6-4c1a-814b-f9b9c81e20ca
let
    pgram = SClosedPolygon(
        Point(0.25 + poly_skewx, 0.25 - poly_skewy),
        Point(0.25 - poly_skewx, 0.75 - poly_skewy),
        Point(0.75 - poly_skewx, 0.75 + poly_skewy),
        Point(0.75 + poly_skewx, 0.25 + poly_skewy),
    )
    testq = SClosedPolygon(
        Point(test_cx - 0.05, test_cy - 0.05),
        Point(test_cx - 0.05, test_cy + 0.05),
        Point(test_cx + 0.05, test_cy + 0.05),
        Point(test_cx + 0.05, test_cy - 0.05),
    )
    data = reduce(hcat, edge_starts(pgram))
    p = plot(
        data[1, :],
        data[2, :];
        seriestype = :shape,
        legend = false,
        xlims = (-0.5, 1.5),
        ylims = (-0.5, 1.5),
        aspect_ratio = :equal,
    )
    test_color = are_polygons_intersecting(pgram, testq) ? :red : :green
    data = reduce(hcat, edge_starts(testq))
    plot!(
        p,
        data[1, :],
        data[2, :];
        seriestype = :shape,
        legend = false,
        color = test_color,
        fillalpha = 0.5,
    )
    ns = outward_edge_normals(testq)
    lines = [SVector(test_cx * one(eltype(n)), test_cy * one(eltype(n)), n...) for n in ns]
    imgs = [_poly_image(ell, pgram) for ell in lines]
    for (a, b) in zip(lines, imgs)
        pt1 = point_on(a) + direction_of(a) * b[1]
        pt2 = point_on(a) + direction_of(a) * b[2]
        c = hcat(pt1, pt2)
        plot!(p, c[1, :], c[2, :]; color = :black, lw = 2)
    end
    imgs = [_poly_image(ell, testq) for ell in lines]
    for (a, b) in zip(lines, imgs)
        pt1 = point_on(a) + direction_of(a) * b[1]
        pt2 = point_on(a) + direction_of(a) * b[2]
        c = hcat(pt1, pt2)
        plot!(p, c[1, :], c[2, :]; color = :red, lw = 1, ls = :dash)
    end

    ns = outward_edge_normals(pgram)
    lines = [SVector(0.5 * one(eltype(n)), 0.5 * one(eltype(n)), n...) for n in ns]
    imgs = [_poly_image(ell, testq) for ell in lines]
    for (a, b) in zip(lines, imgs)
        pt1 = point_on(a) + direction_of(a) * b[1]
        pt2 = point_on(a) + direction_of(a) * b[2]
        c = hcat(pt1, pt2)
        plot!(p, c[1, :], c[2, :]; color = :purple, lw = 2)
    end
    imgs = [_poly_image(ell, pgram) for ell in lines]
    for (a, b) in zip(lines, imgs)
        pt1 = point_on(a) + direction_of(a) * b[1]
        pt2 = point_on(a) + direction_of(a) * b[2]
        c = hcat(pt1, pt2)
        plot!(p, c[1, :], c[2, :]; color = :green, lw = 1, ls = :dash)
    end
    p
end

# ╔═╡ 501536eb-2f3a-4dfd-b126-0c7999782802
function _translate_state_velocity(u, v_coords)
	ρv =select_middle(u)	
	v= ρv/u[1]
	ρke = ρv⋅v/2
	ρe = u[4] - ρke
	v★ = v - v_coords
	ρv★ = u[1]*v★
	ρke★ = ρv★ ⋅ v★ / 2
	ρE★ = ρe + ρke★
	return SVector(u[1], ρv★..., ρE★)
end

# ╔═╡ 9c005d9f-46bb-4b75-8233-f4e1f2cf0663
ambient_u

# ╔═╡ cac50bdc-55f8-4073-9048-c83deda836b7
_translate_state_velocity(ambient_u, SVector(1.0, 10.0))

# ╔═╡ 8f3d93ad-06fd-43bd-bf8f-004b2c80a5f2
Euler2D._pressure(_translate_state_velocity(ambient_u, SVector(1.0, 1.0)), DRY_AIR)

# ╔═╡ 70d0aac2-6813-4d02-a363-0951a3d9592c
Euler2D._pressure(ambient_u, DRY_AIR)

# ╔═╡ 284ad6c0-b211-48aa-aa4f-d354c0d5520b
md"""
### Helper Methods
"""

# ╔═╡ 61945d45-03f4-4401-8b18-3d11420047d0
minimum_cell_size(sim) = tmapreduce((a, b,)->min.(a, b), sim.cell_ids; init=(Inf, Inf)) do id
	id == 0 && return (Inf, Inf)
	return nth_step(sim, 1)[2][id].extent
end

# ╔═╡ c1a81ef6-5e0f-4ad5-8e73-e9e7f09cefa6
function dimensionless_speed_of_sound(
    u_star::SVector{N,T},
    gas::CaloricallyPerfectGas,
) where {N,T}
    P_star = Euler2D._pressure(u_star, gas)
    return sqrt(gas.γ * (P_star / u_star[1]))
end

# ╔═╡ e55363f4-5d1d-4837-a30f-80b0b9ae7a8e
function dimensionless_mach_number(
    u_star::SVector{N,T},
    gas::CaloricallyPerfectGas,
) where {N,T}
    a = dimensionless_speed_of_sound(u_star, gas)
    ρa = u_star[1] * a
    return Euler2D.select_middle(u_star) ./ ρa
end

# ╔═╡ 6da05b47-9763-4d0c-99cc-c945630c770d
#assumes stationary shock "edge"
function rh_error_lab_frame(cell_front, cell_behind, θ, gas)
    m1 = dimensionless_mach_number(cell_front.u, gas)
    m2 = dimensionless_mach_number(cell_behind.u, gas)
    dir = sincos(θ)
    n̂ = SVector(dir[2], dir[1])
    m_ratio = ShockwaveProperties.shock_normal_mach_ratio(m1, n̂, gas)
    m1_norm = abs(m1 ⋅ n̂)
    m2_norm_rh = m1_norm * m_ratio
    m2_norm_sim = abs(m2 ⋅ n̂)
    return (abs(m2_norm_rh - m2_norm_sim) / m2_norm_sim, abs(m1_norm / m2_norm_sim - 1))
end

# ╔═╡ 351d4e18-4c95-428e-a008-5128f547c66d
function find_shock_in_timestep(
    sim::CellBasedEulerSim{T,C},
    t,
    gas;
    rh_rel_error_max = 0.5,
    continuous_variation_thold = 0.01,
) where {T,C}
    # TODO really gotta figure out how to deal with nothings or missings in this matrix
    pfield = map(p -> isnothing(p) ? zero(T) : p, pressure_field(sim, t, gas))
    Gx, Gy = convolve_sobel(pad_pfield_from_bcs(pfield))
    dP2 = Gx .^ 2 + Gy .^ 2
    edge_candidates = Array{Bool,2}(undef, size(dP2) .- 2)
    window_size = CartesianIndex(2, 2)
    for i ∈ eachindex(IndexCartesian(), edge_candidates)
        edge_candidates[i] = mark_edge_candidate(
            @view(dP2[i:i+window_size]),
            Gx[i+CartesianIndex(1, 1)],
            Gy[i+CartesianIndex(1, 1)],
        )
    end
    @info "Number of candidates..." n_candidates = sum(edge_candidates)
    Gx_overlay = @view(Gx[2:end-1, 2:end-1])
    Gy_overlay = @view(Gy[2:end-1, 2:end-1])
    id_overlay = @view(sim.cell_ids[2:end-1, 2:end-1])
    num_except = 0
	except_infos = Dict{CartesianIndex{2}, Int}()
    num_reject_too_smooth = 0
    num_reject_rh_fail = 0
    for j ∈ eachindex(IndexCartesian(), edge_candidates, Gx_overlay, Gy_overlay, id_overlay)
        i = j + CartesianIndex(1, 1)
        if id_overlay[j] > 0 && edge_candidates[j]
            θ = atan(Gy_overlay[j], Gx_overlay[j])
            θ_disc = discretize_gradient_direction(θ)
            θ_grid = gradient_grid_direction(θ_disc)
            # gradient points in direction of steepest increase...
            # cell in "front" of shock should be opposite the gradient?
            id_front = sim.cell_ids[i+θ_grid]
            id_back = sim.cell_ids[i-θ_grid]
            if id_front == 0 || id_back == 0
                edge_candidates[j] = false
                continue
            end

            cell_front = sim.cells[t][id_front]
            cell_back = sim.cells[t][id_back]
            try
                rh_err, sim_err = rh_error_lab_frame(cell_front, cell_back, θ_disc, gas)
                if rh_err > rh_rel_error_max
                    # discard edge candidate
                    num_reject_rh_fail += 1
                    edge_candidates[j] = false
                elseif sim_err < continuous_variation_thold
                    num_reject_too_smooth += 1
                    edge_candidates[j] = false
                end
            catch de
                if de isa DomainError
                    #@warn "Cell shock comparison caused error" typ=typeof(de) j θ_grid
					except_infos[θ_grid] = get(except_infos, θ_grid, 0) + 1
                    edge_candidates[j] = true
                    num_except += 1
                else
                    rethrow()
                end
            end
        else
            edge_candidates[j] = false
        end
    end
    @info "Number of candidates after RH condition thresholding..." n_candidates =
        sum(edge_candidates) num_except num_reject_rh_fail num_reject_too_smooth
	@info "Exceptions triggered on following:" except_infos
    return edge_candidates
end

# ╔═╡ 4b036b02-1089-4fa8-bd3a-95659c9293cd
sf = find_shock_in_timestep(
    sim_with_ad,
    FINAL_TSTEP_IDX,
    DRY_AIR;
    rh_rel_error_max = rh_err,
    continuous_variation_thold = smoothness_err,
);

# ╔═╡ 24da34ca-04cd-40ae-ac12-c342824fa26e
let
    data = map(sf, @view sim_with_ad.cell_ids[2:end-1, 2:end-1]) do v1, v2
        if v2 == 0
            missing
        else
            v1
        end
    end
    p = heatmap(
        cell_centers(sim_with_ad, 1)[2:end-1],
        cell_centers(sim_with_ad, 2)[2:end-1],
        data';
        cbar = false,
        aspect_ratio = :equal,
        xlims = (-1.5, 0.),
        ylims = (0., 2.0),
        xlabel = L"x",
        ylabel = L"y",
        size = (700, 500),
        dpi = 1000,
    )
    #savefig(p, "../gfx/shock_sensor_07_01.pdf")
    p
end

# ╔═╡ 62ebd91b-8980-4dc5-b61b-ba6a21ac357d
all_shock_points = shock_points(sim_with_ad, FINAL_TSTEP_IDX, sf);

# ╔═╡ be8ba02d-0d31-4720-9e39-595b549814cc
sp_interp = linear_interpolation(all_shock_points[:, 2], all_shock_points[:, 1]);

# ╔═╡ 604d3736-15b0-44f6-99eb-49ba7186762b
dimensionless_speed_of_sound(_translate_state_velocity(ambient_u, SVector(1.0, -10.0)), DRY_AIR)

# ╔═╡ 6f303343-1f76-46f9-80e8-2dd4ae1b5427
function fdiff_eps(arg::T) where {T<:Real}
	cbrt_eps = cbrt(eps(T))
	h = 2^(round(log2((1+abs(arg))*cbrt_eps)))
	return h
end

# ╔═╡ 436a3325-51bb-4718-aa08-f835e7f98ddd
md"""
---
"""

# ╔═╡ 729ebc48-bba1-4858-8369-fcee9f133ee0
function is_cell_contained_by(cell::Union{Euler2D.QuadCell,CoarseQuadCell}, closed_poly)
    return all(edge_starts(cell_boundary_polygon(cell))) do p
        PlanePolygons.point_inside_strict(closed_poly, p)
    end
end

# ╔═╡ 5cffaaf5-9a5e-4839-a056-30e238308c51
function is_cell_overlapping(cell::Union{Euler2D.QuadCell,CoarseQuadCell}, closed_poly)
    contained = is_cell_contained_by(cell, closed_poly)
    return (
        !contained && are_polygons_intersecting(cell_boundary_polygon(cell), closed_poly)
    )
end

# ╔═╡ f252b8d0-f067-468b-beb3-ff6ecaeca722
function all_cells_contained_by(poly, sim::CellBasedEulerSim)
	Δx, Δy = minimum_cell_size(sim)
	window_x = extrema(pt->pt[1], edge_starts(poly)) .+ (-Δx, Δx)
    window_y = extrema(pt->pt[2], edge_starts(poly)) .+ (-Δy, Δy)
    _, cells = nth_step(sim, 1)
	in_window = Iterators.filter(sim.cell_ids) do id
		id == 0 && return false
		window_x[1] ≤ cells[id].center[1] ≤ window_x[2] || return false
		window_y[1] ≤ cells[id].center[2] ≤ window_y[2] || return false
		return true
	end
	return collect(Iterators.filter(in_window) do id
        return is_cell_contained_by(cells[id], poly)
    end)
end

# ╔═╡ 571b1ee7-bb07-4b30-9870-fbd18349a2ef
function all_cells_overlapping(poly, sim::CellBasedEulerSim)
	Δx, Δy = minimum_cell_size(sim)
	window_x = extrema(pt->pt[1], edge_starts(poly)) .+ (-Δx, Δx)
    window_y = extrema(pt->pt[2], edge_starts(poly)) .+ (-Δy, Δy)
    _, cells = nth_step(sim, 1)
	in_window = Iterators.filter(sim.cell_ids) do id
		id == 0 && return false
		(
			window_x[1] ≤ cells[id].center[1] ≤ window_x[2] && 
			window_y[1] ≤ cells[id].center[2] ≤ window_y[2] 
		) || return false
		return is_cell_overlapping(cells[id], poly)
	end
	return collect(in_window)
end

# ╔═╡ 80cde447-282a-41e5-812f-8eac044b0c15
function overlapping_cell_area(cell1, cell2)
    isect = poly_intersection(cell_boundary_polygon(cell1), cell_boundary_polygon(cell2))
    return poly_area(isect)
end

# ╔═╡ d87e0bb8-317e-4d48-8008-a7071c74ab31
# gets the jacobian of the intersection area w.r.t. the first argument
function intersection_area_jacobian(flat_poly1, poly2)
    grad1 = zero(flat_poly1)
    for i in eachindex(flat_poly1)
		h = fdiff_eps(flat_poly1[i])
        in1 = Accessors.@set flat_poly1[i] += h
        in2 = Accessors.@set flat_poly1[i] -= h
        out1 = poly_area(poly_intersection(in1, poly2))
        out2 = poly_area(poly_intersection(in2, poly2))
        @reset grad1[i] = (out1 - out2) / (2 * h)
    end
    return grad1
end

# ╔═╡ 244838df-ae1f-40a3-a94e-50222969a3c0
PlanePolygons._POINT_DOES_NOT_EXIST(Float64)

# ╔═╡ 95c1ba4e-32c0-4b4c-b30b-766b5fd9f622
md"""
---
"""

# ╔═╡ d44322b1-c67f-4ee8-b168-abac75fb42a1
const num_coarse_cells = begin
    ypts2 = begin
        yr = range(; start = 0.0, stop = 2.0, length = num_coarse_cells_pos_y + 1)
        s = step(yr)
        y = collect(-2*s:s:2.0)
        y
    end
    num_coarse_cells_pos_y + 2
end

# ╔═╡ 3a8cd7e2-fae9-4e70-8c92-004b17352506
md"""
## Solving for ``\xi``

Each of the points on the shock has been used to define new cells ``P_i``. For each of the original cells, as well as the new cells, we know that:
```math
\oint_{\partial P_i} \tilde F(\bar u_i)\cdot\hat n\,ds = 0
```

There are $(num_coarse_cells) cells directly behind the shock. For these cells, we know the following:
- Its cell-average value ``u`` and its tangent ``\dot{u}``
- Its bounding polygon, made up of points ``p_k``
- ``\partial_{p_k}u``

Since the cell is a quadrilateral, we can rewrite the boundary integral above into the following sum:
```math
	\tilde F(u_i)\hat n_{i,S}L_{i, S} + \tilde F(u_i)\hat n_{i,E}L_{i, E} + \tilde F(u_i)\hat n_{i,N}L_{i, N} + \tilde F(u_i)\hat n_{i,W}L_{i, W} = 0
```

This sum defines an implicit relationship between the simulation paramters ``p``, which are used to determine ``u(t, x)``, and a set of points on the shock $x_s$, which enter the boundary integral via the side lengths of the cell. If we label the boundary integral ``\mathcal G(x_s, p)``, we can use the implicit selection theorem to find the Jacobian of the implicit relationship between ``x_s`` and ``p``.

```math
\begin{aligned}
0 &= \mathcal {G}(x_s, p)\\
g(p) &= x_s \\
\frac{\partial g(p)}{\partial p_j} &= \left[\frac{\partial G_i(x_s, p)}{\partial x_k}\right]^{+}\frac{\partial G_i(x_s, p)}{\partial p_j}
\end{aligned}
```

Since ``u``, ``x_s``, and ``\dot u`` are known, we can compute the dependence of the shock position on the initial parameters via:
```math
\xi = \frac{\partial g(p)}{\partial p_j}
```
"""

# ╔═╡ 34c24942-a195-4ce4-98fb-bad2f76d3a2c
const smallest_cell_width = minimum_cell_size(sim_with_ad)[1]

# ╔═╡ d19fff76-e645-4d9d-9989-50019b6356ad
function _point_collinear_between(p, p0, p1)
    ell = Line(p0, p1 - p0)
    return (
        is_other_point_on_line(ell, p) && 0 ≤ PlanePolygons.projected_component(ell, p) ≤ 1
    )
end

# ╔═╡ 0e0a049b-e2c3-4fe9-8fb8-186cdeb60485
function are_coarse_neighbors(c1, c2)
    if c1.id == c2.id
        return (false, nothing)
    end
    e1 = cell_boundary_polygon(c1)
    e2 = cell_boundary_polygon(c2)
    # hardcoded in this order (I think)
    dirs = (:south, :west, :north, :east)
    #clockwise around c1
    for (dir, p1, p2) ∈ zip(dirs, edge_starts(e1), edge_ends(e1))
        for (q1, q2) ∈ zip(edge_ends(e2), edge_starts(e2))
            if (
                (
                    _point_collinear_between(q1, p1, p2) &&
                    _point_collinear_between(q2, p1, p2)
                ) || (
                    _point_collinear_between(p1, q1, q2) &&
                    _point_collinear_between(p2, q1, q2)
                )
            )
                return (true, (dir, p1, p2))
            end
        end
    end
    return (false, nothing)
end

# ╔═╡ 2f088a0c-165e-47f9-aaeb-6e4ab31c9d26
begin
    slope_above =
        (all_shock_points[end, 2] - all_shock_points[end-2, 2]) /
        (all_shock_points[end, 1] - all_shock_points[end-2, 1])

    slope_below =
        (all_shock_points[3, 2] - all_shock_points[1, 2]) /
        (all_shock_points[3, 1] - all_shock_points[1, 1])

    function x_shock(y)
		if y < 0
			return x_shock(-y)
		end
        if all_shock_points[1, 2] < y < all_shock_points[end, 2]
            return sp_interp(y)
        elseif all_shock_points[1, 2] ≥ y
            return (y - all_shock_points[1, 2]) / slope_below + all_shock_points[1, 1]
        else
            return (y - all_shock_points[end, 2]) / slope_above + all_shock_points[end, 1]
        end
    end
end

# ╔═╡ ae7bca74-2a14-4a62-b6e9-669260fa4011
let
	f(y) = BilligShockParametrization.shock_front(y, ambient_primitives[2], 0.75)[1]
	y = 0.:0.01:2.0
	y2 = 0.:0.05:2.0
	x = f.(y)
	p = plot(x, y, label="Billig")
	plot!(p, x_shock.(y2), y2, label="Extracted", title="Shock Front (simulated) / Billig (analytic) Shock Front")
end

# ╔═╡ 79f2ec98-efc4-4215-968a-9c3c2cb52004
x_shock_adjusted(y, α) = x_shock(y) + α * smallest_cell_width

# ╔═╡ cd312803-3819-4451-887b-ce2b53bb6e1b
x_right(y) = x_shock(y) + cell_width_at_shock

# ╔═╡ 881e3b6e-ecdc-41b1-b21a-69de7e42af36
x_midleft(y) = x_shock(y) - cell_width_at_shock

# ╔═╡ ac412980-1013-450f-bb23-0dc7c2b3f199
function x_body(y)
    if y > 0.75 || y < -0.75
        return 0.0
    else
        return -sqrt(0.75^2 - y^2)
    end
end

# ╔═╡ 0d6ae7cf-edae-48f6-a257-6223563b7c76
function points_row(y)
    return vcat(
        x_midleft(y),
        range(x_shock(y), x_body(y); step = cell_width_at_shock)[1:3],
    )
end

# ╔═╡ 54ed2abb-81bb-416c-b7be-1125e41622f5
all_polys = mapreduce(vcat, 1:(num_coarse_cells-1)) do i
    x1 = points_row(ypts2[i])
    x2 = points_row(ypts2[i+1])
    polys = make_polys(ypts2[i], ypts2[i+1], x1, x2)
    return polys
end

# ╔═╡ f30619a3-5344-4e81-a4b5-6a11100cd056
empty_coarse = Dict([
    (id + 1) => CoarseQuadCell(
        id,
        PlanePolygons._flatten(poly),
        zero(SVector{4,Float64}),
        zero(SMatrix{4,3,Float64,12}),
        zero(SMatrix{4,8,Float64,32}),
    ) for (id, poly) ∈ enumerate(all_polys)
])

# ╔═╡ 48148c30-486b-4952-9c5f-aea722ff74cf
@bind n_cell Slider(9:length(empty_coarse); show_value=true, default=9)

# ╔═╡ 8f36cc9d-c2c5-4bea-9dc7-e5412a2960f9
let
    n = n_cell﻿
    poly = cell_boundary_polygon(empty_coarse[n])
    pts = mapreduce(pt -> pt', vcat, edge_starts(poly))
    xlims = extrema(pts[:, 1]) .+ (-0.05, 0.05)
    ylims = extrema(pts[:, 2]) .+ (-0.05, 0.05)
    # pts = vcat(pts, first(edge_starts(poly))')
    p = plot(
        pts[:, 1],
        pts[:, 2];
        lw = 0.5,
        fill = true,
        ylims = ylims,
        xlims = xlims,
        dpi = 1000,
        fillalpha = 0.5,
        seriestype = :shape,
        label = "Cell $n",
    )
    foreach(all_cells_overlapping(empty_coarse[n].pts, sim_with_ad)) do id
        _, c = nth_step(sim_with_ad, FINAL_TSTEP_IDX)
        data = mapreduce(pt -> pt', vcat, edge_starts(cell_boundary_polygon(c[id])))
        plot!(
            p,﻿
            data[:, 1],
            data[:, 2];
            lw = 0.5,
            fill = true,
            fillcolor = :red,
            fillalpha = 0.5,
            seriestype = :shape,
            label = false,
        )
    end
    for ell ∈ edge_lines(poly)
        x1 = point_on(ell)[1] - 10 * direction_of(ell)[1]
        x2 = point_on(ell)[1] + 10 * direction_of(ell)[1]
        y1 = point_on(ell)[2] - 10 * direction_of(ell)[2]
        y2 = point_on(ell)[2] + 10 * direction_of(ell)[2]
        plot!(p, [x1, x2], [y1, y2]; color = :black, label = false, ls = :dash)
    end
    xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
    yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
    plot!(p, xc, yc; color = :black, label=false)
    data = mapreduce(vcat, all_cells_contained_by(empty_coarse[n].pts, sim_with_ad)) do id
        _, c = nth_step(sim_with_ad, FINAL_TSTEP_IDX)
        return Vector(c[id].center)'
    end
    scatter!(p, data[:, 1], data[:, 2]; marker = :x, ms = 2, label = false)
	#savefig(p, "../gfx/debug-cell.pdf")
    p
end

# ╔═╡ 37eb63be-507a-475f-a6f6-8606917b8561
# grab AD via Mooncake.jl
# I think it works well.
begin
    diff_backend = AutoMooncake(; config = nothing)
    fdiff_backend = AutoForwardDiff()
    poly_area_prep = prepare_gradient(poly_area, diff_backend, empty_coarse[2].pts)
end;

# ╔═╡ a968296a-43c1-48f3-b4b8-0d81cb162b7b
function intersection_point_jacobian(point, poly1, poly2)
	N = num_vertices(poly1)
	J = zeros(eltype(point), (2, 2*N))
	foreach(enumerate(zip(edge_starts(poly1), edge_ends(poly1)))) do (i, (p1, p2))
		if is_other_point_on_line(Line(p1, p2-p1), point)
			for ell2 ∈ Iterators.filter(ℓ->is_other_point_on_line(ℓ, point), edge_lines(poly2))
				jac = DifferentiationInterface.jacobian(fdiff_backend, vcat(p1, p2)) do vals
					q1 = vals[SVector(1,2)]
					q2 = vals[SVector(3,4)]
					return line_intersect(Line(q1, q2-q1), ell2)
				end
				j = ((i+1) % N) + 1
				J[2*i-1:2*i] = jac[SVector(1,2)]
				J[2*j-1:2*j] = jac[SVector(3,4)]
			end
		end
	end
	return J
end

# ╔═╡ dea032e2-bf23-42da-8dac-d3368c2bdec6
let
    xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
    yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
    p = plot(
        xc,
        yc;
        aspect_ratio = :equal,
        xlims = (-1.5, 0.0),
        ylims = (0.0, 2.0),
        label = "Blunt Body",
        fill = true,
        dpi = 1000,
        size = (1150, 1000),
        ls = :dash,
        lw = 4,
        fillalpha = 0.5,
		legendfontsize=18,
		tickfontsize=16,
		
    )
	spys = range(-0.25, 2.0; length = 20)
    spxs = x_shock.(spys)
	plot!(
		p,
        spxs,
        spys;
        label = "Shock Front (Interpolated)",
        lw = 4,
    )
    pts = [Point(x, y) for y ∈ ypts2 for x ∈ points_row(y)]
    scatter!(p, [pt[1] for pt ∈ pts], [pt[2] for pt ∈ pts]; marker = :x, label="New Cell Vertices", ms=8)

    for i = 1:num_coarse_cells
        x1 = points_row(ypts2[i])
        x2 = points_row(ypts2[i+1])
        polys = make_polys(ypts2[i], ypts2[i+1], x1, x2)
        for poly ∈ polys
            pl = reduce(hcat, edge_starts(poly))
            plot!(
                p,
                @view(pl[1, :]),
                @view(pl[2, :]);
                lw = 0.5,
                fill = true,
                fillalpha = 1.,
				fillstyle = :/,
                label = false,
                color = :green,
                seriestype = :shape,
            )
        end
    end
	
    savefig(p, "../gfx/new_cells_N$(num_coarse_cells_pos_y).pdf")
    p
end

# ╔═╡ 602b8184-90d5-4d19-906a-b41f278d1761
md"""
---
"""

# ╔═╡ fd9b689f-275c-4c91-9b6c-4e63c68d6ab2
struct DualNodeKind{K} end # trait struct for dispatch

# ╔═╡ ead8c1a5-9f4e-4d92-b4ca-1650ad34bdca
const DUAL_NODE_TYPE = Tuple{
    DualNodeKind{S},
    Union{Nothing,SVector{4,Float64},CoarseQuadCell{Float64,3,12}},
} where {S}

# ╔═╡ e650d2b0-366c-4595-8fc3-69db62c14579
function produce_dual_graph_from_coarse_cells(cells)
	g = MetaGraph(DiGraph(), Int, DUAL_NODE_TYPE, Tuple{Symbol,Point{Float64},Point{Float64}})
    g[1] = (DualNodeKind{:boundary_ambient}(), ambient_u)
    for (k, v) ∈ cells
        g[k] = (DualNodeKind{:cell}(), v)
    end
    phantom_idx = 10000 * nv(g) + 1
    for (k1, v1) ∈ cells
        for (k2, v2) ∈ cells
            flag, val = are_coarse_neighbors(v1, v2)
            if flag
                g[k1, k2] = val
            end
        end

        poly = cell_boundary_polygon(v1)
        symy = PlanePolygons.Line(Point(-2.0, 0.0), Vec(1.0, 0.0))
        inflow = PlanePolygons.Line(Point(-2.0, 0.0), Vec(0.0, 1.0))
        vN1 = PlanePolygons.Line(Point(-2.0, 2.0), Vec(1.0, 0.0))
        vN2 = PlanePolygons.Line(Point(0.0, 0.0), Vec(0.0, 1.0))

        dirs = (:south, :west, :north, :east)
        for (p1, p2, dir, t, n) ∈ zip(
            edge_starts(poly),
            edge_ends(poly),
            dirs,
            edge_tangents(poly),
            outward_edge_normals(poly),
        )
            #if is_other_point_on_line(symy, p1) && is_other_point_on_line(symy, p2)
            #	g[phantom_idx] = (DualNodeKind{:boundary_sym}(), nothing)
            #	g[k1, phantom_idx] = (dir, p1, p2)
            #	phantom_idx += 1
            #else
            if (p1[1] ≈ x_midleft(p1[2]) && p2[1] ≈ x_midleft(p2[2]))
                g[k1, 1] = (dir, p1, p2)
            elseif (
                is_other_point_on_line(vN1, p1) && is_other_point_on_line(vN1, p2) ||
                is_other_point_on_line(vN2, p1) && is_other_point_on_line(vN2, p2)
            )
                g[phantom_idx] = (DualNodeKind{:boundary_vN}(), nothing)
                g[k1, phantom_idx] = (dir, p1, p2)
                phantom_idx += 1
            elseif (norm(p1) ≈ body.radius && norm(p2) ≈ body.radius)
                g[phantom_idx] = (DualNodeKind{:boundary_body}(), nothing)
                g[k1, phantom_idx] = (dir, p1, p2)
                phantom_idx += 1
            end
        end
    end
    g
end

# ╔═╡ 0a3a069c-e72c-4a47-9a11-00f049dc137c
const _dirs_scale = map(b -> b ? -1 : 1, Euler2D._dirs_bc_is_reversed);

# ╔═╡ df5a5737-41e7-447d-ad5f-ebdbf07996ca
begin
    # compute the "index" for the coarse dual component "nbr"
    # because the values in nbr may actually be computed directly from u
    # or from "ambient" (idx is 1)
    _u_idx(::DualNodeKind{:boundary_sym}, id, nbr) = id
    _u_idx(::DualNodeKind{:boundary_vN}, id, nbr) = id
    _u_idx(::DualNodeKind{:boundary_body}, id, nbr) = id
    _u_idx(::DualNodeKind{:cell}, id, nbr) = nbr
    _u_idx(::DualNodeKind{:boundary_ambient}, id, nbr) = 1
end

# ╔═╡ 8a76792c-6189-4d39-9147-5a7ea9b074f9
begin
    project_to_basis(vec, w1, w2) = inv(hcat(w1, w2)) * vec
    function project_to_orthonormal_basis(vec, ŵ1)
        ŵ2 = SVector(-ŵ1[2], ŵ1[1])
        return project_to_basis(vec, ŵ1, ŵ2)
    end
    function project_state_to_basis(u, w1, w2)
        return SVector(u[1], project_to_basis(select_middle(u), w1, w2)..., u[4])
    end
    function project_state_to_orthonormal_basis(u, v̂_1)
        v̂_2 = SVector(-v̂_1[2], v̂_1[1])
        return project_state_to_basis(u, v̂_1, v̂_2)
    end
end

# ╔═╡ 6822601e-f40a-4fa1-a0f8-a8bb09809549
# computes ∇_x u using a right-facing stencil in x and centered stencil in y
# if we switch the solver to MUSCL2D, then we won't need this.
# this is just a proof-of-concept
function grad_x_u_at(sim, n, x, y)
	(dx_max, dy_max) = 2 .* minimum_cell_size(sim)
	window_x = (x-dx_max, x+dx_max)
	window_y = (y-dy_max, y+dy_max)
	_, cells = nth_step(sim, n)
	i = findfirst(>=(x), cell_boundaries(sim, 1))
	j = findfirst(>=(y), cell_boundaries(sim, 2))
	slice1 = max(1,i-3):min(sim.ncells[1], i+2)
	slice2 = max(1, j-3):min(sim.ncells[2], j+2)
	in_window = @view sim.cell_ids[slice1, slice2]
	cells_containing = Iterators.filter(in_window) do id
		id == 0 && return false
		p = cell_boundary_polygon(cells[id])
		return PlanePolygons.point_inside(p, Point(x, y))	
	end |> collect
	# we know that we have rectangular cells
	# in general this is not the case but I am not complaining
	return (mapreduce(+, cells_containing; init=zero(SMatrix{4, 2, Float64, 8})) do id
		cell = cells[id]
		nbrs = Euler2D.neighbor_cells(cell, cells, boundary_conditions, DRY_AIR)
		dudx = (nbrs.east.u - nbrs.west.u) / cell.extent[1]
		dudy = (nbrs.north.u - nbrs.south.u) / cell.extent[2]
		return hcat(dudx, dudy)
	end / length(cells_containing))
end

# ╔═╡ 5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
function compute_coarse_cell_contents(
    coarse_cell::CoarseQuadCell{T,NS,NTAN},
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NS,NTAN}},
    n,
) where {T,NS,NTAN}
    p = cell_boundary_polygon(coarse_cell)
    contained = all_cells_contained_by(p, sim)
    overlapped = all_cells_overlapping(p, sim)
    T1 = SVector{4,T}
    T2 = SMatrix{4,NS,T,NTAN}
    T3 = SMatrix{4,8,T,32}
	# total cell mass
    M = sum(contained; init = zero(T1)) do id
            _, cs = nth_step(sim, n)
            return Euler2D.cell_volume(cs[id]) * cs[id].u
        end + tmapreduce(+, overlapped; init = zero(T1)) do id
            _, cs = nth_step(sim, n)
            A_union = overlapping_cell_area(cs[id], coarse_cell)
            return A_union * cs[id].u
        end
	# dependence of cell mass on the larger polygon points
	# only depends on cells that are overlapped not contained
    dM_dp = tmapreduce(+, overlapped; init = zero(T3)) do id
        _, cs = nth_step(sim, n)
        p_cell = cell_boundary_polygon(cs[id])
		p_union = poly_intersection(p, p_cell)
        A_union = poly_area(p_union)
		du = mapreduce(hcat, edge_starts(p_union)) do pt
			grad_x_u_at(sim, n, pt...)
		end
		dxj = mapreduce(vcat, edge_starts(p_union)) do pt
			intersection_point_jacobian(pt, p, p_union)
		end
        dAdxj = intersection_area_jacobian(p, p_cell) 
        return cs[id].u * dAdxj' + A_union * du * dxj
    end
    Ṁ =
        sum(contained; init = zero(T2)) do id
            _, cs = nth_step(sim, n)
            return Euler2D.cell_volume(cs[id]) * cs[id].u̇
        end + tmapreduce(+,overlapped; init = zero(T2)) do id
            _, cs = nth_step(sim, n)
            A_union = overlapping_cell_area(cs[id], coarse_cell)
            return A_union * cs[id].u̇
        end
    A_loc, A_tangent_loc = DifferentiationInterface.value_and_gradient(
        poly_area,
        poly_area_prep,
        diff_backend,
        p,
    )
    # unpack from mooncake
    dA = SVector(A_tangent_loc.fields.data...)
    du_dp = (A_loc * dM_dp - M * dA') / (A_loc * A_loc)
    u, u̇ = (M, Ṁ) ./ A_loc
    return u, u̇, du_dp
end

# ╔═╡ dbc4b74e-572a-4c6c-8d10-a94ee7990950
tvals = compute_coarse_cell_contents(tcell1, sim_with_ad, FINAL_TSTEP_IDX)

# ╔═╡ c4f936ab-8ed0-406b-8ae8-d530f2f181df
sum(tvals[3]; dims=2)

# ╔═╡ a93d032b-525b-4748-8c83-950bad619b89
sanity_check_cell = compute_coarse_cell_contents(sanity_check[1], sim_with_ad, 1)

# ╔═╡ ed0c8c2a-8e27-4888-bf5f-08e788395bd1
map(v->isnan(v) ? 0. : v, sanity_check_cell[3]) * x_select

# ╔═╡ 1c7f6812-daef-4946-882d-19ee826d93a6
sanity_check_cell[3]

# ╔═╡ e62fd900-36b6-4369-a238-62a2478b116a
function populate_coarse_cells(empty_coarse, sim, n)
	d = copy(empty_coarse)
    for c ∈ keys(d)
		needs_flip = all(edge_starts(cell_boundary_polygon(d[c]))) do pt
			pt[2] ≤ 0.0
		end
		cell = d[c] 
		if needs_flip
			flipped = map(edge_starts(cell_boundary_polygon(d[c]))) do pt
				return SVector(pt[1], -pt[2])
			end
			new_poly = SVector(flipped[4]..., flipped[3]..., flipped[2]..., flipped[1]...)
			@reset cell.pts = new_poly
		end
        v1, v2, v3 = compute_coarse_cell_contents(cell, sim, n)
        @reset cell.u = v1
        @reset cell.u̇ = v2
        @reset cell.du_dpts = v3
		@reset cell.pts = d[c].pts
		d[c] = cell
    end
	idxs = filter(keys(d)) do k
		v = d[k]
		return all(edge_starts(cell_boundary_polygon(v))) do pt
			return pt[1] ≤ x_shock(pt[2])
		end
	end
	# foreach(idxs) do k
	# 	v = d[k]
	# 	@reset v.u = ambient_u
	# 	@reset v.u̇ = ambient_u̇
	# 	d[k] = v
	# end
    return d
end

# ╔═╡ 5d77d782-2def-4b3a-ab3a-118bf8e96b6b
coarse_cells = populate_coarse_cells(empty_coarse, sim_with_ad, FINAL_TSTEP_IDX)

# ╔═╡ c6e3873e-7fef-4c38-bf3f-de71f866057f
let
    xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
    yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
    spys = range(-0.25, 2.0; length = 20)
    spxs = x_shock.(spys)
    id = 0
    maxdensity = maximum(coarse_cells) do (_, c)
        c.u[1]
    end
    p = plot(
        xc,
        yc;
        aspect_ratio = :equal,
        xlims = (-1.5, 0.0),
        ylims = (-.05, 1.5),
        label = "Blunt Body",
        fill = true,
        dpi = 1000,
        size = (1200, 600),
        ls = :dash,
        lw = 4,
        fillalpha = 0.5,
		tickfontsize=16,
		legendfontsize=20
    )
    plot!(
        p,
        [-1.5, -0.75],
        [0.0, 0.0];
        color = :black,
        label = "Symmetry Boundary",
        lw = 2,
        ls = :dash,
    )
    plot!(
        p,
        [-1.5, -1.5],
        [0.0, 2.0];
        color = :green,
        label = "Inflow Boundary",
        lw = 6,
        ls = :dot,
    )
    plot!(
        p,
        [-1.5, 0.0],
        [2.0, 2.0];
        label = "v.N. Boundary",
        lw = 6,
        ls = :dashdot,
        legendfontsize = 14,
        color = :orange,
    )
	for (id, cell) ∈ coarse_cells
        poly = cell.pts
        pl = reduce(hcat, edge_starts(poly))
        plot!(
            p,
            @view(pl[1, :]),
            @view(pl[2, :]);
            lw = 0.5,
            fill = true,
            fillalpha = (cell.u[1] / maxdensity),
            label = false,
            color = :red,
            seriestype = :shape,
        )
        v = sum(eachcol(pl)) / 4
        id > 43 || id < 8 || annotate!(p, v..., Plots.text(L"%$id", 12))
    end
	plot!(
		p,
        spxs,
        spys;
        label = "Strong Shock Front (with extension)",
        lw = 4,
        legend = :outertopright,
    )
    savefig(p, "../gfx/new_cells_density_shaded_N$(num_coarse_cells_pos_y).pdf")
    p
end

# ╔═╡ 7e9ac0e4-37d7-41d0-98a7-7284634cb404
coarse_dual = produce_dual_graph_from_coarse_cells(coarse_cells)

# ╔═╡ 6cfeac68-96ac-401c-8f61-c2cafe8e9d8d
coarse_dual[9][2]

# ╔═╡ 9a31abc2-291a-4b76-be20-1135931d0dd2
coarse_dual[9][2].pts |> edge_starts

# ╔═╡ a30270bb-894e-47f2-9d88-ea354f32bf7d
[coarse_dual[9, i] for i in neighbor_labels(coarse_dual, 9)]

# ╔═╡ 5b7a3783-ef40-468f-93ac-91cb46929bd6
ne(coarse_dual)

# ╔═╡ 74525445-19f6-471f-878e-a60f07ba9f01
nv(coarse_dual)

# ╔═╡ 8aa0d03b-d363-4926-8297-b65673104910
[grad_x_u_at(sim_with_ad, 1, pt...) for pt ∈ edge_starts(coarse_dual[9][2].pts)]

# ╔═╡ bcf3281d-594e-4baf-add0-432c02882b1f
grad_x_u_at(sim_with_ad, 1, -1.1921875, 0.1248046875)

# ╔═╡ 768fa7ca-a8a3-49fc-ad6b-c47e68e71237
heatmap(-1.25:0.01:-0.75, 0.0:0.01:1.0, (x, y)->grad_x_u_at(sim_with_ad, FINAL_TSTEP_IDX, x, y)[1,1])

# ╔═╡ 2501fa77-dadb-428d-b98d-58b050e21a7d
heatmap(-1.25:0.01:-0.75, 0.0:0.01:1.0, (x, y)->grad_x_u_at(sim_with_ad, FINAL_TSTEP_IDX, x, y)[4,2])

# ╔═╡ 654863a5-9fc5-461a-b5ee-a2ffb2379888
edge_lengths(poly) = (norm(a - b) for (a, b) ∈ zip(edge_ends(poly), edge_starts(poly)))

# ╔═╡ 63c1d6d9-ad64-4074-83c4-40b5df0e0b1f
## compute L, n, t and e for a given pair of edge endpoints
function _edge_basis(edge_data)
    (_, p1, p2) = edge_data
    L = norm(p2 - p1)
    t̂ = (p2 - p1) / L
    n̂ = SVector(-t̂[2], t̂[1])
    ê1 = project_to_orthonormal_basis(SVector(1.0, 0.0), n̂)
    return L, n̂, t̂, ê1
end

# ╔═╡ cec776ee-f81d-4457-8555-24eaf80e4cca
begin
    function _compute_ϕ(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:cell},
        other_data,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        u_R = project_state_to_orthonormal_basis(other_data.u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    function _compute_ϕ(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:boundary_vN},
        ::Nothing,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        u_R = project_state_to_orthonormal_basis(cell_data.u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    function _compute_ϕ(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:boundary_ambient},
        other_data,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        u_R = project_state_to_orthonormal_basis(other_data, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    function _compute_ϕ(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
        ::Nothing,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        ρv = select_middle(cell_data.u)
        ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
        other_u = SVector(cell_data.u[1], ρv_reflected..., cell_data.u[4])
        u_R = project_state_to_orthonormal_basis(other_u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end
end

# ╔═╡ 4f79b51b-459a-46e1-a6d0-257ce08b029e
function boundary_integral(dual, id)
    nbrs = neighbor_labels(dual, id)
    return sum(nbrs) do nbr
        return _compute_ϕ(dual[id]..., dual[nbr]..., dual[id, nbr])
    end
end

# ╔═╡ 3c835380-3580-4881-a0ad-e466eb99fcb8
begin
    function _compute_grad_ϕ_u(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:cell},
        other_data,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        dϕ_dcell = jacobian(fdiff_backend, 
							cell_data.u, 
							Constant(other_data.u)
						   ) do u, u_other
            u_L = project_state_to_orthonormal_basis(u, n̂)
            u_R = project_state_to_orthonormal_basis(u_other, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end

        dϕ_dother = jacobian(fdiff_backend, other_data.u) do u
            u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
            u_R = project_state_to_orthonormal_basis(u, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end

        return dϕ_dcell, dϕ_dother
    end

    function _compute_grad_ϕ_u(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:boundary_ambient},
        other_data,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
            u_L = project_state_to_orthonormal_basis(u, n̂)
            u_R = project_state_to_orthonormal_basis(other_data, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end

        dϕ_dother = jacobian(fdiff_backend, other_data) do u
            u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
            u_R = project_state_to_orthonormal_basis(u, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end

        return dϕ_dcell, dϕ_dother
    end

    function _compute_grad_ϕ_u(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
        ::Nothing,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
            u_L = project_state_to_orthonormal_basis(u, n̂)
            ρv = select_middle(u)
            ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
            other_u = SVector(u[1], ρv_reflected..., u[4])
            u_R = project_state_to_orthonormal_basis(other_u, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end
        return dϕ_dcell, zero(dϕ_dcell)
    end

    function _compute_grad_ϕ_u(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:boundary_vN},
        ::Nothing,
        edge_data,
    )
        (L, n̂, t̂, ê1) = _edge_basis(edge_data)
        dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
            u_L = project_state_to_orthonormal_basis(u, n̂)
            u_R = project_state_to_orthonormal_basis(u, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end
        return dϕ_dcell, zero(dϕ_dcell)
    end
end

# ╔═╡ 03f640b6-851c-4c57-9ff8-5549415b558b
function boundary_integral_ddu(dual, id)
    return map(neighbor_labels(dual, id)) do nbr
        cell_kind, cell_data = dual[id]
        nbr_kind, nbr_data = dual[nbr]
        dϕ_did, dϕ_dnbr =
            _compute_grad_ϕ_u(cell_kind, cell_data, nbr_kind, nbr_data, dual[id, nbr])
        other_idx = _u_idx(nbr_kind, id, nbr)
        return (id, other_idx, dϕ_did, dϕ_dnbr)
    end
end

# ╔═╡ 76a93a70-0f45-45f5-b058-6c6aceb7fdae
function _marshal_edge_data(edge_data)
	return vcat(edge_data[2], edge_data[3])
end

# ╔═╡ efe20ee6-a671-4dc8-bc7b-ab7f76ce15cb
function _unmarshal_edge_data(v)
	return (SVector(v[1], v[2]), SVector(v[3], v[4]))
end

# ╔═╡ f3209f08-0d0c-4810-bf9b-86f2323799b4
begin
    function _compute_grad_ϕ_edge(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:cell},
        other_data,
        edge_data,
    )
		blob = _marshal_edge_data(edge_data)
		dϕ_dedge = jacobian(
			fdiff_backend, 
			blob, 
			Constant(cell_data.u), 
			Constant(other_data.u)
		) do v, cell_u, other_u
			p1, p2 = _unmarshal_edge_data(v)
			(L, n̂, t̂, ê1) = _edge_basis((nothing, p1, p2))
			u_L = project_state_to_orthonormal_basis(cell_u, n̂)
			u_R = project_state_to_orthonormal_basis(other_u, n̂)
			ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
			return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
		end
        return dϕ_dedge
    end
	
	function _compute_grad_ϕ_edge(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
        ::Nothing,
        edge_data,
    )
		blob = _marshal_edge_data(edge_data)
        dϕ_dedge = jacobian(
			fdiff_backend, 
			blob, 
			Constant(cell_data.u)
		) do v, u
			p1, p2 = _unmarshal_edge_data(v)
			(L, n̂, t̂, ê1) = _edge_basis((nothing, p1, p2))
            u_L = project_state_to_orthonormal_basis(u, n̂)
            ρv = select_middle(u)
            ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
            other_u = SVector(u[1], ρv_reflected..., u[4])
            u_R = project_state_to_orthonormal_basis(other_u, n̂)
            ϕ_n = ϕ(u_L, u_R, 1, DRY_AIR)
            return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
        end
        return dϕ_dedge
    end
end

# ╔═╡ 725a9b6f-5b70-4554-9e4b-ac7658e929fc
function boundary_integral_gradient(dual, id)
	edge_gradients = map(neighbor_labels(dual, id)) do nbr
		cell_kind, cell_data = dual[id]
		nbr_kind, nbr_data = dual[nbr]
		edge_data = dual[id, nbr]
		edge_length = norm(edge_data[3]-edge_data[2])
		dϕ_did, dϕ_dnbr = _compute_grad_ϕ_u(cell_kind, cell_data, nbr_kind, nbr_data, edge_data)
		dLϕ_dedge = _compute_grad_ϕ_edge(cell_kind, cell_data, nbr_kind, nbr_data, edge_data)
		other_idx = _u_idx(nbr_kind, id, nbr)
		return (edge_data[1], id, other_idx, dϕ_did, dϕ_dnbr, edge_length, dLϕ_dedge)
	end
	ddparams = sum(edge_gradients) do (_, i, j, dϕ_dui, dϕ_duj, _, _)
		return dϕ_dui * dual[i][2].u̇ + dϕ_duj * dual[j][2].u̇
	end
	_edge_point_indices = (north = (3, 4), south=(1, 2), east=(4, 1), west=(2, 3))
	_edge_opposites = (north=:south, south=:north, east=:west, west=:east)
	ddpts = zeros(size(ddparams)[1], 8)
	foreach(edge_gradients) do (dir, i, j, dϕ_dui, dϕ_duj, L, dLϕ_dpts)
		local (k, l) = _edge_point_indices[dir]
		# reversed numbering in the cell on the other side!
		local (n, m) = _edge_point_indices[_edge_opposites[dir]]
		ddpts[:, (2*k-1):2*k] += dLϕ_dpts[:, SVector(1,2)]
		ddpts[:, (2*l-1):2*l] += dLϕ_dpts[:, SVector(3,4)]

		ddpts[:, (2*k-1):2*k] += dϕ_dui * dual[i][2].du_dpts[:, (2*k-1):2*k]
		ddpts[:, (2*l-1):2*l] += dϕ_dui * dual[i][2].du_dpts[:, (2*l-1):2*l]

		ddpts[:, (2*k-1):2*k] += dϕ_duj * dual[j][2].du_dpts[:, (2*m-1):2*m]
		ddpts[:, (2*l-1):2*l] += dϕ_duj * dual[j][2].du_dpts[:, (2*n-1):2*n]
	end
	return ddparams, ddpts
end

# ╔═╡ c978c79b-1503-4818-92e1-3d1660d9ae2a
let
	a, b = boundary_integral_gradient(coarse_dual, 12)
	c = b * x_select
	-pinv(c) * a
end

# ╔═╡ d0b579a8-38a4-45a9-b87b-7afa68fdb957
function ξ_local_pseudoinversion(dual, id, ε) # where the magic happens
	# A = boundary_integral_ddparams(dual, id)
	# @assert rank(A) == min(size(A)...)
	# B = boundary_integral_ddL(dual, id)
	# @assert rank(B) == min(size(B)...)
	# C = pinv(B)*A
	local (A, B) = boundary_integral_gradient(dual, id)
	local C = pinv(B * x_select)
	return -(C*A)*ε
end

# ╔═╡ 389b5069-b0fd-410b-82b1-7eebcb3c5ade
ξ_local_pseudoinversion(coarse_dual, 9, [0., 0.1, 0.])

# ╔═╡ b0f33a6e-52d5-471c-929b-532b70e9f874
function plot_local_pseudoinversion(ε, cells, cells_dual)
	# since we only took three coarse cells per row!
	behind_shock_cells = (keys(cells) |> collect |> sort!)[2:3:end-2]
	infos_temp = mapreduce(vcat, behind_shock_cells) do id
		ξ_local_pseudoinversion(cells_dual, id, ε)
	end
	infos = (infos_temp[1:2:end] + infos_temp[2:2:end]) ./ 2
	info_ys = map(behind_shock_cells) do k
		v = coarse_cells[k]
		pts = reduce(hcat, edge_starts(cell_boundary_polygon(v)))
		return sum(pts'[2:3, 2]) / 2
	end
	f(y) = BilligShockParametrization.shock_front(y, ambient_primitives[2], 0.75)[1]
	df(y) = begin
		ForwardDiff.gradient(ambient_primitives) do uinf
			BilligShockParametrization.shock_front(y, uinf[2], 0.75)[1]
		end
	end
	y = 0.:0.01:2.0
	y2 = 0.:0.05:2.0
	x = f.(y)
	eps_titlestr = reduce(*, [@sprintf("%.3f ", v) for v ∈ ε])
	p = plot(x, y, label="Estimated "*L"(M=4.0)", title=L"N_x=%$(NCELLS_X), N_{coarse}=%$(num_coarse_cells_pos_y),\varepsilon="*eps_titlestr, titlefontsize=12)
	#@show length(infos), length(info_ys)
	#plot!(p, x .+ map(y->(df(y))'*ε, y), y, label="Billig "*L"(M = 4.0+ε)")
	plot!(p, x_shock.(y2), y2, label="Extracted via Shock Sensor")
	plot!(p, x_shock.(info_ys) .+ infos, info_ys, label="Extracted + Perturbation", marker=:circ)
	p
end

# ╔═╡ b2840f27-70da-4d5a-a8f1-2ebcb7db2896
let # mach number perturbation
	ε = test_perturbations[2,:]
	p = plot_local_pseudoinversion(ε, coarse_cells, coarse_dual)
	savefig(p, "../gfx/local_pseudoinversion_Nx$(NCELLS_X)_Ncoarse$(num_coarse_cells_pos_y)_perturbationscale$(ε_scale)_mach_number.pdf")
	p
end

# ╔═╡ 3ce0c10d-ee28-4d89-bae3-3f0060ec7475
let # temperature perturbation
	ε = test_perturbations[3,:]
	p = plot_local_pseudoinversion(ε, coarse_cells, coarse_dual)
	savefig(p, "../gfx/local_pseudoinversion_Nx$(NCELLS_X)_Ncoarse$(num_coarse_cells_pos_y)_perturbationscale$(ε_scale)_temperature.pdf")
	p
end

# ╔═╡ 4dcafe21-9172-4473-bf1b-5e96786067da
let # density perturbation
	ε = test_perturbations[1,:]
	p = plot_local_pseudoinversion(ε, coarse_cells, coarse_dual)
	savefig(p, "../gfx/local_pseudoinversion_Nx$(NCELLS_X)_Ncoarse$(num_coarse_cells_pos_y)_perturbationscale$(ε_scale)_density.pdf")
	p
end

# ╔═╡ 08643da2-a430-4661-9fa3-61ffb78a7572
function _marshal_edge_basis(args)
	return vcat(args...)
end

# ╔═╡ bf889528-da6b-4b75-8eb4-84e5c19cb3c8
function _unmarshal_edge_basis(v)
	@assert length(v) == 7
	return (v[1], SVector(v[2], v[3]), SVector(v[4],v[5]), SVector(v[6], v[7]))
end

# ╔═╡ dbf7ca07-521e-4e83-aea7-12678d095ea0
function _unmarshal_edge_data_gradient(g)
	@assert size(g) == (7, 4)
	return (g[1, :], g[SVector(2, 3), :], g[SVector(4, 5), :], g[SVector(6, 7), :])
end

# ╔═╡ 4dcf8a00-8b44-4634-9c8e-7febb2e3f583
function _edge_basis_and_gradient(edge_data)
	blob = _marshal_edge_data(edge_data)
	res, dres =  value_and_jacobian(fdiff_backend, blob) do v
		p1, p2 = _unmarshal_edge_data(v)
		return _marshal_edge_basis(_edge_basis((nothing, p1, p2)))
	end
	return (_unmarshal_edge_basis(res), _unmarshal_edge_data_gradient(dres))
end

# ╔═╡ Cell order:
# ╠═6f1542ea-a747-11ef-2466-fd7f67d1ef2c
# ╠═2e9dafda-a95c-4277-9a7c-bc80d97792f0
# ╠═3aea1823-86ff-4b8b-9a4c-c87a4fbf4dff
# ╠═f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
# ╟─e7001548-a4b4-4709-b10c-0633a11bd624
# ╟─c87b546e-8796-44bf-868c-b2d3ad340aa1
# ╠═4267b459-7eb7-4678-8f06-7b9deab1f830
# ╟─2716b9b5-07fd-4175-a83e-22be3810e4b3
# ╟─afc11d27-1958-49ba-adfa-237ba7bbd186
# ╟─3fe1be0d-148a-43f2-b0a5-bb177d1c041d
# ╠═0df888bd-003e-4b49-9c2a-c28a7ccc33d2
# ╟─136ab703-ae33-4e46-a883-0ed159360361
# ╟─a9d31e2d-fc4b-4fa5-9015-eb2ac2a3df5d
# ╟─e55363f4-5d1d-4837-a30f-80b0b9ae7a8e
# ╟─d832aeb4-42d6-4b72-88ee-4cdd702a4f48
# ╠═90bf50cf-7254-4de8-b860-938430e121a9
# ╟─d29aa465-fcba-4210-9809-c92e6bf604bf
# ╟─09efacdf-2e99-4e74-b00b-b021955a3220
# ╟─7422611b-596a-48e1-9e24-0e8d8bd1eba8
# ╠═fffcb684-9b58-43d7-850a-532c609c5389
# ╟─33e635b3-7c63-4b91-a1f2-49da93307f29
# ╟─0cedbab2-ade9-4e8d-aa3c-a0c9cc28eaaa
# ╟─8bd1c644-1690-46cf-ac80-60654fc6d8c0
# ╟─893ec2c8-88e8-4d72-aab7-88a1efa30b47
# ╠═f6147284-02ec-42dd-9c2f-a1a7534ae9fa
# ╟─d14c3b81-0f19-4207-8e67-13c09fd7636a
# ╠═cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
# ╠═d5db89be-7526-4e6d-9dec-441f09606a04
# ╠═2e3b9675-4b66-4623-b0c4-01acdf4e158c
# ╟─0536f540-fa7b-48ed-8746-cd7cbce16555
# ╟─db1c858e-a483-40ef-aabc-2aa58de73e92
# ╟─bcdd4862-ac68-4392-94e2-30b1456d411a
# ╠═4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
# ╟─44ff921b-09d0-42a4-8852-e911212924f9
# ╟─4f8b4b5d-58de-4197-a676-4090912225a1
# ╠═6e4d2f60-3c40-4a2b-be2b-8c4cc40fb911
# ╟─706146ae-3dbf-4b78-9fcc-e0832aeebb28
# ╟─9b6ab300-6434-4a96-96be-87e30e35111f
# ╠═21cbdeec-3438-4809-b058-d23ebafc9ee2
# ╠═df4a4b15-b581-4e3c-a363-c1ac5a5a2999
# ╟─90ff1023-103a-4342-b521-e229157001fc
# ╟─5c0be95f-3c4a-4062-afeb-3c1681cae549
# ╟─88889293-9afc-4540-a2b9-f30afb62b1de
# ╟─6da05b47-9763-4d0c-99cc-c945630c770d
# ╟─351d4e18-4c95-428e-a008-5128f547c66d
# ╟─bc0c6a41-adc8-4d18-9574-645704f54b72
# ╟─4a5086bc-5c36-4e71-9d3a-8f77f48a90f9
# ╠═747f0b67-546e-4222-abc8-2007daa3f658
# ╠═2cb33e16-d735-4e60-82a5-aa22da0288fb
# ╠═4b036b02-1089-4fa8-bd3a-95659c9293cd
# ╟─24da34ca-04cd-40ae-ac12-c342824fa26e
# ╟─9ac61e80-7d6a-40e8-8254-ee306f8248c3
# ╟─9c601619-aaf1-4f3f-b2e2-10422d8ac640
# ╟─e0a934d6-3962-46d5-b172-fb970a537cc0
# ╠═62ebd91b-8980-4dc5-b61b-ba6a21ac357d
# ╠═be8ba02d-0d31-4720-9e39-595b549814cc
# ╟─92044a9f-2078-48d1-8181-34be87b03c4c
# ╟─93043797-66d1-44c3-b8bb-e17deac70cfa
# ╠═766b440b-0001-4037-8959-c0b7f04d999e
# ╠═7468fbf2-aa57-4505-934c-baa4dcb646fc
# ╠═af0732ba-b280-4dd0-8bed-2170a2a4d6f8
# ╠═76cb5757-140e-429f-8190-2e5ae888d3a4
# ╠═4d202323-e1a9-4b24-b98e-7d17a6cc144f
# ╠═2041d463-5382-4c38-bf52-23d22820ac59
# ╠═54ed2abb-81bb-416c-b7be-1125e41622f5
# ╠═f30619a3-5344-4e81-a4b5-6a11100cd056
# ╠═9a239bb8-9140-43ad-9b1b-b85c5e78b40a
# ╠═dea032e2-bf23-42da-8dac-d3368c2bdec6
# ╟─e98df040-ac22-4184-95dd-a8635ab72af6
# ╟─95947312-342f-44b3-90ca-bd8ad8204e18
# ╠═d790b1ba-180e-4de2-bed2-56155a8b8dff
# ╠═dbc4b74e-572a-4c6c-8d10-a94ee7990950
# ╠═c4f936ab-8ed0-406b-8ae8-d530f2f181df
# ╠═5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
# ╠═a93d032b-525b-4748-8c83-950bad619b89
# ╠═ed0c8c2a-8e27-4888-bf5f-08e788395bd1
# ╠═e62fd900-36b6-4369-a238-62a2478b116a
# ╠═5d77d782-2def-4b3a-ab3a-118bf8e96b6b
# ╟─a1dc855f-0b24-4373-ba00-946719d38d95
# ╟─6aaf12a0-c2d9-48ab-9e13-94039cf95258
# ╟─c6e3873e-7fef-4c38-bf3f-de71f866057f
# ╠═6cfeac68-96ac-401c-8f61-c2cafe8e9d8d
# ╠═1c7f6812-daef-4946-882d-19ee826d93a6
# ╠═9a31abc2-291a-4b76-be20-1135931d0dd2
# ╠═a30270bb-894e-47f2-9d88-ea354f32bf7d
# ╠═8aa0d03b-d363-4926-8297-b65673104910
# ╠═bcf3281d-594e-4baf-add0-432c02882b1f
# ╟─3a8cd7e2-fae9-4e70-8c92-004b17352506
# ╠═ead8c1a5-9f4e-4d92-b4ca-1650ad34bdca
# ╠═e650d2b0-366c-4595-8fc3-69db62c14579
# ╟─7e9ac0e4-37d7-41d0-98a7-7284634cb404
# ╠═fd73e7b8-5887-4fee-9d8c-c8df45e54d11
# ╠═4f79b51b-459a-46e1-a6d0-257ce08b029e
# ╠═03f640b6-851c-4c57-9ff8-5549415b558b
# ╠═da01e1fa-876e-4bdd-8f2e-e8df524fe7cb
# ╠═725a9b6f-5b70-4554-9e4b-ac7658e929fc
# ╠═c978c79b-1503-4818-92e1-3d1660d9ae2a
# ╠═d0b579a8-38a4-45a9-b87b-7afa68fdb957
# ╠═389b5069-b0fd-410b-82b1-7eebcb3c5ade
# ╟─cdbc037e-d542-47be-b78e-b396c5c3d8df
# ╠═e5036dd3-9070-4521-9d7d-e0293b967d78
# ╟─ae7bca74-2a14-4a62-b6e9-669260fa4011
# ╠═b0f33a6e-52d5-471c-929b-532b70e9f874
# ╠═4cb96329-f3f4-4874-9ec3-63665abc8d22
# ╠═3ee5edb5-e648-489f-9c17-10c3912e0fcd
# ╠═b2840f27-70da-4d5a-a8f1-2ebcb7db2896
# ╠═3ce0c10d-ee28-4d89-bae3-3f0060ec7475
# ╠═4dcafe21-9172-4473-bf1b-5e96786067da
# ╟─9e9cc14d-bca7-4bea-832e-f4cecb556e90
# ╟─1f174634-812d-4704-8e82-36935e8f0cb5
# ╟─8f36cc9d-c2c5-4bea-9dc7-e5412a2960f9
# ╠═48148c30-486b-4952-9c5f-aea722ff74cf
# ╠═768fa7ca-a8a3-49fc-ad6b-c47e68e71237
# ╠═2501fa77-dadb-428d-b98d-58b050e21a7d
# ╠═5b7a3783-ef40-468f-93ac-91cb46929bd6
# ╠═74525445-19f6-471f-878e-a60f07ba9f01
# ╠═d8bfb40f-f304-41b0-9543-a4b10e95d182
# ╠═2c0133c2-35d5-4cfd-b19d-dfb08712479f
# ╠═0ef2267a-2116-469c-9b50-9b85be4ccc45
# ╠═3750e0d5-5ca3-447a-bbf5-add15a4a4e9b
# ╠═a1d7888c-34a2-425b-a8e4-2329b6400901
# ╟─68650137-22f6-4c1a-814b-f9b9c81e20ca
# ╠═501536eb-2f3a-4dfd-b126-0c7999782802
# ╠═604d3736-15b0-44f6-99eb-49ba7186762b
# ╠═9c005d9f-46bb-4b75-8233-f4e1f2cf0663
# ╠═cac50bdc-55f8-4073-9048-c83deda836b7
# ╠═8f3d93ad-06fd-43bd-bf8f-004b2c80a5f2
# ╠═70d0aac2-6813-4d02-a363-0951a3d9592c
# ╟─284ad6c0-b211-48aa-aa4f-d354c0d5520b
# ╠═e4b54bd3-5fa9-4291-b2a4-6b10c494ce34
# ╠═37eb63be-507a-475f-a6f6-8606917b8561
# ╠═61945d45-03f4-4401-8b18-3d11420047d0
# ╟─c1a81ef6-5e0f-4ad5-8e73-e9e7f09cefa6
# ╠═6f303343-1f76-46f9-80e8-2dd4ae1b5427
# ╟─436a3325-51bb-4718-aa08-f835e7f98ddd
# ╠═729ebc48-bba1-4858-8369-fcee9f133ee0
# ╠═5cffaaf5-9a5e-4839-a056-30e238308c51
# ╠═f252b8d0-f067-468b-beb3-ff6ecaeca722
# ╠═571b1ee7-bb07-4b30-9870-fbd18349a2ef
# ╠═80cde447-282a-41e5-812f-8eac044b0c15
# ╠═d87e0bb8-317e-4d48-8008-a7071c74ab31
# ╠═244838df-ae1f-40a3-a94e-50222969a3c0
# ╠═a968296a-43c1-48f3-b4b8-0d81cb162b7b
# ╠═0e0a049b-e2c3-4fe9-8fb8-186cdeb60485
# ╟─95c1ba4e-32c0-4b4c-b30b-766b5fd9f622
# ╠═d44322b1-c67f-4ee8-b168-abac75fb42a1
# ╟─34c24942-a195-4ce4-98fb-bad2f76d3a2c
# ╠═d19fff76-e645-4d9d-9989-50019b6356ad
# ╟─2f088a0c-165e-47f9-aaeb-6e4ab31c9d26
# ╟─79f2ec98-efc4-4215-968a-9c3c2cb52004
# ╟─cd312803-3819-4451-887b-ce2b53bb6e1b
# ╟─881e3b6e-ecdc-41b1-b21a-69de7e42af36
# ╟─ac412980-1013-450f-bb23-0dc7c2b3f199
# ╠═0d6ae7cf-edae-48f6-a257-6223563b7c76
# ╟─602b8184-90d5-4d19-906a-b41f278d1761
# ╠═fd9b689f-275c-4c91-9b6c-4e63c68d6ab2
# ╠═0a3a069c-e72c-4a47-9a11-00f049dc137c
# ╠═df5a5737-41e7-447d-ad5f-ebdbf07996ca
# ╠═8a76792c-6189-4d39-9147-5a7ea9b074f9
# ╠═cec776ee-f81d-4457-8555-24eaf80e4cca
# ╠═f3209f08-0d0c-4810-bf9b-86f2323799b4
# ╠═3c835380-3580-4881-a0ad-e466eb99fcb8
# ╠═6822601e-f40a-4fa1-a0f8-a8bb09809549
# ╠═654863a5-9fc5-461a-b5ee-a2ffb2379888
# ╠═63c1d6d9-ad64-4074-83c4-40b5df0e0b1f
# ╠═4dcf8a00-8b44-4634-9c8e-7febb2e3f583
# ╠═76a93a70-0f45-45f5-b058-6c6aceb7fdae
# ╠═efe20ee6-a671-4dc8-bc7b-ab7f76ce15cb
# ╠═08643da2-a430-4661-9fa3-61ffb78a7572
# ╠═bf889528-da6b-4b75-8eb4-84e5c19cb3c8
# ╠═dbf7ca07-521e-4e83-aea7-12678d095ea0
