### A Pluto.jl notebook ###
# v0.20.8

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
begin
    using Accessors
    using DifferentiationInterface
    using ForwardDiff
    using Interpolations: linear_interpolation
    using LaTeXStrings
    using LinearAlgebra
    import Mooncake
    using Plots
    using PlutoUI
    using Printf
    using ShockwaveProperties
    using StaticArrays
    using Unitful
end

# ╔═╡ 2e9dafda-a95c-4277-9a7c-bc80d97792f0
begin
    using Euler2D
    using Euler2D: TangentQuadCell
end

# ╔═╡ 15c5d10c-4d9a-4f57-a8fd-b65a970ac73c
begin
	using OhMyThreads
end

# ╔═╡ 31009964-3f32-4f97-8e4a-2b95be0f0037
using PlanePolygons

# ╔═╡ 0679a676-57fa-45ee-846d-0a8961562db3
begin
    using Graphs
    using MetaGraphsNext
end

# ╔═╡ e4b54bd3-5fa9-4291-b2a4-6b10c494ce34
# get some internal bindings from Euler2D
using Euler2D: _dirs_dim, select_middle

# ╔═╡ e5036dd3-9070-4521-9d7d-e0293b967d78
using ShockwaveProperties: BilligShockParametrization

# ╔═╡ d8bfb40f-f304-41b0-9543-a4b10e95d182
using PlanePolygons: _poly_image

# ╔═╡ f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
PlutoUI.TableOfContents()

# ╔═╡ e7001548-a4b4-4709-b10c-0633a11bd624
md"""
# Numerical GTVs

"""

# ╔═╡ c87b546e-8796-44bf-868c-b2d3ad340aa1
md"""
## Setup
Declare ``u(\vec{x}, 0; p) = u_0`` and provide a useful scale to to nondimensionalize the Euler equations.

The parameters (taken from a previously-done simulation) are:
 - ``ρ_\inf=0.662\frac{\mathrm{kg}}{\mathrm{m}^3}``
 - ``M_\inf=4.0``
 - ``T_\inf=220\mathrm{K}``

"""

# ╔═╡ 4267b459-7eb7-4678-8f06-7b9deab1f830
const ambient_primitives = SVector(0.662, 4.0, 220.0)

# ╔═╡ 2716b9b5-07fd-4175-a83e-22be3810e4b3
md"""
We can set up `u0` to always return the conserved quantities computed from the ambient primitives. 
"""

# ╔═╡ afc11d27-1958-49ba-adfa-237ba7bbd186
function u0(x, p)
    # ρ, M, T -> ρ, ρv, ρE
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

# ╔═╡ 100a1f91-120b-43a4-8486-8f9e64b8b71e
md"""
And, of course, the corresponding scaling for the Euler equations.
"""

# ╔═╡ 0df888bd-003e-4b49-9c2a-c28a7ccc33d2
const ambient = u0((-Inf, 0.0), ambient_primitives)

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

# ╔═╡ d832aeb4-42d6-4b72-88ee-4cdd702a4f48
md"""
Load up a data file. This contains a forward-mode computation on a fine grid allowed to run to $T=20$.
"""

# ╔═╡ 90bf50cf-7254-4de8-b860-938430e121a9
sim_with_ad = 
	#load_cell_sim("../data/tangent_last_tstep.celltape");
	load_cell_sim("../data/probe_obstacle_tangent_very_long_time_selected_tsteps.celltape");

# ╔═╡ fffcb684-9b58-43d7-850a-532c609c5389
boundary_conditions = (ExtrapolateToPhantom(), StrongWall(), ExtrapolateToPhantom(), ExtrapolateToPhantom(), StrongWall())

# ╔═╡ 33e635b3-7c63-4b91-a1f2-49da93307f29
md"""
We also know that this simulation was done with a blunt, cylindrical obstacle of radius ``0.75`` located at the origin.
"""

# ╔═╡ 4dc7eebf-48cc-4474-aef0-0cabf1d8eda5
body = CircularObstacle(SVector(0.0, 0.0), 0.75);

# ╔═╡ 8bd1c644-1690-46cf-ac80-60654fc6d8c0
md"""
## Pressure Field Sensitivities
This mirrors the declaration of `pressure_field`, but returns `missing` values when there's no pressure field value to compute.
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
Computing the full gradient ``\nabla_pP`` is a bit finicky, but ultimately works out to be repeated Jacobian-vector products over the pressure field.
"""

# ╔═╡ 2e3b9675-4b66-4623-b0c4-01acdf4e158c
@bind n Slider(1:n_tsteps(sim_with_ad); default = 2, show_value = true)

# ╔═╡ f6147284-02ec-42dd-9c2f-a1a7534ae9fa
pfield = map(pressure_field(sim_with_ad, n, DRY_AIR)) do val
    isnothing(val) ? missing : val
end;

# ╔═╡ cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
pressure_tangent = dPdp(sim_with_ad, n);

# ╔═╡ d5db89be-7526-4e6d-9dec-441f09606a04
begin
    pplot = heatmap(
        pfield';
        xlims = (0, 400),
        ylims = (0, 400),
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
            xlims = (0, 400),
            ylims = (0, 400),
            aspect_ratio = :equal,
            clims = cbar_limits[i],
            title = titles[i],
        ) for i = 1:3
    ]
    plots = reshape(vcat(pplot, dpplot), (2, 2))
    xlabel!.(plots, L"i")
    ylabel!.(plots, L"j")
    plot(plots...; size = (800, 800), dpi = 1000)
end

# ╔═╡ 4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
@bind n2 Slider(1:n_tsteps(sim_with_ad), default = 2, show_value = true)

# ╔═╡ bcdd4862-ac68-4392-94e2-30b1456d411a
let
    dPdM = dPdp(sim_with_ad, n2)
    title = L"\frac{\partial P}{\partial M_\inf}"
    p = heatmap(
        (@view(dPdM[2, :, :]))';
        xlims = (0, 400),
        ylims = (0, 400),
        aspect_ratio = :equal,
        clims = (-10, 10),
        title = title,
        size = (450, 600),
    )
    p
end

# ╔═╡ e2bdc923-53e6-4a7d-9621-4d3b356a6e41
md"""
## Shock Sensitivities
"""

# ╔═╡ 44ff921b-09d0-42a4-8852-e911212924f9
md"""
### Shock Sensor
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

# ╔═╡ 21cbdeec-3438-4809-b058-d23ebafc9ee2
function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
    Gy = _avg_op(T) * _diff_op(T)'
    Gx = _diff_op(T) * _avg_op(T)'
    new_size = size(matrix) .- 2
    outX = similar(matrix, new_size)
    outY = similar(matrix, new_size)
    for i ∈ eachindex(IndexCartesian(), outX, outY)
        view_range = i:(i+CartesianIndex(2, 2))
        outX[i] = Gx ⋅ @view(matrix[view_range])
        outY[i] = Gy ⋅ @view(matrix[view_range])
    end
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

# ╔═╡ a974c692-5171-4049-95aa-def2c776061b
_diff_op(Float64) * _avg_op(Float64)'

# ╔═╡ 4db61526-1e3a-47ac-8e2f-66634c3947c2
_avg_op(Float64) * _diff_op(Float64)'

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

# ╔═╡ 4b036b02-1089-4fa8-bd3a-95659c9293cd
sf = find_shock_in_timestep(
    sim_with_ad,
    6,
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
        xlims = (-1.5, 0.5),
        ylims = (0., 2.0),
        xlabel = L"x",
        ylabel = L"y",
        size = (700, 500),
        dpi = 1000,
    )
    #savefig(p, "../gfx/shock_sensor_07_01.pdf")
    p
end

# ╔═╡ 9ac61e80-7d6a-40e8-8254-ee306f8248c3
let
	pf = map(p -> isnothing(p) ? 0.0 : p, pressure_field(sim_with_ad, 6, DRY_AIR))
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
	p2 = heatmap(x[3:end-2], y[3:end-2], edge_candidates', aspect_ratio=:equal, ylims=(0., 2.), xlims=(-1.5, 0.5))
	#plot(p1, p2)
	p2
end

# ╔═╡ 92044a9f-2078-48d1-8181-34be87b03c4c
md"""
## Drawing a New Mesh

We are primarily interested in what happens when we vary any of the parameters in the ambient flow change. 

If we take the cell-average data available from the simulation, we can choose a (perhaps even slightly) coarser grid than was used for the simulation, and extract a linear interpolation of the bow shock.
"""

# ╔═╡ 5268fe37-b827-42ad-9977-6efbf4ecfad1
md"""
It's also necessary to expand the existing interface to Euler2D to polygon types...
"""

# ╔═╡ e98df040-ac22-4184-95dd-a8635ab72af6
md"""
---
"""

# ╔═╡ 4d202323-e1a9-4b24-b98e-7d17a6cc144f
struct CoarseQuadCell{T,NS,NTAN}
    id::Int
    pts::SVector{8,T}
    u::SVector{4,T}
    u̇::SMatrix{4,NS,T,NTAN}
    du_dpts::SMatrix{4,8,T,32}
end

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

# ╔═╡ eb5a2dc6-9c7e-4099-a564-15f1cec11caa
md"""
---
"""

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

# ╔═╡ 62ebd91b-8980-4dc5-b61b-ba6a21ac357d
all_shock_points = shock_points(sim_with_ad, 2, sf);

# ╔═╡ be8ba02d-0d31-4720-9e39-595b549814cc
sp_interp = linear_interpolation(all_shock_points[:, 2], all_shock_points[:, 1]);

# ╔═╡ a1dc855f-0b24-4373-ba00-946719d38d95
md"""
---
"""

# ╔═╡ 93043797-66d1-44c3-b8bb-e17deac70cfa
md"""
If we take a set of points along the ``y``-axis, we can create cells that have vertices on any of the:
 - computational domain boundaries
 - bow shock
 - blunt body

We can construct cells from these points and apply the conservation law again to compute ``εξ``.
"""

# ╔═╡ 7468fbf2-aa57-4505-934c-baa4dcb646fc
const cell_width_at_shock = 0.075

# ╔═╡ 50a46d6d-deb7-4ad0-867a-4429bf55632f
md"""
This should be ``8k``...
"""

# ╔═╡ 766b440b-0001-4037-8959-c0b7f04d999e
const num_coarse_cells_pos_y = 16

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

# ╔═╡ 24fa22e6-1cd0-4bcb-bd6d-5244037e58e2
x_midleft(y) = x_shock(y) - cell_width_at_shock

# ╔═╡ cd312803-3819-4451-887b-ce2b53bb6e1b
x_right(y) = x_shock(y) + cell_width_at_shock

# ╔═╡ 7c8cde49-b9c6-4889-a328-bf46b6f82a01
x_midright(y) = x_shock(y) + 2 * cell_width_at_shock

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
        range(x_shock(y), x_body(y); step = cell_width_at_shock),
        x_body(y),
    )
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

# ╔═╡ bfe8cb7d-8e5e-4dda-88cb-356b34017335
md"""
---
"""

# ╔═╡ dea032e2-bf23-42da-8dac-d3368c2bdec6
let
    xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
    yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
    p = plot(
        xc,
        yc;
        aspect_ratio = :equal,
        xlims = (-2.05, 0.05),
        ylims = (-0.2, 2.2),
        label = "Blunt Body",
        fill = true,
        dpi = 1000,
        size = (1000, 1000),
        ls = :dash,
        lw = 4,
        fillalpha = 0.5,
    )
    pts = [Point(x, y) for y ∈ ypts2 for x ∈ points_row(y)]
    scatter!(p, [pt[1] for pt ∈ pts], [pt[2] for pt ∈ pts]; marker = :x)

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
                fillalpha = 0.5,
                label = false,
                color = :red,
                seriestype = :shape,
            )
        end
    end

    p
end

# ╔═╡ fc646016-a30e-4c60-a895-2dde771f79cb
md"""
---

We can draw a new set of quadrilaterals:
- Some in front of the shock
- Some immediately behind the shock
- Some further away from the shock

We'll use these to solve the conservation law again. The index `1` is reserved as a placeholder for all cells with ``u=u_\infty`` and ``\dot u = \dot u_\infty``. 
"""

# ╔═╡ 54ed2abb-81bb-416c-b7be-1125e41622f5
all_polys = mapreduce(vcat, 1:num_coarse_cells) do i
    x1 = points_row(ypts2[i])
    x2 = points_row(ypts2[i+1])
    polys = make_polys(ypts2[i], ypts2[i+1], x1, x2)
    return polys
end

# ╔═╡ 435887e7-1870-4677-a09d-020be5761039
md"""

We'll need lots of helper functions to deal with the new mesh.

---
"""

# ╔═╡ 61945d45-03f4-4401-8b18-3d11420047d0
minimum_cell_size(sim) = mapreduce(c->c.extent, (a, b,)->min.(a, b), values(nth_step(sim, 1)[2]); init=(Inf, Inf))

# ╔═╡ 0511fbcd-8b44-481c-a176-c0657b6557c2
minimum_cell_size(sim_with_ad)

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
		window_x[1] ≤ cells[id].center[1] ≤ window_x[2] || return false
		window_y[1] ≤ cells[id].center[2] ≤ window_y[2] || return false
		return true
	end
	return collect(Iterators.filter(in_window) do id
        return is_cell_overlapping(cells[id], poly)
    end)
end

# ╔═╡ 80cde447-282a-41e5-812f-8eac044b0c15
function overlapping_cell_area(cell1, cell2)
    isect = poly_intersection(cell_boundary_polygon(cell1), cell_boundary_polygon(cell2))
    return poly_area(isect)
end

# ╔═╡ 48a2b845-c466-4bbc-aa16-46a95ed7be35
md"""
---

Then, we'll want to compute the coarse mesh geometry.
"""

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

# ╔═╡ 6c2b1a68-dd43-4449-9dc7-4b7849081cc3
md"""
Followed by the cell values in the new mesh, for which we'll need an AD tool:
"""

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

# ╔═╡ 6f303343-1f76-46f9-80e8-2dd4ae1b5427
function fdiff_eps(arg::T) where {T<:Real}
	cbrt_eps = cbrt(eps(T))
	h = 2^(round(log2((1+abs(arg))*cbrt_eps)))
	return h
end

# ╔═╡ d87e0bb8-317e-4d48-8008-a7071c74ab31
# gets the jacobian of the intersection area w.r.t. the first argument
function intersection_area_jacobian(flat_poly1, poly2)
    grad1 = zero(flat_poly1)
    for i in eachindex(flat_poly1)
		h = fdiff_eps(flat_poly1[i])
        in1 = @set flat_poly1[i] += h
        in2 = @set flat_poly1[i] -= h
        out1 = poly_area(poly_intersection(in1, poly2))
        out2 = poly_area(poly_intersection(in2, poly2))
        @reset grad1[i] = (out1 - out2) / (2 * h)
    end
    return grad1
end

# ╔═╡ 6aaf12a0-c2d9-48ab-9e13-94039cf95258
md"""
Numerical viscosity or numerical dissipation may have affected things more strongly than we would like. Brief inspection reveals that the relative error in _front_ of the shock is around 10%.

A reasonable correction here is to fix the cell values in front of the shock to the free-stream values. We can evaluate the quality of this fix later.
"""

# ╔═╡ 5c2db847-a2ec-4d36-bdf7-7ad2393f67f3
md"""
We can store the dual graph of the mesh in a graph data structure to aid in lookup of neighbor relationships and cell data.
"""

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

# ╔═╡ fd9b689f-275c-4c91-9b6c-4e63c68d6ab2
struct DualNodeKind{K} end

# ╔═╡ ead8c1a5-9f4e-4d92-b4ca-1650ad34bdca
const DUAL_NODE_TYPE = Tuple{
    DualNodeKind{S},
    Union{Nothing,SVector{4,Float64},CoarseQuadCell{Float64,3,12}},
} where {S}

# ╔═╡ 3a8cd7e2-fae9-4e70-8c92-004b17352506
md"""
## Solving for ``\dot x``

Each of the points on the shock can be used to define new cells ``P_i``. For each of the original cells, as well as the new cells, we know that:
```math
\oint_{\partial P_i} \tilde F(\bar u_i)\cdot\hat n\,ds = 0
```

We can write a system of equations for the $(num_coarse_cells) cells directly in front of the shock and their counterparts directly behind. For these $(2*num_coarse_cells) cells, we know the following:
- Its cell-average value ``\bar u`` and its tangent ``\dot{\bar u}``
- Its bounding polygon, made up of points ``p_k``
- ``\partial_{p_k}\bar u``

Which will yield ``i`` instances of the following equation:
```math
	\tilde F(\bar u_i)\hat n_{i,S}L_{i, S} + \tilde F(\bar u_i)\hat n_{i,E}L_{i, E} + \tilde F(\bar u_i)\hat n_{i,N}L_{i, N} + \tilde F(\bar u_i)\hat n_{i,W}L_{i, W} = 0
```

We can stack these ``i`` equations into ``\mathcal G``, and then use the implicit function theorem to compute :
```math
\begin{aligned}
0 &= \mathcal {G}(x_s, \bar u)\\
g(x_s) &= \bar u \\
\frac{\partial g_j(x_s)}{\partial x_k} &= \left[\frac{\partial G_i(x_s, \bar u)}{\partial u_j}\right]^{-1}\frac{\partial G_i(x_s, \bar u)}{\partial x_k}\\
\frac{\partial x_k}{\partial u_j} &= \left[\frac{\partial g_j(x_s)}{\partial x_k}\right]^{+}
\end{aligned}
```

Since ``\bar u``, ``x_s``, and ``\dot u`` are known, we can compute the dependence of the shock position on the initial parameters via:
```math
\dot x = \frac{\partial x_i}{\partial u_j}\dot u
```
"""

# ╔═╡ b88373c7-fe33-4373-b4ea-036688c0114c
md"""

Computing the shock shift using the IFT on ``\mathcal G(x, \bar{u})`` is impossible, since ``\mathcal G`` is not invertible. We might have a few horrible numerical hacks to get around this.

### Implicit Selection Theorem?
One coarse cell immediately to the right of the shock has 4 neighbors: the cell with value ``u_\infty`` on the left and its north, south, and east neighbors. The value ``u(p)`` is known in each of these cells, as is ``\partial_{p}u``.

This system can be transformed into Cartesian coordinates, since we only care about the length of the north and south edges. In fact, for the coarse cell at the symmetry boundary, we know the shock shift should be identical to the shock shift in its mirror neighbor. Since ``d\mathcal{G}(u, x)`` isn't invertible anyway, why not "zoom out" and take the local cell boundary integral ``\mathcal{F}(p, L):\mathbb{R}^{n_p\times n_L}\mapsto\mathbb{R}^4`` and try the Implicit Selection Theorem?
 - ``p`` is the vector of free-stream parameters
 - ``L`` is the vector of cell widths to the right of the shock
"""

# ╔═╡ 0a3a069c-e72c-4a47-9a11-00f049dc137c
const _dirs_scale = map(b -> b ? -1 : 1, Euler2D._dirs_bc_is_reversed)

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

# ╔═╡ fd73e7b8-5887-4fee-9d8c-c8df45e54d11
# we need this to match the syntax for ϕ_hll
# but we might want to choose other flux functions later...
begin
    ϕ = Euler2D.ϕ_hll
end

# ╔═╡ caa666e2-73fe-435e-a136-dc7fdcff03eb
begin
    # make this stuff diffpro-friendly 🔫
    _edge_length(edge_data) = 0
end

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
        dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
            u_L = project_state_to_orthonormal_basis(u, n̂)
            u_R = project_state_to_orthonormal_basis(other_data.u, n̂)
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

# ╔═╡ f3209f08-0d0c-4810-bf9b-86f2323799b4
begin
    function _compute_grad_ϕ_pts(
        cell_kind::DualNodeKind{:cell},
        cell_data,
        other_kind::DualNodeKind{:cell},
        other_data,
        edge_data,
    )
        return 0
    end
end

# ╔═╡ 54a7d1a4-beca-4e35-83a4-bc8f20381529
md"""
### Plan to correct:
1. Compute gradient in coarse cells correctly
    - What is the appropriate approximation? Linear interpolation of the derivative?
    - Deal with boundary conditions properly
2. Compute the gradient of the boundary integral correctly.
    - Replace with an exact Riemann solver instead of using HLL? How much error is acceptable?
3. Nail down the process of "local pseudoinversion".
    - If the current algebra is correct, which I think it is, fixing the gradient calcuations should work just fine
    - If the current algebra is wrong... oh no
4. What precisely constitutes an acceptable result?
"""

# ╔═╡ 4f79b51b-459a-46e1-a6d0-257ce08b029e
function boundary_integral(dual, id)
    nbrs = neighbor_labels(dual, id)
    return sum(nbrs) do nbr
        return _compute_ϕ(dual[id]..., dual[nbr]..., dual[id, nbr])
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

# ╔═╡ 4902bd21-2d6f-4b8d-84fe-e669d3dcb327
test_perturbation = [0., 1., 0.]

# ╔═╡ 654863a5-9fc5-461a-b5ee-a2ffb2379888
edge_lengths(poly) = (norm(a - b) for (a, b) ∈ zip(edge_ends(poly), edge_starts(poly)))

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

# ╔═╡ 5d77d782-2def-4b3a-ab3a-118bf8e96b6b
coarse_cells = let
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
        v1, v2, v3 = compute_coarse_cell_contents(cell, sim_with_ad, 6)
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
	foreach(idxs) do k
		v = d[k]
		@reset v.u = ambient_u
		@reset v.u̇ = ambient_u̇
		d[k] = v
	end
    d
end

# ╔═╡ c6e3873e-7fef-4c38-bf3f-de71f866057f
let
    xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
    yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
    spys = range(-0.25, 2.0; length = 20)
    spxs = x_shock.(spys)
    p = plot(
        spxs,
        spys;
        label = "Strong Shock Front (with extension)",
        lw = 4,
        legend = :outertop,
    )
    id = 0
    maxdensity = maximum(coarse_cells) do (_, c)
        c.u[1]
    end
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
        annotate!(p, v..., Plots.text(L"%$id", 6))
    end
    plot!(
        p,
        xc,
        yc;
        aspect_ratio = :equal,
        xlims = (-1.55, 0.05),
        ylims = (-0.2, 2.2),
        label = "Blunt Body",
        fill = true,
        dpi = 1000,
        size = (800, 1200),
        ls = :dash,
        lw = 4,
        fillalpha = 0.5,
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
        [-1.5, 0.0, 0.0],
        [2.0, 2.0, 0.75];
        label = "v.N. Boundary",
        lw = 6,
        ls = :dashdot,
        legendfontsize = 14,
        color = :orange,
    )
    #savefig(p, "../gfx/silly_rectangles.pdf")
    p
end

# ╔═╡ 7e9ac0e4-37d7-41d0-98a7-7284634cb404
coarse_dual = let
    g = MetaGraph(DiGraph(), Int, DUAL_NODE_TYPE, Tuple{Symbol,Point{Float64},Point{Float64}})
    g[1] = (DualNodeKind{:boundary_ambient}(), ambient_u)
    for (k, v) ∈ coarse_cells
        g[k] = (DualNodeKind{:cell}(), v)
    end
    phantom_idx = 10000 * nv(g) + 1
    for (k1, v1) ∈ coarse_cells
        for (k2, v2) ∈ coarse_cells
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

# ╔═╡ aa61b88c-417d-4598-96bb-3a1cb92fdb18
function boundary_integral_ddparams(dual, id)
    ddu = boundary_integral_ddu(dual, id)
    return sum(ddu) do (i, j, dϕ_dui, dϕ_duj)
        return dϕ_dui * coarse_dual[i][2].u̇ + dϕ_duj * coarse_dual[j][2].u̇
    end
end

# ╔═╡ 6906c857-88ba-4914-8554-c721ad7e87fe
begin
	const x_select = hcat(
    	SVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    	SVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
	);
	function boundary_integral_ddL(dual, id) 
		ddu = boundary_integral_ddu(dual, id)
    	return sum(ddu) do (i, j, dϕ_dui, dϕ_duj)
        	return (dϕ_dui * coarse_dual[i][2].du_dpts + dϕ_duj * coarse_dual[j][2].du_dpts) * x_select
    	end
	end
end

# ╔═╡ d0b579a8-38a4-45a9-b87b-7afa68fdb957
function ξ_local_pseudoinversion(dual, id, ε)
	A = boundary_integral_ddparams(dual, id)
	@assert rank(A) == min(size(A)...)
	B = boundary_integral_ddL(dual, id)
	C = pinv(A)*B
	return -C'*ε
end

# ╔═╡ 5383d7d3-b89e-4990-93d3-5548f3675738
rank(boundary_integral_ddparams(coarse_dual, 45)) # -dG(x_s, p)/dp

# ╔═╡ 0a7ea098-0fc8-4dc5-863e-ce91dc044b66
ξ_local_pseudoinversion(coarse_dual, 19, test_perturbation)

# ╔═╡ 0d04c757-5137-4683-b203-99819f70b84f
md"""
## Solving R-H
"""

# ╔═╡ 9c57268a-8d3d-4bfe-a565-2e78f8998c2f


# ╔═╡ cdbc037e-d542-47be-b78e-b396c5c3d8df
md"""
## Figures
"""

# ╔═╡ 69ef8869-182e-44ec-8fd4-cbc36fcb671b
let
	behind_shock_cells = (filter(keys(coarse_cells)) do k
		v = coarse_cells[k]
		pts = (edge_starts(cell_boundary_polygon(v)) |> collect)[[2, 3]]
		return all(p -> p[1] == x_shock(p[2]), pts)
	end |> collect |> sort)[2:end-1]
	infos = mapreduce(vcat, behind_shock_cells) do id
		sum(ξ_local_pseudoinversion(coarse_dual, id, [1., 0., 0.]))/2
	end
	infos2 = mapreduce(vcat, behind_shock_cells) do id
		sum(ξ_local_pseudoinversion(coarse_dual, id, [0., 1., 0.]))/2
	end
	v1 = 2.0/64
	ss = (range(-v1, 2.; length=length(infos)+1) .+ v1/2)[1:end-1]
	p1 = plot(ss, clamp.(infos, -100.0, 100.0), label=L"\xi_1(y)")
	p2 = plot(ss, clamp.(infos2, -100.0, 100.0), label=L"\xi_2(y)")
	plot(p1, p2, size=(900, 450), dpi=600)
end

# ╔═╡ ae7bca74-2a14-4a62-b6e9-669260fa4011
let
	f(y) = BilligShockParametrization.shock_front(y, ambient_primitives[2], 0.75)[1]
	y = 0.:0.01:2.0
	y2 = 0.:0.05:2.0
	x = f.(y)
	p = plot(x, y, label="Billig")
	plot!(p, x_shock.(y2), y2, label="Extracted")
end

# ╔═╡ b2840f27-70da-4d5a-a8f1-2ebcb7db2896
let
	ε = 0.001
	behind_shock_cells = (filter(keys(coarse_cells)) do k
		v = coarse_cells[k]
		pts = (edge_starts(cell_boundary_polygon(v)) |> collect)[[2, 3]]
		return all(p -> p[1] == x_shock(p[2]), pts)
	end |> collect |> sort)[2:end-1]
	infos_temp = mapreduce(vcat, behind_shock_cells) do id
		ξ_local_pseudoinversion(coarse_dual, id, [0., ε, 0.])
	end
	infos = (infos_temp[1:2:end] + infos_temp[2:2:end]) ./ 2
	info_ys = map(behind_shock_cells) do k
		v = coarse_cells[k]
		pts = reduce(hcat, edge_starts(cell_boundary_polygon(v)))
		return sum(pts'[2:3, 2]) / 2
	end
	f(y) = BilligShockParametrization.shock_front(y, ambient_primitives[2], 0.75)[1]
	df(y) = begin
		ForwardDiff.derivative(ambient_primitives[2]) do M
			BilligShockParametrization.shock_front(y, M, 0.75)[1]
		end
	end
	y = 0.:0.01:2.0
	y2 = 0.:0.05:2.0
	x = f.(y)
	p = plot(x, y, label="Billig "*L"(M=4.0)", title="Local Pseudoinversion; "*L"N_{coarse}=%$(num_coarse_cells_pos_y),\varepsilon=%$(ε)")
	@show length(infos), length(info_ys)
	plot!(p, x .+ ε .* df.(y), y, label="Billig "*L"(M = 4.0+\varepsilon)")
	plot!(p, x_shock.(y2), y2, label="Extracted via Shock Sensor")
	plot!(p, x_shock.(info_ys) .+ infos, info_ys, label="Extracted + Perturbation", marker=:circ)
	savefig(p, "../gfx/local_pseudoninversion_Ny$(num_coarse_cells_pos_y)eps$(ε)cellwidth$(cell_width_at_shock).pdf")
	p
end

# ╔═╡ a24be473-d2cd-46eb-bbab-0fb9766cbfae
md"""
- Find an explanation for the differing magnitudes of xi and d(billig)/dm
- Plot comparisons against Billig and (when available) against other grid sizes
- Parameter study on xi: does the oscillation in xi change if the grid size changes? (only need to coarsen)
"""

# ╔═╡ 1f174634-812d-4704-8e82-36935e8f0cb5
md"""
## Debug GFX and Information
"""

# ╔═╡ 273dade2-9bda-46d1-945d-70fc42f9bb1c
ξ_local_pseudoinversion(coarse_dual, 172, I)

# ╔═╡ b0a62334-fc75-4c51-b637-03c606f49910
let
	x = -1.5:0.01:-0.75
	y = 1
end

# ╔═╡ 256e1cef-2145-4ff8-bee1-d18bcd3ad75a
let
	polyA = SVector(0., 0., 0., 1., 1., 0.)
	polyB = SVector(0.5, -0.5, 0.5, 1., 1.5, 1., 1.5, -0.5)
	polyC = poly_intersection(polyA, polyB)
	J = intersection_point_jacobian(SVector(0.5, 0.5), polyB, polyA)
	(J, J*[0., 1., 0., 0., 0., 0., 0., 0.])
end

# ╔═╡ 48148c30-486b-4952-9c5f-aea722ff74cf
@bind n_cell Slider(31:145)

# ╔═╡ 8f36cc9d-c2c5-4bea-9dc7-e5412a2960f9
let
    n = n_cell
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
        _, c = nth_step(sim_with_ad, 2)
        data = mapreduce(pt -> pt', vcat, edge_starts(cell_boundary_polygon(c[id])))
        plot!(
            p,
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
    plot!(p, xc, yc; color = :black)
    data = mapreduce(vcat, all_cells_contained_by(empty_coarse[n].pts, sim_with_ad)) do id
        _, c = nth_step(sim_with_ad, 2)
        return Vector(c[id].center)'
    end
    scatter!(p, data[:, 1], data[:, 2]; marker = :x, ms = 2, label = false)
    p
end

# ╔═╡ 752ab770-eb30-4e3f-ba85-304b977aea02
coarse_dual[31][2].du_dpts * [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ╔═╡ b943979b-ed0d-45e1-b4fa-b3ab2f6a4acb
mapreduce(hcat, [(-1.0, 0.01), (-1.0, 0.02)]) do pt
	grad_x_u_at(sim_with_ad, 6, pt...)
end

# ╔═╡ 768fa7ca-a8a3-49fc-ad6b-c47e68e71237
heatmap(-1.25:0.01:-0.75, 0.0:0.01:1.0, (x, y)->grad_x_u_at(sim_with_ad, 6, x, y)[1,1])

# ╔═╡ 2501fa77-dadb-428d-b98d-58b050e21a7d
heatmap(-1.25:0.01:-0.75, 0.0:0.01:1.0, (x, y)->grad_x_u_at(sim_with_ad, 6, x, y)[1,2])

# ╔═╡ 5b7a3783-ef40-468f-93ac-91cb46929bd6
ne(coarse_dual)

# ╔═╡ 74525445-19f6-471f-878e-a60f07ba9f01
nv(coarse_dual)

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
Euler2D = "c24a2923-03cb-4692-957a-ccd31f2ad327"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MetaGraphsNext = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
PlanePolygons = "f937978b-597e-4162-b50e-ad07b6cf7ab2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ShockwaveProperties = "77d2bf28-a3e9-4b9c-9fcf-b85f74cc8a50"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
Accessors = "~0.1.42"
DifferentiationInterface = "~0.7.0"
Euler2D = "~0.5.2"
ForwardDiff = "~0.10.38"
Graphs = "~1.12.1"
Interpolations = "~0.15.1"
LaTeXStrings = "~1.4.0"
MetaGraphsNext = "~0.7.3"
Mooncake = "~0.4.118"
OhMyThreads = "~0.8.3"
PlanePolygons = "~0.1.12"
Plots = "~1.40.13"
PlutoUI = "~0.7.62"
ShockwaveProperties = "~0.2.6"
StaticArrays = "~1.9.13"
Unitful = "~1.22.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "a74a1ff9a38867efb2f9181362a4afeb0b1ce8d0"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

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

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "83881aca52d132932f1827e2571e7d703ab26aff"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.0"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.Euler2D]]
deps = ["Accessors", "Dates", "ForwardDiff", "LinearAlgebra", "ShockwaveProperties", "StaticArrays", "Tullio", "Unitful", "UnitfulChainRules"]
git-tree-sha1 = "589f7a68799b5db135f349b7c85f41adde25420b"
uuid = "c24a2923-03cb-4692-957a-ccd31f2ad327"
version = "0.5.2"

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

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

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

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "3169fd3440a02f35e549728b0890904cfd4ae58a"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.1"

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

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "8e071648610caa2d3a5351aba03a936a0c37ec61"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.13"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

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
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "1e3b196ecbbf221d4d3696ea9de4288bea4c39f9"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MistyClosures]]
git-tree-sha1 = "1142aefd845c608f3c70e4c202c4aae725cab67b"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.0.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Mooncake]]
deps = ["ADTypes", "ChainRules", "ChainRulesCore", "DiffRules", "ExprTools", "FunctionWrappers", "GPUArraysCore", "Graphs", "InteractiveUtils", "LinearAlgebra", "MistyClosures", "Random", "Test"]
git-tree-sha1 = "a9122d7a195f712ed56d9e0d399e29547c075c69"
uuid = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
version = "0.4.118"

    [deps.Mooncake.extensions]
    MooncakeAllocCheckExt = "AllocCheck"
    MooncakeCUDAExt = "CUDA"
    MooncakeFluxExt = "Flux"
    MooncakeJETExt = "JET"
    MooncakeLuxLibExt = "LuxLib"
    MooncakeLuxLibSLEEFPiratesExtension = ["LuxLib", "SLEEFPirates"]
    MooncakeNNlibExt = "NNlib"
    MooncakeSpecialFunctionsExt = "SpecialFunctions"

    [deps.Mooncake.weakdeps]
    AllocCheck = "9b6a8646-10ed-4001-bbdc-1d2f46dfbb1a"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
    LuxLib = "82251201-b29d-42c6-8e01-566dec8acb11"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

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

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "e0a1a8b92f6c6538b2763196f66417dddb54ac0c"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.3"
weakdeps = ["Markdown"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

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

[[deps.PlanePolygons]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "943170aca816b869b0f965e494ab346bad3e3631"
uuid = "f937978b-597e-4162-b50e-ad07b6cf7ab2"
version = "0.1.12"
weakdeps = ["Mooncake"]

    [deps.PlanePolygons.extensions]
    MooncakeExt = "Mooncake"

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

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShockwaveProperties]]
deps = ["LinearAlgebra", "StaticArrays", "Unitful"]
git-tree-sha1 = "e1e1e5d9176e0601059812fe2ebcb60e4d796639"
uuid = "77d2bf28-a3e9-4b9c-9fcf-b85f74cc8a50"
version = "0.2.6"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

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

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.TaskLocalValues]]
git-tree-sha1 = "d155450e6dff2a8bc2fcb81dcb194bd98b0aeb46"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.2"

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
git-tree-sha1 = "d62610ec45e4efeabf7032d67de2ffdea8344bed"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

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

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

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
# ╠═6f1542ea-a747-11ef-2466-fd7f67d1ef2c
# ╠═2e9dafda-a95c-4277-9a7c-bc80d97792f0
# ╠═15c5d10c-4d9a-4f57-a8fd-b65a970ac73c
# ╠═f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
# ╟─e7001548-a4b4-4709-b10c-0633a11bd624
# ╟─c87b546e-8796-44bf-868c-b2d3ad340aa1
# ╠═4267b459-7eb7-4678-8f06-7b9deab1f830
# ╟─2716b9b5-07fd-4175-a83e-22be3810e4b3
# ╠═afc11d27-1958-49ba-adfa-237ba7bbd186
# ╟─100a1f91-120b-43a4-8486-8f9e64b8b71e
# ╟─3fe1be0d-148a-43f2-b0a5-bb177d1c041d
# ╠═0df888bd-003e-4b49-9c2a-c28a7ccc33d2
# ╟─136ab703-ae33-4e46-a883-0ed159360361
# ╟─a9d31e2d-fc4b-4fa5-9015-eb2ac2a3df5d
# ╟─c1a81ef6-5e0f-4ad5-8e73-e9e7f09cefa6
# ╟─e55363f4-5d1d-4837-a30f-80b0b9ae7a8e
# ╟─d832aeb4-42d6-4b72-88ee-4cdd702a4f48
# ╠═90bf50cf-7254-4de8-b860-938430e121a9
# ╠═fffcb684-9b58-43d7-850a-532c609c5389
# ╟─33e635b3-7c63-4b91-a1f2-49da93307f29
# ╠═4dc7eebf-48cc-4474-aef0-0cabf1d8eda5
# ╟─8bd1c644-1690-46cf-ac80-60654fc6d8c0
# ╟─893ec2c8-88e8-4d72-aab7-88a1efa30b47
# ╠═f6147284-02ec-42dd-9c2f-a1a7534ae9fa
# ╟─d14c3b81-0f19-4207-8e67-13c09fd7636a
# ╠═cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
# ╠═2e3b9675-4b66-4623-b0c4-01acdf4e158c
# ╟─d5db89be-7526-4e6d-9dec-441f09606a04
# ╟─4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
# ╟─bcdd4862-ac68-4392-94e2-30b1456d411a
# ╟─e2bdc923-53e6-4a7d-9621-4d3b356a6e41
# ╟─44ff921b-09d0-42a4-8852-e911212924f9
# ╟─4f8b4b5d-58de-4197-a676-4090912225a1
# ╟─6e4d2f60-3c40-4a2b-be2b-8c4cc40fb911
# ╟─706146ae-3dbf-4b78-9fcc-e0832aeebb28
# ╟─9b6ab300-6434-4a96-96be-87e30e35111f
# ╟─21cbdeec-3438-4809-b058-d23ebafc9ee2
# ╟─90ff1023-103a-4342-b521-e229157001fc
# ╟─5c0be95f-3c4a-4062-afeb-3c1681cae549
# ╟─88889293-9afc-4540-a2b9-f30afb62b1de
# ╠═6da05b47-9763-4d0c-99cc-c945630c770d
# ╟─351d4e18-4c95-428e-a008-5128f547c66d
# ╠═a974c692-5171-4049-95aa-def2c776061b
# ╠═4db61526-1e3a-47ac-8e2f-66634c3947c2
# ╟─bc0c6a41-adc8-4d18-9574-645704f54b72
# ╟─4a5086bc-5c36-4e71-9d3a-8f77f48a90f9
# ╠═747f0b67-546e-4222-abc8-2007daa3f658
# ╠═2cb33e16-d735-4e60-82a5-aa22da0288fb
# ╠═4b036b02-1089-4fa8-bd3a-95659c9293cd
# ╟─24da34ca-04cd-40ae-ac12-c342824fa26e
# ╟─9ac61e80-7d6a-40e8-8254-ee306f8248c3
# ╟─92044a9f-2078-48d1-8181-34be87b03c4c
# ╟─5268fe37-b827-42ad-9977-6efbf4ecfad1
# ╟─e98df040-ac22-4184-95dd-a8635ab72af6
# ╠═31009964-3f32-4f97-8e4a-2b95be0f0037
# ╠═4d202323-e1a9-4b24-b98e-7d17a6cc144f
# ╟─95947312-342f-44b3-90ca-bd8ad8204e18
# ╟─eb5a2dc6-9c7e-4099-a564-15f1cec11caa
# ╟─9c601619-aaf1-4f3f-b2e2-10422d8ac640
# ╟─e0a934d6-3962-46d5-b172-fb970a537cc0
# ╠═62ebd91b-8980-4dc5-b61b-ba6a21ac357d
# ╠═be8ba02d-0d31-4720-9e39-595b549814cc
# ╟─a1dc855f-0b24-4373-ba00-946719d38d95
# ╟─93043797-66d1-44c3-b8bb-e17deac70cfa
# ╠═0511fbcd-8b44-481c-a176-c0657b6557c2
# ╠═7468fbf2-aa57-4505-934c-baa4dcb646fc
# ╟─50a46d6d-deb7-4ad0-867a-4429bf55632f
# ╠═766b440b-0001-4037-8959-c0b7f04d999e
# ╟─d44322b1-c67f-4ee8-b168-abac75fb42a1
# ╟─2f088a0c-165e-47f9-aaeb-6e4ab31c9d26
# ╟─24fa22e6-1cd0-4bcb-bd6d-5244037e58e2
# ╟─cd312803-3819-4451-887b-ce2b53bb6e1b
# ╟─7c8cde49-b9c6-4889-a328-bf46b6f82a01
# ╟─ac412980-1013-450f-bb23-0dc7c2b3f199
# ╟─0d6ae7cf-edae-48f6-a257-6223563b7c76
# ╟─2041d463-5382-4c38-bf52-23d22820ac59
# ╟─bfe8cb7d-8e5e-4dda-88cb-356b34017335
# ╟─dea032e2-bf23-42da-8dac-d3368c2bdec6
# ╟─fc646016-a30e-4c60-a895-2dde771f79cb
# ╠═54ed2abb-81bb-416c-b7be-1125e41622f5
# ╟─c6e3873e-7fef-4c38-bf3f-de71f866057f
# ╟─435887e7-1870-4677-a09d-020be5761039
# ╟─61945d45-03f4-4401-8b18-3d11420047d0
# ╟─729ebc48-bba1-4858-8369-fcee9f133ee0
# ╟─5cffaaf5-9a5e-4839-a056-30e238308c51
# ╟─f252b8d0-f067-468b-beb3-ff6ecaeca722
# ╟─571b1ee7-bb07-4b30-9870-fbd18349a2ef
# ╟─80cde447-282a-41e5-812f-8eac044b0c15
# ╟─48a2b845-c466-4bbc-aa16-46a95ed7be35
# ╟─f30619a3-5344-4e81-a4b5-6a11100cd056
# ╟─6c2b1a68-dd43-4449-9dc7-4b7849081cc3
# ╠═37eb63be-507a-475f-a6f6-8606917b8561
# ╠═a968296a-43c1-48f3-b4b8-0d81cb162b7b
# ╠═5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
# ╟─6f303343-1f76-46f9-80e8-2dd4ae1b5427
# ╟─d87e0bb8-317e-4d48-8008-a7071c74ab31
# ╟─5d77d782-2def-4b3a-ab3a-118bf8e96b6b
# ╟─6aaf12a0-c2d9-48ab-9e13-94039cf95258
# ╟─5c2db847-a2ec-4d36-bdf7-7ad2393f67f3
# ╠═0679a676-57fa-45ee-846d-0a8961562db3
# ╟─d19fff76-e645-4d9d-9989-50019b6356ad
# ╟─0e0a049b-e2c3-4fe9-8fb8-186cdeb60485
# ╠═fd9b689f-275c-4c91-9b6c-4e63c68d6ab2
# ╟─ead8c1a5-9f4e-4d92-b4ca-1650ad34bdca
# ╟─7e9ac0e4-37d7-41d0-98a7-7284634cb404
# ╟─3a8cd7e2-fae9-4e70-8c92-004b17352506
# ╟─b88373c7-fe33-4373-b4ea-036688c0114c
# ╠═e4b54bd3-5fa9-4291-b2a4-6b10c494ce34
# ╠═0a3a069c-e72c-4a47-9a11-00f049dc137c
# ╟─8a76792c-6189-4d39-9147-5a7ea9b074f9
# ╟─fd73e7b8-5887-4fee-9d8c-c8df45e54d11
# ╟─caa666e2-73fe-435e-a136-dc7fdcff03eb
# ╟─63c1d6d9-ad64-4074-83c4-40b5df0e0b1f
# ╟─cec776ee-f81d-4457-8555-24eaf80e4cca
# ╟─df5a5737-41e7-447d-ad5f-ebdbf07996ca
# ╟─3c835380-3580-4881-a0ad-e466eb99fcb8
# ╠═f3209f08-0d0c-4810-bf9b-86f2323799b4
# ╟─54a7d1a4-beca-4e35-83a4-bc8f20381529
# ╠═4f79b51b-459a-46e1-a6d0-257ce08b029e
# ╠═03f640b6-851c-4c57-9ff8-5549415b558b
# ╠═aa61b88c-417d-4598-96bb-3a1cb92fdb18
# ╠═6906c857-88ba-4914-8554-c721ad7e87fe
# ╠═5383d7d3-b89e-4990-93d3-5548f3675738
# ╠═d0b579a8-38a4-45a9-b87b-7afa68fdb957
# ╠═4902bd21-2d6f-4b8d-84fe-e669d3dcb327
# ╠═0a7ea098-0fc8-4dc5-863e-ce91dc044b66
# ╠═654863a5-9fc5-461a-b5ee-a2ffb2379888
# ╟─6822601e-f40a-4fa1-a0f8-a8bb09809549
# ╠═0d04c757-5137-4683-b203-99819f70b84f
# ╠═9c57268a-8d3d-4bfe-a565-2e78f8998c2f
# ╟─cdbc037e-d542-47be-b78e-b396c5c3d8df
# ╠═e5036dd3-9070-4521-9d7d-e0293b967d78
# ╠═69ef8869-182e-44ec-8fd4-cbc36fcb671b
# ╠═ae7bca74-2a14-4a62-b6e9-669260fa4011
# ╠═b2840f27-70da-4d5a-a8f1-2ebcb7db2896
# ╟─a24be473-d2cd-46eb-bbab-0fb9766cbfae
# ╟─1f174634-812d-4704-8e82-36935e8f0cb5
# ╠═273dade2-9bda-46d1-945d-70fc42f9bb1c
# ╠═b0a62334-fc75-4c51-b637-03c606f49910
# ╠═256e1cef-2145-4ff8-bee1-d18bcd3ad75a
# ╟─8f36cc9d-c2c5-4bea-9dc7-e5412a2960f9
# ╠═48148c30-486b-4952-9c5f-aea722ff74cf
# ╠═752ab770-eb30-4e3f-ba85-304b977aea02
# ╠═b943979b-ed0d-45e1-b4fa-b3ab2f6a4acb
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
