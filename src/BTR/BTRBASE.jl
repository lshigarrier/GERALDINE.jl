struct BasicTrustRegion{T<:Real}
    η1::T
    η2::T
    γ1::T
    γ2::T
end

function BTRDefaults()
    return BasicTrustRegion(0.01, 0.9, 0.5, 0.5)
end

mutable struct BTRState{T} <: AbstractState where T
    iter::Int64
    start::Int64
    x::Vector
    xcand::Vector
    g::Vector
    H::T
    step::Vector
    Δ::Float64
    ρ::Float64
    fx::Float64
    function BTRState(H::T) where T
        state = new{T}()
        state.H = H
        return state
    end
end

import Base.println

function println(state::BTRState)
    println(round.(state.x, digits = 3))
end
function acceptCandidate!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η1
        return true
    else
        return false
    end
end

function updateRadius!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η2
        stepnorm = norm(state.step)
        state.Δ = min(10e12, max(4*stepnorm, state.Δ))
    elseif state.ρ >= b.η1
        state.Δ *= b.γ2
    else
        state.Δ *= b.γ1
    end
end

function stopCG(normg::Float64, normg0::Float64, k::Int, kmax::Int)
    χ::Float64 = 0.1
    θ::Float64 = 0.5
    if (k == kmax) || (normg <= normg0*min(χ, normg0^θ))
        return true
    else
        return false
    end
end

function TruncatedCG(state::BTRState)
    H = state.H
    g = state.g
    Δ = state.Δ*state.Δ
    n = length(g)
    s = zeros(n)
    normg0 = norm(g)
    v = g
    d = -v
    gv = dot(g, v)
    norm2d = gv
    norm2s = 0
    sMd = 0
    k = 0
    while ! stopCG(norm(g), normg0, k, n)
        Hd = H*d
        κ = dot(d, Hd)
        if κ <= 0
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        α = gv/κ
        norm2s += α*(2*sMd+α*norm2d)
        if norm2s >= Δ
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        s += α*d
        g += α*Hd
        v = g
        newgv = dot(g, v)
        β = newgv/gv
        gv = newgv
        d = -v+β*d
        sMd = β*(sMd+α*norm2d)
        norm2d = gv+β*β*norm2d
        k += 1
    end
    return s
end

function TCG_LBHHH(state::BTRState, Hx::Function)
    g = state.g
    Δ = state.Δ*state.Δ
    n = length(g)
    s = zeros(n)
    normg0 = norm(g)
    v = g
    d = -v
    gv = dot(g, v)
    norm2d = gv
    norm2s = 0
    sMd = 0
    k = 0
    while ! stopCG(norm(g), normg0, k, n)
        Hd = Hx(d)
        κ = dot(d, Hd)
        if κ <= 0
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        α = gv/κ
        norm2s += α*(2*sMd+α*norm2d)
        if norm2s >= Δ
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        s += α*d
        g += α*Hd
        v = g
        newgv = dot(g, v)
        β = newgv/gv
        gv = newgv
        d = -v+β*d
        sMd = β*(sMd+α*norm2d)
        norm2d = gv+β*β*norm2d
        k += 1
    end
    return s
end
