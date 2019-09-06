"""
'Stop_optimize(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)'
robust stoping criteria
"""

function Stop_optimize(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)
    if k == nmax
        return true
    end
    for i in 1:length(x)
        if abs(grad[i]*max(x[i], typX[i])/max(value, typVal)) > tol
            return false
        end
    end
    return true
end

function Stop_optimize_mod(state::BTRState, b::BasicTrustRegion, tTest::Function,
                typVal::Float64 = 1.0, typX::Vector = ones(length(state.g));
                tol::Float64 = 1e-4, nmax::Int64 = 500, time_tol::Float64 = 120)
    res = tTest(state)
    if res
        println("Insufficient progress")
        return true
    end
    if abs(time_ns() - state.start)/1e9 > time_tol
        println("Maximum iteration time exceeded")
        return true
    end
    if state.iter == nmax
        println("Maximum number of iterations")
        return true
    end
    for i in 1:length(state.x)
        if abs(state.g[i]*max(state.x[i], typX[i])/max(state.fx, typVal)) > tol
            return false
        end
    end
    println("First-order condition")
    return true
end
