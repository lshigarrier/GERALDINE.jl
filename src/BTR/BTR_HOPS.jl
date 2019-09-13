function btr_HOPS(f::Function, g!::Function, Hx::Function, state::BTRState, x0::Vector, tTest::Function; 
        verbose::Bool = false, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, time_tol::Float64 = 120.0,
        accumulate!::Function, accumulator::Array)
    
    b = BTRDefaults()
    state.fx = f(x0)
    if verbose
        println(state.fx)
    end
    g!(x0, state.g, state.H)
    state.Δ = 0.1*norm(state.g)
    state.step = x0
    state.ρ = 1.0
    state.start = time_ns()
    
    while !Stop_optimize_mod(state, b, tTest, nmax = nmax, tol = epsilon, time_tol = time_tol)
        state.start = time_ns()
        accumulate!(state, accumulator)
        if verbose
            println(state.iter+1)
            #println(state)
        end
        state.step = TCG_HOPS(state, Hx)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.step, state.g)+0.5*dot(state.step, Hx(state.step)))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            g!(state.x, state.g, state.H)
            state.fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
        if verbose
            println(state.fx)
            println("$((time_ns()-state.start)/1e9) s")
        end
    end
    return state, accumulator
end

function OPTIM_btr_HOPS(f::Function, g_score!::Function, x0::Vector, weights::Vector;
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-4,
        time_tol::Float64 = 1e4, tTest::Function = par -> false)

    function accumulate!(state::BTRState, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []

    n = sum(weights)
    inds = length(weights)
    S = [zeros(length(x0)) for i = 1:inds]
    state = BTRState(S)
    state.x = copy(x0)
    state.iter = 0
    state.g = zeros(length(x0))

    Hx(x::Vector) = (1/n)*sum(weights[i]*dot(state.H[i], x)*state.H[i] for i in 1:inds)
        
    state, accumulator = btr_HOPS(f, g_score!, Hx, state, x0, tTest,
                verbose = verbose, nmax = nmax, epsilon = epsilon, time_tol = time_tol,
                accumulate! = accumulate!, accumulator = accumulator)
    return state, accumulator
end
