module NonlinearSolveSpeedMappingExt

using SpeedMapping: speedmapping

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, SpeedMappingJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::SpeedMappingJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        maxtime = nothing, store_trace::Val = Val(false),
        termination_condition = nothing, kwargs...
	)

    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    m!, u, resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0, make_fixed_point = Val(true)
    )
    abstol = NonlinearSolveBase.get_tolerance(abstol, eltype(u)) # Changed tol to abstol

    time_limit = ifelse(maxtime === nothing, 1000, maxtime)

    sol = speedmapping(
		u; m!, abstol = abstol, pnorm = Inf, iter_limit = maxiters, 
		store_trace = store_trace isa Val{true}, time_limit, f = alg.f, algo = alg.algo, 
		lags = alg.lags, condition_max = alg.condition_max, relax_default = alg.relax_default, 
		ada_relax = alg.ada_relax, composite = alg.composite, maps_limit = alg.maps_limit, 
		reltol_resid_grow = alg.reltol_resid_grow, abstol_obj_grow = alg.abstol_obj_grow, 
		lower = alg.lower, upper = alg.upper, buffer = alg.buffer
    )

    res = prob.u0 isa Number ? first(sol.minimizer) : sol.minimizer
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, res)
    nsolve = alg.algo == :aa ? sol.iterations - 1 : 0 # The :aa algorithm neds sol.iterations - 1 linear solves during the algorithm. :acx needs none.

    return SciMLBase.build_solution(
        prob, alg, res, resid;
        original = sol, stats = SciMLBase.NLStats(sol.maps, 0, 0, nsolve, sol.iterations), 
        retcode = ifelse(sol.status == :first_order, ReturnCode.Success, ReturnCode.Failure)
    )
end

end
