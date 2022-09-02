## Problem: Job sequencing and tool switching problem (SSP) - da Silva, Chaves, and Yanasse (2020) multi-commodity flow formulation
## Solver: Gurobi
## Language: Julia (JuMP)
## Written by: @setyotw
## Date: Sept 2, 2022

#%% import packages
using Pkg, JuMP, Gurobi, DataStructures
Pkg.status()
import FromFile: @from

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
#function UniformSSP_daSilva_Formulation(instanceSSP, magazineCap, MILP_Limit)
    # 1 | initialize sets and notations
    # number of available jobs (horizontal)
    n = length(instanceSSP[1,:])
    # number of available tools (vertical)
    m = length(instanceSSP[:,1])

    J = [i for i in range(1, n)] ## list of jobs
    T = [i for i in range(1, m)] ## list of tools
    V = [i for i in range(0, n+2)] ## list of tools
    
    # creating arcs
    A = []
    for i in range(0,n+1)
        append!(A, [(i,i+1)])
    end
    for i in range(0,n-1)
        append!(A, [(i,n+2)])
    end
    for i in range(1,n)
        append!(A, [(n+2,i)])
        append!(A, [(i,n+2)])
    end
        
    arcIK = [(i,k) for i in J for k in J if i!=k]
    arcIKT = [(i,k,t) for i in V for k in V if i!=k for t in T]
    
    Tj = Dict((j) => [] for j in J)

    for job in J
        for tools in T
            if instanceSSP[tools,job] == 1
                append!(Tj[job], tools)
            end
        end
    end

    # 2 | initialize parameters
    C = Int(magazineCap)
    
    # 3 | initialize the model
    model = Model(Gurobi.Optimizer)

    # 4 | initialize decision variables
    @variable(model, X[arcIK], Bin) # X[ik]
    @variable(model, Y[arcIKT], Bin) # Y[ikt]
    
    # 5 | define objective function
    @objective(model, Min, 
        #sum(Y[(i,(n+1),t)] for t in T for i in range(1,n-1)) + sum(Y[(i,(n+2),t)] for t in T for i in range(1,n-2)))
        sum(Y[(0,1,t)] for t in T) + sum(Y[(i,(n+1),t)] for t in T for i in range(1,n-1)) + sum(Y[(i,(n+2),t)] for t in T for i in range(1,n-2)))

    # 6 | define constraints
    for i in J
        @constraint(model, sum(X[(i,k)] for k in J if i!=k) == 1)
    end

    for k in J
        @constraint(model, sum(X[(i,k)] for i in J if i!=k) == 1)
    end
        
    for t in T
        @constraint(model, Y[(0,1,t)] + Y[(0,(n+2),t)] == 1)
    end

    for i in range(1,n-2)
        for t in T
            @constraint(model, Y[((i-1),i,t)] + Y[((n+2),i,t)] - Y[(i,(n+1),t)] - Y[(i,(i+1),t)] - Y[(i,(n+2),t)] == 0)
        end
    end
        
    for t in T
        @constraint(model, Y[((n-2),(n-1),t)] + Y[((n+2),(n-1),t)] - Y[((n-1),n,t)] - Y[((n-1),(n+1),t)] == 0)
    end

    for t in T
        @constraint(model, Y[((n-1),n,t)] - Y[(n,(n+1),t)] == 0)
    end

    for t in T
        @constraint(model, sum(Y[(i,(n+1),t)] for i in J) == 1)
    end

    for t in T
        @constraint(model, sum(Y[(i,(n+2),t)] for i in range(0,n-2)) - sum(Y[((n+2),i,t)] for i in range(1,n-1)) == 0)
    end

    for (i,k) in arcIK
        for t in Tj[i]
            @constraint(model, X[(i,k)] <= Y[((k-1),k,t)])
        end
    end

    for k in J
        @constraint(model, sum(Y[((k-1),k,t)] for t in T) == C)
    end
        
    # 7 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    JuMP.set_time_limit_sec(model, MILP_Limit)
    JuMP.optimize!(model)

    # 8 | extract the results    
    completeResults = solution_summary(model)
    solutionObjective = objective_value(model)
    solutionGap = relative_gap(model)
    runtimeCount = solve_time(model)
    all_var_list = all_variables(model)
    all_var_value = value.(all_variables(model))
    X_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'X']
    Y_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'Y']
    
    return solutionObjective, solutionGap, X_active, Y_active, runtimeCount, completeResults
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple uniform SSP case with 5 different jobs, 6 different tools, and 3 capacity of magazine (at max, only 3 different tools could be installed at the same time)
instanceSSP = Array{Int}([
        1 1 0 0 1;
        1 0 0 1 0;
        0 1 1 1 0;
        1 0 1 0 1;
        0 0 1 1 0;
        0 0 0 0 1])

initialSetupSSP = Array{Int}([2 1 1 2 3 2 3 3 2 3 1 3 3 2 2 2 3 3 1 1])

magazineCap = Int(3)

#%% termination time for the solver (Gurobi)
MILP_Limit = Int(3600)

#%% implement the mathematical formulation
# solutionObjective --> best objective value found by the solver
# solutionGap --> solution gap, (UB-LB)/UB
# U_active, V_active, W_active --> return the active variables
# runtimeCount --> return the runtime in seconds
# completeResults --> return the complete results storage
solutionObjective, solutionGap, X_active, Y_active, runtimeCount, completeResults = UniformSSP_daSilva_Formulation(instanceSSP, magazineCap, MILP_Limit)