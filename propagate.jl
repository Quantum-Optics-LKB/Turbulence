using DifferentialEquations
# using CUDA

function propagate_fwm(A, z0)
    # send arrays to GPU
    # A_gpu = cu(A) 
    @inbounds @fastmath function f!(dA, A, p, t)
        # laplacien plus abs + termes diag dans le n2
        for x = 2:size(dA,2)-1 # compute ∇²f in interior of f
            Threads.@threads for z = 1:size(dA, 1)
                dA[z,x] = -0.5*(1/k)*(dA[z,x-1] + dA[z,x+1] + dA[z,x] + dA[z,x] - 4*dA[z,x])/(delta_X*delta_Y)
            end
        end
        
        dA .+= (- alpha/2 .+ im*k*n2*c*epsilon_0*abs.(A).^2).*A
    end
    prob = ODEProblem(f!, A, (0.0f0, Float32(z0)))
    sol = solve(prob, TRBDF2(autodiff=false), atol=1e-6, rtol=1e-3)
    return Array(sol.u)
end