using DifferentialEquations
# using CUDA

function propagate_fwm(A, z0)
    # send arrays to GPU
    # A_gpu = cu(A)
    function lap(f)
        laplacian = zero(f) # initialize to zero since we don't touch the boundaries
        for y = 2:size(f,2)-1, x = 2:size(f,1)-1 # compute ∇²f in interior of f
            laplacian[x,y] = (f[x-1,y] + f[x+1,y] + f[x,y-1] + f[x,y+1] - 4f[x,y])/(delta_X*delta_Y)
        end
        return laplacian
     end
        
    # @inbounds begin
    # @fastmath begin
    # function f!(dA, A, p, t)
    #     dA[1, :, :] .= -0.5*(1/k)*lap(A[1, :, :]) -alpha/2 * A[1, :, :] + im*k*n2*c*epsilon_0 * (
    #         abs.(A[1, :, :]).^2 + 2*abs.(A[2, :, :]).^2 + 2*abs.(A[3, :, :]).^2).*A[1, :, :] +
    #         2*A[3, :, :]*A[2, :, :]*conj.(A[1, :, :])
    #     dA[2, :, :] .= -0.5*(1/k)*lap(A[2, :, :]) -alpha/2 * A[2, :, :] + im*k*n2*c*epsilon_0 * (
    #             abs.(A[2, :, :]).^2 + 2*abs.(A[1, :, :]).^2 + 2*abs.(A[3, :, :]).^2).*A[2, :, :] +
    #             A[1, :, :].^2 * conj.(A[3, :, :])
    #     dA[3, :, :] .= -0.5*(1/k)*lap(A[3, :, :]) -alpha/2 * A[3, :, :] + im*k*n2*c*epsilon_0 * (
    #             abs.(A[3, :, :]).^2 + 2*abs.(A[1, :, :]).^2 + 2*abs.(A[2, :, :]).^2).*A[3, :, :] +
    #             A[1, :, :].^2 * conj.(A[2, :, :])
    # end
    # end
    # end
    @inbounds begin
        @fastmath begin
        function f!(dA, A, p, t)
            for j = 1:size(A, 2)
                Threads.@threads  for i = 1:size(A, 1)
                    dA[1, i, j] = -0.5*(1/k)*lap(A[1, i, j]) -alpha/2 * A[1, i, j] + im*k*n2*c*epsilon_0 * (
                        abs(A[1, i, j])^2 + 2*abs(A[2, i, j])^2 + 2*abs(A[3, i, j])^2)*A[1, i, j] +
                        2*A[3, i, j]*A[2, i, j]*conj(A[1, i, j])
                    dA[2, i, j] = -0.5*(1/k)*lap(A[2, i, j]) -alpha/2 * A[2, i, j] + im*k*n2*c*epsilon_0 * (
                            abs(A[2, i, j])^2 + 2*abs(A[1, i, j])^2 + 2*abs(A[3, i, j])^2)*A[2, i, j] +
                            A[1, i, j]^2 * conj(A[3, i, j])
                    dA[3, i, j] = -0.5*(1/k)*lap(A[3, i, j]) -alpha/2 * A[3, i, j] + im*k*n2*c*epsilon_0 * (
                            abs(A[3, i, j])^2 + 2*abs(A[1, i, j])^2 + 2*abs(A[2, i, j])^2)*A[3, i, j] +
                            A[1, i, j]^2 * conj(A[2, i, j])
                end
            end
        end
        end
    end
    prob = ODEProblem(f!, A, (0.0f0, Float32(z0)))
    sol = solve(prob, TRBDF2(), atol=1e-10, rtol=1e-8)
    return Array(sol.u)
end