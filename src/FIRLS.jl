module FIRLS
"""
https://cnx.org/contents/6x7LNQOp@7/Linear-Phase-Fir-Filter-Design-By-Least-Squares
https://www.dsprelated.com/Plishowarticle/808.php
https://eeweb.engineering.nyu.edu/iselesni/EL713/zoom/linphase.pdf
"""
# using LinearAlgebra
using LinearAlgebra
import LinearAlgebra: I, Diagonal, UniformScaling

export firls_design, freqz

function validate_inputs(filter_order, bands_D, D, fs)
    @assert filter_order >= 0 "Filter order cannot be negative."
    @assert fs > 0 "Sampling frequency should be larger than 0 Hz"
    validate_bands_D(bands_D, D, fs)
end
function validate_inputs(filter_order, bands_DW, D, W, fs)
    @assert filter_order >= 0 "Filter order cannot be negative."
    @assert fs > 0 "Sampling frequency should be larger than 0 Hz"
    validate_bands_D(bands_DW, D, fs)
    validate_bands_W(bands_DW, W, fs)
end
function validate_inputs(filter_order, bands_D, D, bands_W, W, fs)
    @assert filter_order >= 0 "Filter order cannot be negative."
    @assert fs > 0 "Sampling frequency should be larger than 0 Hz"
    validate_bands_D(bands_D, D, fs)
    validate_bands_W(bands_W, W, fs)
end

function validate_bands_D(bands_D, D, fs)
    validate_bands(bands_D, fs)
    @assert size(bands_D,2) == size(D,2) == 2 "Frequency bands and desired response should be N x 2 matrices"
    @assert size(bands_D,1) == size(D,1) "Frequency bands and desired response should be N x 2 matrices"
end

function validate_bands_W(bands_W, W, fs)
    validate_bands(bands_W, fs)
    @assert size(bands_W,2) == size(W,2) == 2 "Frequency band and weight matrix should be N x 2."
    @assert size(bands_W,1) == size(W,1) "Number of frequency bands should match number of weights."
end

function validate_bands(fbands::Matrix{T}, fs) where T
    @assert fbands[1] == 0 "Frequency bands should start at 0."
    @assert fbands[end] == fs * (1//2) "Frequency bands should end at fs/2"
    @views @assert !any((fbands[2:end,1] .- fbands[1:end-1,2]) .> zero(T)) "Frequency bands should cover entire [0,fs/2] interval, without gaps."
    @views @assert !any((fbands[2:end,1] .- fbands[1:end-1,2]) .< zero(T)) "Frequency bands should cover entire [0,fs/2] interval, without overlaps."
end

to_mat(A::AbstractMatrix) = A
function to_mat(y::AbstractVector)
    ymat = zeros(eltype(y), length(y)-1, 2);
    ymat[:,1] .= @view y[1:end-1];
    ymat[:,2] .= @view y[2:end]
    return ymat
end
function to_mat(pairs::Vector{Pair{T1,T2}} where {T1<:Tuple{<:Real,<:Real}, T2<:Union{Real, Tuple{<:Real,<:Real}}})
    xmat, ymat = zeros(length(pairs), 2), zeros(length(pairs), 2)
    for i = 1:length(pairs)
        xmat[i,:] .= pairs[i].first
        ymat[i,:] .= pairs[i].second
    end
    return xmat, ymat
end

function get_flength_M(filter_order)
    filter_length = filter_order + 1
    M = isodd(filter_length) ? (filter_length-1)÷2 : filter_length÷2-1
    return filter_length, M
end

function firls_design(filter_order::Integer, bands_DW::Matrix, D::Matrix, W::Matrix, antisymmetric::Bool; fs::Real = 1, solver::Function = \)
    validate_inputs(filter_order, bands_DW, D, W, fs)
    filter_length, M = get_flength_M(filter_order)
    odd = isodd(filter_length)
    a = _calc_amplitude_coeff(M, bands_DW, D, W, solver, Val(odd), Val(antisymmetric))
    h = _to_impulse_response(a, Val(odd), Val(antisymmetric))
end
function firls_design(filter_order::Integer, bands_DW::Matrix, D::Matrix, W::Vector, antisymmetric::Bool; fs::Real = 1, solver::Function = \)
    firls_design(filter_order, bands_DW, D, hcat(W,W), antisymmetric; fs = fs, solver = solver)
end
function firls_design(filter_order::Integer, bands_DW::Matrix, D::Matrix, antisymmetric::Bool; fs::Real = 1, solver::Function = \)
    validate_inputs(filter_order, bands_DW, D, fs)
    filter_length, M = get_flength_M(filter_order)
    odd = isodd(filter_length)
    a = _calc_amplitude_coeff(M, bands_DW, D, solver, Val(odd), Val(antisymmetric))
    h = _to_impulse_response(a, Val(odd), Val(antisymmetric))
end

function _calc_amplitude_coeff(M, bands_DW, D, W, solver, odd, antisymmetric)
    Q = get_Q(M, bands_DW, W, odd, antisymmetric)
    b = get_b(M, bands_DW, D, W, odd, antisymmetric)
    a = _solve(Q, b, solver, odd, antisymmetric)
end
function _calc_amplitude_coeff(M, bands_DW, D, solver, odd, antisymmetric)
    Q = get_Q(M, odd, antisymmetric)
    b = get_b(M, bands_DW, D, ones(size(D)), odd, antisymmetric)
    a = _solve(Q, b, solver, odd, antisymmetric)
end

_solve(Q, b, solver, odd, antisymmetric) = solver(Q, b)
function _solve(Q, b, solver, odd::Val{true}, antisymmetric::Val{true})
    a = zeros(length(b))
    if length(a) > 1
        @views a[2:end] .= solver(Q[2:end,2:end], b[2:end])
    end
    return a
end
function _solve(Q::UniformScaling, b, solver, odd::Val{true}, antisymmetric::Val{true})
    a = zeros(length(b))
    if length(a) > 1
        @views a[2:end] .= solver(Q, b[2:end])
    end
    return a
end

function get_Q(M, f, W::Matrix, odd, antisymmetric)
    q = get_q(M, f, W::Matrix, odd)
    Q1, Q2 = q_to_Q1Q2(q, M, odd)
    Q = Q1Q2_to_Q(Q1, Q2, antisymmetric)
end
get_Q(M, odd, antisymmetric) = I
function get_Q(M, odd::Val{true}, antisymmetric::Val{false})
    v_diag = fill(1., M+1)
    v_diag[1] *= 2
    Q = Diagonal(v_diag)
end

function q_to_Q1Q2(q, M, odd::Val{true})
    Q1 = @views to_toeplitz(q[1:M+1])
    Q2 = @views to_hankel(q[1:M+1], q[M+1:end])
    return Q1, Q2
end
function q_to_Q1Q2(q, M, odd::Val{false})
    Q1 = @views to_toeplitz(q[1:M+1])
    Q2 = @views to_hankel(q[1+1:M+1+1], q[M+1+1:end])
    return Q1, Q2
end

Q1Q2_to_Q(Q1, Q2, antisymmetric::Val{false}) = Q1 .+ Q2
Q1Q2_to_Q(Q1, Q2, antisymmetric::Val{true}) = Q1 .- Q2

allocate_q(M, odd::Val{true}) = q_out = zeros(2M+1)
allocate_q(M, odd::Val{false}) = q_out = zeros(2M+1+1)

function get_q(M, f, W::Matrix, odd)
    a, b, α, β, γ, k = constants_q(f, W)
    _αn, _βn⁻¹, _γ⁻² = copy(α), copy(β), copy(γ)
    q_out, _qn = allocate_q(M, odd), zeros(size(f))
    q_out[1] = qn_n0!(_qn, idx2n_q(1, odd), k, f, a, b, _αn, _βn⁻¹, _γ⁻², odd)
    for idx = 2:length(q_out)
        _αn .= α; _βn⁻¹ .= β; _γ⁻² .= γ
        n = idx2n_q(idx, odd)
        q_out[idx] = qn!(_qn, n, k, _αn, _βn⁻¹, _γ⁻²)
    end
    return q_out
end

function constants_q(f, W::Matrix)
    fs = 2f[end,2]; k = 2/fs
    a = @views @. (W[:,2] - W[:,1]) / (f[:,2] - f[:,1])
    b = @views @. W[:,1] - f[:,1]*a
    # α = πkf
    α = copy(f); α *= k*π
    # β = (af + b)/(πk)
    β = @. (a*f + b); β *= 1/(π*k)
    # γ = a/(π²k²) 
    γ = copy(a); γ *= 1/(π*k)^2
    return a, b, α, β, γ, k
end

function qn_n0!(_qn, n, k, f, a, b, _αn, _βn⁻¹, _γn⁻², odd::Val{true})
    qn!(_qn, k, f, a, b)
end
function qn_n0!(_qn, n, k, f, a, b, _αn, _βn⁻¹, _γn⁻², odd::Val{false})
    # qn!(_qn, n, k, _αn, _βn⁻¹, _γn⁻²)
    qn!(_qn, k, f, a, b)
end

idx2n_q(idx, odd::Val{true}) = idx - 1.
idx2n_q(idx, odd::Val{false}) = idx - 1.

function qn!(_qn, k::Real, f, a, b)
    # Fallback for when n = 0
    # _qn = (af²/2 + bf)
    #_qn = 1/2 * (af² + 2bf)
    # bn = k * (_qn[:,2] - _qn[:,1]) 
    @. _qn = b*f
    @. _qn *= 2
    @. _qn += a*f^2
    nan2zero!(_qn)
    bn = @views k/2 * (sum(_qn[:,2]) - sum(_qn[:,1]))
end

function qn!(_qn, n::Real, k::Real, _αn, _βn⁻¹, _γn⁻²)
    _αn .*= n
    @. _βn⁻¹ *= 1/n
    @. _γn⁻² *= 1/n^2
    @. _qn = _βn⁻¹ * sin(_αn) + _γn⁻² * cos(_αn)
    nan2zero!(_qn)
    bn = @views k * (sum(_qn[:,2]) - sum(_qn[:,1]))
end

function get_b(M, f, D, W::Matrix, odd, antisymmetric)
    a, b, c, d, α, β, γ, δ, k = constants_b(f, D, W)
    _αn, _βn², _δn = copy(α), copy(β), copy(δ)
    b_out, _bn = zeros(M+1), zeros(size(f))
    b_out[1] = bn_n0!(_bn, idx2n_b(1, odd), k, f, a, b, c, d, _αn, _βn², γ, _δn, odd, antisymmetric)
    for idx = 2:length(b_out)
        _αn .= α; _βn² .= β; _δn .= δ
        n = idx2n_b(idx, odd)
        b_out[idx] = bn!(_bn, n, k, _αn, _βn², γ, _δn, antisymmetric)
    end
    return b_out
end

function constants_b(f, D, W::Matrix)
    fs = 2f[end,2]; k = 2/fs
    a = @views @. (D[:,2] - D[:,1]) / (f[:,2] - f[:,1])
    b = @views @. D[:,1] - f[:,1]*a
    c = @views @. (W[:,2] - W[:,1]) / (f[:,2] - f[:,1])
    d = @views @. W[:,1] - f[:,1]*c
    # α = kπf
    α = @. f; α *= k*π
    # β = k²π²(acf² + (ad+bc)f + bd)
    β = @. a*c*f^2 + (a*d+b*c)*f + b*d; β *= (k*π)^2
    # γ = -2ac 
    γ = @. a*c; γ *= -2
    # δ = kπ(2acf + ad + bc)
    δ = @. a*c*f; δ *= 2; @. δ += a*d + b*c; δ *= π*k
    return a, b, c, d, α, β, γ, δ, k
end

function bn_n0!(_bn, n, k, f, a, b, c, d, _αn, _βn², γ, _δn, odd, antisymmetric)
    bn!(_bn, n, k, _αn, _βn², γ, _δn, antisymmetric)
end
function bn_n0!(_bn, n, k, f, a, b, c, d, _αn, _βn², γ, _δn, odd::Val{true}, antisymmetric::Val{false})
    bn!(_bn, k, f, a, b, c, d)
end

idx2n_b(idx, odd::Val{true}) = idx - 1.
idx2n_b(idx, odd::Val{false}) = idx - 0.5

function bn!(_bn, k::Real, f, a, b, c, d)
    # Fallback for when n = 0
    # _bn = acf³/3 + (ad+bc)f²/2 + bdf
    # bn = k * (_bn[:,2] - _bn[:,1]) 
    @. _bn = b*d*f
    @. _bn *= 2
    @. _bn += (a*d+b*c)*f^2
    @. _bn *= 3/2
    @. _bn += a*c*f^3
    nan2zero!(_bn)
    bn = @views k/3 * (sum(_bn[:,2]) - sum(_bn[:,1]))
end

function bn!(_bn, n::Real, k::Real, _αn, _βn², γ, _δn, antisymmetric)
    # if antisymmetric is false: _bn = 1/(π³k³n³) * (sin(αn) * (βn² + γ) + δncos(αn))
    # if antisymmetric is true:  _bn = 1/(π³k³n³) * (-cos(αn) * (βn² + γ) + δnsin(αn))
    #                 equal to:  _bn = 1/(π³k³n³) * (sin(αn-π/2) * (βn² + γ) + δncos(αn-π/2))
    # see _update_trig_arg_b
    #  bn = k * (_bn[:,2] - _bn[:,1])
    _update_trig_arg_b!(_αn, n, antisymmetric)
    @. _βn² *= n^2
    @. _δn *= n
    @. _bn = sin(_αn) * (_βn² + γ) + _δn*cos(_αn)
    nan2zero!(_bn)
    bn = @views k/(π*n*k)^3 * (sum(_bn[:,2]) - sum(_bn[:,1]))
end

_update_trig_arg_b!(_αn, n, antisymmetric::Val{false}) = _αn .*= n
function _update_trig_arg_b!(_αn, n, antisymmetric::Val{true})
    _αn .*= n
    _αn .-= π/2
end

function _to_impulse_response(a, odd::Val{true}, antisymmetric::Val{false})
    """ Type I linear phase FIR filter """
    filter_length = 2length(a) - 1
    N_a, h = length(a), zeros(eltype(a), filter_length)
    h[1:N_a-1] .= @view(a[end:-1:2])
    h[N_a] = 2a[1]
    h[N_a+1:end] .= @view(a[2:end])
    return h
end
function _to_impulse_response(a, odd::Val{false}, antisymmetric::Val{false})
    """ Type II linear phase FIR filter """
    filter_length = 2length(a)
    N_a, h = length(a), zeros(eltype(a), filter_length)
    h[1:N_a] .= @view(a[end:-1:1])
    h[N_a+1:end] .= a
    return h
end
function _to_impulse_response(a, odd::Val{true}, antisymmetric::Val{true})
    """ Type III linear phase FIR filter """
    filter_length = 2length(a)-1
    N_a, h = length(a), zeros(eltype(a), filter_length)
    h[1:N_a-1] .= @view(a[end:-1:2])
    h[N_a+1:end] .-= @view(a[2:end])
    return h
end
function _to_impulse_response(a, odd::Val{false}, antisymmetric::Val{true})
    """ Type IV linear phase FIR filter """
    filter_length = 2length(a)
    N_a, h = length(a), zeros(eltype(a), filter_length)
    h[1:N_a] .= @view(a[end:-1:1])
    h[N_a+1:end] .-= a
    return h
end

#################################################################
to_toeplitz(vals) = to_toeplitz(vals, vals)
function to_toeplitz(vals_left::AbstractVector{T1}, vals_top::AbstractVector{T2}) where {T1,T2}
    @assert vals_left[1] == vals_top[1]
    N_rows, N_cols = length(vals_left), length(vals_top)
    A = zeros(promote_type(T1,T2), N_rows, N_cols)
    vals_total = zeros(promote_type(T1,T2), N_rows+N_cols-1)
    vals_total[1:N_cols] .= @view(vals_top[end:-1:1])
    vals_total[N_rows+1:end] .= @view(vals_left[2:end])
    for j = 1:N_cols
        A[:,j] .= @view(vals_total[end-N_rows+2-j:end+1-j])
    end
    return A
end

function to_hankel(vals_left::AbstractVector{T1}, vals_bottom::AbstractVector{T2}) where {T1,T2}
    @assert vals_left[end] == vals_bottom[1]
    N_rows, N_cols = length(vals_left), length(vals_bottom)
    A = zeros(promote_type(T1,T2), N_rows, N_cols)
    vals_total = zeros(promote_type(T1,T2), N_rows+N_cols-1)
    vals_total[1:N_rows] .= vals_left
    vals_total[N_rows+1:end] .= @view(vals_bottom[2:end])
    for j = 1:N_cols
        A[:,j] .= @view(vals_total[j:j+N_rows-1])
    end
    return A
end

function nan2zero!(x::Array{T}) where T
    for idx = 1:length(x)
        if isnan(x[idx])
            x[idx] = zero(T)
        end
    end
end

#########################################################################
function freqz(h::Vector{T}; fs::Real = 1, N = 1000) where T
    ω = range(0, stop = π, length = N)
    H = zeros(Complex{T}, length(ω))
    for (idx,h_n) in enumerate(h)
        n = idx-1
        H .+= h_n * exp.(-n*1im*ω)
    end
    return H, ω * (fs/2π)
end

end


