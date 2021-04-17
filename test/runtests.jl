using FIRLSFilterDesign
using Test

function generate_bands_vec(N; n_duplicates = 0, fs = 1.)
    bands_vec = rand(N)
    bands_vec = vcat(bands_vec, bands_vec[1:n_duplicates])
    sort!(bands_vec)
    bands_vec .-= bands_vec[1]
    bands_vec ./= bands_vec[end]
    bands_vec .*= fs/2
    return bands_vec
end

vec2mat(vec) = @views hcat(vec[1:end-1], vec[2:end])

function all_perms_q(f,W)
    M = 5
    q_odd = FIRLSFilterDesign.get_q(M, f, W, FIRLSFilterDesign.FIR_I())
    q_even = FIRLSFilterDesign.get_q(M, f, W, FIRLSFilterDesign.FIR_II())
    return q_odd, q_even
end

function all_perms_b(f,W,D)
    M = 5
    b_FIR_I     = FIRLSFilterDesign.get_b(M, f, D, W, FIRLSFilterDesign.FIR_I())
    b_FIR_II    = FIRLSFilterDesign.get_b(M, f, D, W, FIRLSFilterDesign.FIR_II())
    b_FIR_III   = FIRLSFilterDesign.get_b(M, f, D, W, FIRLSFilterDesign.FIR_III())
    b_FIR_IV    = FIRLSFilterDesign.get_b(M, f, D, W, FIRLSFilterDesign.FIR_IV())
    return b_FIR_I, b_FIR_II, b_FIR_III, b_FIR_IV
end

function all_perms_firlsdesign(args...; kwargs...)
    order_odd = 10
    order_even = order_odd + 1
    h_FIR_I     = firls_design(order_odd,  args..., false; kwargs...)
    h_FIR_II    = firls_design(order_even, args..., false; kwargs...)
    h_FIR_III   = firls_design(order_odd,  args..., true;  kwargs...)
    h_FIR_IV    = firls_design(order_even, args..., true;  kwargs...)
    return h_FIR_I, h_FIR_II, h_FIR_III, h_FIR_IV
end

function all_perms_toimpulseresponse(a)
    h_FIR_I     = FIRLSFilterDesign._to_impulse_response(a, FIRLSFilterDesign.FIR_I())
    h_FIR_II    = FIRLSFilterDesign._to_impulse_response(a, FIRLSFilterDesign.FIR_II())
    h_FIR_III   = FIRLSFilterDesign._to_impulse_response(a, FIRLSFilterDesign.FIR_III())
    h_FIR_IV    = FIRLSFilterDesign._to_impulse_response(a, FIRLSFilterDesign.FIR_IV())
    return h_FIR_I, h_FIR_II, h_FIR_III, h_FIR_IV
end

# Randomized inputs
fs_rand = 1 + rand() * 10
bands_D_mat_rand = vec2mat(generate_bands_vec(100, n_duplicates = 10, fs = fs_rand))
D_mat_rand = rand(size(bands_D_mat_rand)...)
bands_W_mat_rand = vec2mat(generate_bands_vec(100, n_duplicates = 10, fs = fs_rand))
W_mat_rand = rand(size(bands_W_mat_rand)...)
# Fixed inputs
fs = 1.
bands_D_mat = [0. .25; 0.25 0.25; .25 .5]
D_mat = [2. 1.; 10. 10.; 0. 0.]
bands_W_mat = [0. .25; 0.25 0.25; .25 .5]
W_mat = [1. 2.; 11. 11.; 2. 1.]

hs_rand1 = all_perms_firlsdesign(bands_D_mat_rand, D_mat_rand; fs = fs_rand)
hs_rand2 = all_perms_firlsdesign(bands_D_mat_rand, D_mat_rand, fill(rand(), size(D_mat_rand)); fs = fs_rand)
# hs_rand3 = all_perms_firlsdesign(bands_D_mat_rand, D_mat_rand, bands_D_mat_rand, ones(size(D_mat_rand)); fs = fs_rand)

qs_fixed = all_perms_q(bands_D_mat, W_mat)
qs_fixed_data = [
    [
        1.5
        0.0
        -0.2026423672846756
        0.0
        0.0
        0.0
        -0.02251581858718621
        0.0
        0.0
        0.0
        -0.008105694691387062
   ],
   [
        1.5
        0.0
        -0.2026423672846756
        0.0
        0.0
        0.0
        -0.02251581858718621
        0.0
        0.0
        0.0
        -0.008105694691387062
        0.0
   ]
]

bs_fixed = all_perms_b(bands_D_mat, D_mat, W_mat)
bs_fixed_data = [
    [
        1.0833333333333333
        0.6919896805485019
        0.0
        -0.24427841957880944
        -0.025330295910584485
        0.12128235798585402
    ],
    [
        0.9761255156564024
        0.32778347331995833
       -0.2012359779620491
       -0.16111118889304135
        0.08495418615790015
        0.080981441609415 
    ],
    [
        NaN
        0.6919896805485019
        0.7011228412339804
        0.2442784195788095
        0.0
        0.12128235798585399
    ],
    [
        0.40432442716331324
        0.7913393068108028
        0.48582662723339226
        0.06673443948955127
        0.03518917608697089
        0.19550649463397424
    ]
]

hs_fixed = all_perms_firlsdesign(bands_D_mat, D_mat, W_mat)
hs_fixed_data = [
    [
        0.07788241325853798
        -0.0017268410299129148
        -0.08309128924173821
        0.1009958144655533
        0.5217570192015153
        0.7495102634610803
        0.5217570192015153
        0.1009958144655533
        -0.08309128924173821
        -0.0017268410299129148
        0.07788241325853798
    ],
    [
        0.056631088208928274
        0.05686190577259725
        -0.05915886754220385
        -0.034409300873168014
        0.30433805233189765
        0.6880664081058022
        0.6880664081058022
        0.30433805233189765
        -0.034409300873168014
        -0.05915886754220385
        0.05686190577259725
        0.056631088208928274
    ],
    [
        0.10513740773549088
        0.056950501912317146
        0.2320211851531876
        0.4742540923933572
        0.432645024065513
        0.0
       -0.432645024065513
       -0.4742540923933572
       -0.2320211851531876
       -0.056950501912317146
       -0.10513740773549088
    ],
    [
        0.14347815111515985
        0.06421192516629874
        0.12735250873084986
        0.3641022891441058
        0.5103455604394119
        0.24763921096456235
        -0.24763921096456235
        -0.5103455604394119
        -0.3641022891441058
        -0.12735250873084986
        -0.06421192516629874
        -0.14347815111515985
    ]
]

function get_half(h)
    N = length(h)
    M = N÷2
    if iseven(N)
        @views h[1:M], h[end:-1:M+1]
    else
        @views h[1:M+1], h[end:-1:M+1]
    end
end

results = all_perms_toimpulseresponse(rand(5))
halfs = get_half.(results)

all_approx(x,y) = all(@. abs(x - y) < 1e16) 

@testset "FIRLSFilterDesign.jl" begin
    # q vector test
    @test all_approx(qs_fixed[1], qs_fixed[2][1:end-1])
    @test all_approx(qs_fixed[1], qs_fixed_data[1]) # odd filter length
    @test all_approx(qs_fixed[2], qs_fixed_data[2]) # even filter length
    # b vector test
    @test all_approx(bs_fixed[1],        bs_fixed_data[1])        # odd  filter length, symmetric
    @test all_approx(bs_fixed[2],        bs_fixed_data[2])        # even filter length, symmetric
    @test all_approx(bs_fixed[3][2:end], bs_fixed_data[3][2:end]) # odd  filter length, antisymmetric
    @test all_approx(bs_fixed[4],        bs_fixed_data[4])        # even filter length, antisymmetric
    # Symmetry of filter coefficients
    @test all(halfs[1][1] .==  halfs[1][2]) # odd  filter length, symmetric
    @test all(halfs[2][1] .==  halfs[2][2]) # even filter length, symmetric
    @test all(halfs[3][1] .== -halfs[3][2]) # odd  filter length, antisymmetric
    @test all(halfs[4][1] .== -halfs[4][2]) # even filter length, antisymmetric
    # Tests that show equivalence between passing no weights and passing all equal weights
    @test all_approx(hs_rand1[1], hs_rand2[1]) # odd  filter length, symmetric
    @test all_approx(hs_rand1[2], hs_rand2[2]) # even filter length, symmetric
    @test all_approx(hs_rand1[3], hs_rand2[3]) # odd  filter length, antisymmetric
    @test all_approx(hs_rand1[4], hs_rand2[4]) # even filter length, antisymmetric
    # Fixed inputs
    @test all_approx(hs_fixed[1], hs_fixed_data[1]) # odd  filter length, symmetric
    @test all_approx(hs_fixed[2], hs_fixed_data[2]) # even filter length, symmetric
    @test all_approx(hs_fixed[3], hs_fixed_data[3]) # odd  filter length, antisymmetric
    @test all_approx(hs_fixed[4], hs_fixed_data[4]) # even filter length, antisymmetric
end