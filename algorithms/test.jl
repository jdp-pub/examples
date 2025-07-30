include("genutils.jl")

function main()    

    A = diagm(1:10)
    A = diagm([0.1,0.5,0.8, 0.9,1,1.1,1.2,1.3,1.4,1.8, 2.5,3])
    #@time power_iteration(A)
    #@time lanczos(A)
    #@time arnoldi(A)
    @time qr_decomp(A)
end


# for testing functions
main()
