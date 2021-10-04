import Clustering
import Distances
import LinearAlgebra: norm

using MLJBase
using MLJClusteringInterface
using Random: seed!
using Test

seed!(132442)
X, y = @load_crabs

####
#### KMEANS
####

@testset "KMeans" begin
    barekm = KMeans()
    fitresult, cache, report = fit(barekm, 1, X)
    R = matrix(transform(barekm, fitresult, X))
    X_array = matrix(X)
    # distance from first point to second center
    @test R[1, 2] ≈ norm(view(X_array, 1, :) .- view(fitresult[1], :, 2))^2
    @test R[10, 3] ≈ norm(view(X_array, 10, :) .- view(fitresult[1], :, 3))^2

    p = predict(barekm, fitresult, X)
    @test argmin(R[1, :]) == p[1]
    @test argmin(R[10, :]) == p[10]
end

####
#### KMEDOIDS
####

@testset "KMedoids" begin
    barekm = KMedoids()
    fitresult, cache, report = fit(barekm, 1, X)
    X_array = matrix(X)
    R = matrix(transform(barekm, fitresult, X))
    @test R[1, 2] ≈ Distances.evaluate(
        barekm.metric, view(X_array, 1, :), view(fitresult[1], :, 2)
    )
    @test R[10, 3] ≈ Distances.evaluate(
        barekm.metric, view(X_array, 10, :), view(fitresult[1], :, 3)
    )
    p = predict(barekm, fitresult, X)
    @test all(report.assignments .== p)
end

@testset "DBSCAN" begin
   dbscan = DBSCAN() 
   fitresult = fit(dbscan, 1, X)
   A = transform(dbscan, fitresult, X)
   p = predict(dbscan, fitresult, X)
end