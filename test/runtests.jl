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
    # five spot pattern
    X = [
        0.0 0.0
        1.0 0.0
        1.0 1.0
        0.0 1.0
        0.5 0.5
    ]

    # radius < √2 ==> 5 clusters
    dbscan = DBSCAN(radius=0.1) 
    fitresult = fit(dbscan, 1, X)
    A = transform(dbscan, fitresult, X)
    p = predict(dbscan, fitresult, X)
    @test size(matrix(A)) == (5, 2)
    @test A.x2 == [0,0,0,0,0]
    @test Set(p) == Set(unique(p))

    # radius > √2 ==> 1 cluster
    dbscan = DBSCAN(radius=√2+eps()) 
    fitresult = fit(dbscan, 1, X)
    A = transform(dbscan, fitresult, X)
    p = predict(dbscan, fitresult, X)
    @test size(matrix(A)) == (5, 2)
    @test A.x2 == [1,1,1,1,1]
    @test unique(p) == [1]

    # radius < √2 && min_cluster_size = 2 ==> all points are noise
    dbscan = DBSCAN(radius=0.1, min_cluster_size=2) 
    fitresult = fit(dbscan, 1, X)
    A = transform(dbscan, fitresult, X)
    p = predict(dbscan, fitresult, X)
    @test size(matrix(A)) == (5, 2)
    @test A.x2 == [0,0,0,0,0]
    @test unique(p) == [0]
end