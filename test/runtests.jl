import Clustering
import Distances
import LinearAlgebra: norm

using MLJBase
using MLJTestIntegration
using MLJClusteringInterface
using Random: seed!
using Test

seed!(132442)
X, y = @load_crabs


# # K_MEANS

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


# # K_MEDOIDS

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


# # DBSCAN

@testset "DBSCAN" begin

    # five spot pattern
    X = [
        0.0 0.0
        1.0 0.0
        1.0 1.0
        0.0 1.0
        0.5 0.5
    ] |> MLJBase.table

    # radius < √2 ==> 5 clusters
    dbscan = DBSCAN(radius=0.1)
    yhat1, report1 = predict(dbscan, nothing, X)
    @test report1.nclusters == 5
    @test report1.point_types == fill('B', 5)
    @test Set(yhat1) == Set(unique(yhat1))
    @test Set(report1.cluster_labels) == Set(unique(yhat1))

    # DbscanCluster fields:
    @test propertynames(report1.clusters[1]) == (:size, :core_indices, :boundary_indices)

    # radius > √2 ==> 1 cluster
    dbscan = DBSCAN(radius=√2+eps())
    yhat, report = predict(dbscan, nothing, X)
    @test report.nclusters == 1
    @test report.point_types == fill('C', 5)
    @test length(unique(yhat)) == 1

    # radius < √2 && min_cluster_size = 2 ==> all points are noise
    dbscan = DBSCAN(radius=0.1, min_cluster_size=2)
    yhat, report = predict(dbscan, nothing, X)
    @test report.nclusters == 0
    @test report.point_types == fill('N', 5)
    @test length(unique(yhat)) == 1

    # MLJ integration:
    model = DBSCAN(radius=0.1)
    mach = machine(model) # no training data
    yhat = predict(mach, X)
    @test yhat == yhat1
    @test MLJBase.report(mach).point_types == report1.point_types
    @test MLJBase.report(mach).nclusters == report1.nclusters

end

# # HierarchicalClustering

@testset "HierarchicalClustering" begin
    h = Inf; k = 1; linkage = :complete; bo = :optimal;
    metric = Distances.Euclidean()
    mach = machine(HierarchicalClustering(h = h, k = k, metric = metric,
                                          linkage = linkage, branchorder = bo))
    yhat = predict(mach, X)
    @test length(union(yhat)) == 1 # uses h = Inf
    cutter = report(mach).cutter
    @test length(union(cutter(k = 4))) == 4 # uses k = 4
    dendro = Clustering.hclust(Distances.pairwise(metric, hcat(X...), dims = 1),
                               linkage = linkage, branchorder = bo)
    @test cutter(k = 2) == Clustering.cutree(dendro, k = 2)
    @test report(mach).dendrogram.heights == dendro.heights
end

@testset "MLJ interface" begin
    models = [KMeans, KMedoids, DBSCAN, HierarchicalClustering]
    failures, summary = MLJTestIntegration.test(
        models,
        X;
        mod=@__MODULE__,
        verbosity=0,
        throw=false, # set to true to debug
    )
    @test isempty(failures)
end
