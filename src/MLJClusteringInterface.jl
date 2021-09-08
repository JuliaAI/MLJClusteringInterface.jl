# NOTE: there's a `kmeans!` function that updates centers, maybe a candidate
# for the `update` machinery. Same for `kmedoids!`
# NOTE: if the prediction is done on the original array, just the assignment
# should be returned, unclear what's the best way of doing this.

module MLJClusteringInterface

# ===================================================================
# IMPORTS
import Clustering
import MLJModelInterface
import MLJModelInterface: Continuous, Count, Finite, Multiclass, Table, OrderedFactor,
    @mlj_model, metadata_model, metadata_pkg

using Distances

# ===================================================================
## EXPORTS
export KMeans, KMedoids, HierarchicalClustering

# ===================================================================
## CONSTANTS
# Define constants for easy referencing of packages
const MMI = MLJModelInterface
const Cl = Clustering

# Definitions of model descriptions for use in model doc-strings.
const KMeansDescription ="""
K-Means algorithm: find K centroids corresponding to K clusters in the data.
"""

const KMedoidsDescription ="""
K-Medoids algorithm: find K centroids corresponding to K clusters in the data.
Unlike K-Means, the centroids are found among data points themselves.
"""

const HierarchicalClusteringDescription ="""
Hierarchical clustering algorithms: build a dendrogram of nested clusters
by repeatedly merging or splitting clusters.
"""

const KMFields ="""
## Keywords

* `k=3`     : number of centroids
* `metric`  : distance metric to use
"""

const PKG = "MLJClusteringInterface"

####
#### KMeans
####
"""
    KMeans(; kwargs...)

$KMeansDescription

$KMFields

See also the
[package documentation](http://juliastats.github.io/Clustering.jl/latest/kmeans.html).
"""
@mlj_model mutable struct KMeans <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
    init = :kmpp
end

####
#### KMeans
####

function MMI.fit(model::KMeans, verbosity::Int, X)
    # NOTE: using transpose here to get a LinearAlgebra.Transpose object
    # which Kmeans can handle.
    Xarray = transpose(MMI.matrix(X))
    result = Cl.kmeans(Xarray, model.k; distance=model.metric, init = model.init)
    cluster_labels = MMI.categorical(1:model.k)
    fitresult = (result.centers, cluster_labels) # centers (p x k)
    cache = nothing
    report = (
        assignments=result.assignments, # size n
        cluster_labels=cluster_labels
    )
    return fitresult, cache, report
end

MMI.fitted_params(::KMeans, fitresult) = (centers=fitresult[1],)

function MMI.transform(model::KMeans, fitresult, X)
    # pairwise distance from samples to centers
    X̃ = pairwise(
        model.metric,
        transpose(MMI.matrix(X)),
        fitresult[1],
        dims=2
    )
    return MMI.table(X̃, prototype=X)
end

"""
    KMedoids(; kwargs...)

$KMedoidsDescription

$KMFields

See also the
[package documentation](http://juliastats.github.io/Clustering.jl/latest/kmedoids.html).
"""
@mlj_model mutable struct KMedoids <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

function MMI.fit(model::KMedoids, verbosity::Int, X)
    # NOTE: using transpose=true will materialize the transpose (~ permutedims), KMedoids
    # does not yet accept LinearAlgebra.Transpose
    Xarray = MMI.matrix(X, transpose=true)
    # cost matrix: all the pairwise distances
    cost_array = pairwise(model.metric, Xarray, dims=2) # n x n
    result = Cl.kmedoids(cost_array, model.k)
    cluster_labels = MMI.categorical(1:model.k)
    fitresult = (view(Xarray, :, result.medoids), cluster_labels) # medoids
    cache = nothing
    report = (
        assignments=result.assignments, # size n
        cluster_labels=cluster_labels
    )
    return fitresult, cache, report
end

MMI.fitted_params(::KMedoids, fitresult) = (medoids=fitresult[1],)

function MMI.transform(model::KMedoids, fitresult, X)
    # pairwise distance from samples to medoids
    X̃ = pairwise(
        model.metric,
        MMI.matrix(X, transpose=true),
        fitresult[1], dims=2
    )
    return MMI.table(X̃, prototype=X)
end

####
#### Predict methods
####

function MMI.predict(model::Union{KMeans,KMedoids}, fitresult, Xnew)
    locations, cluster_labels = fitresult
    Xarray = MMI.matrix(Xnew)
    (n, p), k = size(Xarray), model.k
    pred = zeros(Int, n)

    @inbounds for i in 1:n
        minv = Inf
        @inbounds @simd for j in 1:k
            curv = evaluate(
                model.metric, view(Xarray, i, :), view(locations, :, j)
            )
            P = curv < minv
            pred[i] = j * P + pred[i] * !P # if P is true --> j
            minv = curv * P + minv * !P # if P is true --> curvalue
        end
    end
    return cluster_labels[pred]
end

####
#### HierarchicalClustering
####

"""
    HierarchicalClustering(; kwargs...)

$HierarchicalClusteringDescription

## Keywords

* `linkage` : `:single` (default), `:average`, `:complete`, `:ward`, `:ward_presquared`
* `metric` : distance metric to use (`SqEuclidean()` by default).
* `branchorder` : algorithm to order leaves and brancges (`:r` by default, `:barjoseph` or `:optimal`)
* `k = 3` : number of clusters to use for prediction (can be changed after fitting).
* `h = nothing` : the height at which the tree is cut during prediction (can be changed after fitting).

See also
[package documentation](https://juliastats.org/Clustering.jl/stable/hclust.html).

## Example
```
using MLJ, MLJClusteringInterface, DataFrames, StatsPlots
data = DataFrame([randn(50, 2) .+ [2 2]; randn(50, 2) .+ [-3 0]], :auto)
scatter(data.x1, data.x2) # plot data
mach = fit!(machine(HierarchicalClustering(), data))
plot(report(mach)) # plot dendrogram
mach.model.k = 2
predictions = predict(mach)
scatter(data.x1, data.x2, color = predictions) # plot data with class labels
```
"""
@mlj_model mutable struct HierarchicalClustering <: MMI.Unsupervised
    linkage::Symbol = :single :: (_ ∈ (:single, :average, :complete, :ward, :ward_presquared))
    metric::SemiMetric = SqEuclidean()
    branchorder::Symbol = :r :: (_ ∈ (:r, :barjoseph, :optimal))
    k::Union{Int, Nothing} = 3
    h::Union{Float64, Nothing} = nothing
end

function MMI.fit(model::HierarchicalClustering, verbosity::Int, X)
    Xarray = MMI.matrix(X)
    # all the pairwise distances
    d = pairwise(model.metric, Xarray, dims = 1) # n x n
    fit_result = Cl.hclust(d, linkage = model.linkage, branchorder = model.branchorder)
    cache = nothing
    report = fit_result
    return fit_result, cache, report
end

function MMI.predict(model::HierarchicalClustering, fitresult, ::Any)
    Cl.cutree(fitresult; k = model.k, h = model.h)
end

####
#### METADATA
####

metadata_pkg.(
    (KMeans, KMedoids, HierarchicalClustering),
    name="Clustering",
    uuid="aaaa29a8-35af-508c-8bc3-b662a17a0fe5",
    url="https://github.com/JuliaStats/Clustering.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
)

metadata_model(
    KMeans,
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    descr = KMeansDescription,
    path = "$(PKG).KMeans"
)

metadata_model(
    KMedoids,
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    descr = KMedoidsDescription,
    path = "$(PKG).KMedoids"
)

metadata_model(
    HierarchicalClustering,
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    descr = HierarchicalClusteringDescription,
    path = "$(PKG).HierarchicalClustering"
)

end # module

