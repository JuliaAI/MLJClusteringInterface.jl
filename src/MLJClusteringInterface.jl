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
using NearestNeighbors

# ===================================================================
## EXPORTS
export KMeans, KMedoids, DBSCAN

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

const DBSCANDescription ="""
DBSCAN algorithm: find clusters through density-based expansion of seed points.
"""

const KMFields ="""
    ## Keywords

    * `k=3`     : number of centroids
    * `metric`  : distance metric to use
"""

const DBFields ="""
    ## Keywords

    * `radius=1.0`         : query radius
    * `leafsize=20`        : number of points binned in each leaf node
    * `min_neighbors=1`    : minimum number of core point neighbors
    * `min_cluster_size=1` : minimum number of points in a valid cluster
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
end

####
#### KMeans
####

function MMI.fit(model::KMeans, verbosity::Int, X)
    # NOTE: using transpose here to get a LinearAlgebra.Transpose object
    # which Kmeans can handle.
    Xarray = transpose(MMI.matrix(X))
    result = Cl.kmeans(Xarray, model.k; distance=model.metric)
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
#### DBSCAN
####
"""
DBSCAN(; kwargs...)

$DBSCANDescription

$DBFields

See also the 
[package documentation](https://juliastats.org/Clustering.jl/stable/dbscan.html).
"""
@mlj_model mutable struct DBSCAN <: MMI.Unsupervised
    radius::Real = 1.0::(_ > 0)
    leafsize::Int = 20::(_ > 0)
    min_neighbors::Int = 1::(_ > 0)
    min_cluster_size::Int = 1::(_ > 0)
end

function MMI.fit(model::DBSCAN, verbosity::Int, X)
    Xarray   = MMI.matrix(X, transpose=true)
    clusters = Cl.dbscan(Xarray, model.radius;
                       leafsize=model.leafsize,
                       min_neighbors=model.min_neighbors,
                       min_cluster_size=model.min_cluster_size)

    # assignments and point types
    npoints     = size(Xarray, 2)
    assignments = zeros(Int, npoints)
    pointtypes  = zeros(Int, npoints)
    for (k, cluster) in enumerate(clusters)
        for i in cluster.core_indices
            assignments[i] = k
            pointtypes[i] = 1
        end
        for i in cluster.boundary_indices
            assignments[i] = k
            pointtypes[i] = 0
        end
    end

    result = (Xarray, assignments, pointtypes)
    cache  = nothing
    report = nothing
    result, cache, report
end

MMI.fitted_params(::DBSCAN, fitresult) = (assignments=fitresult[1][2],
                                          pointtypes=fitresult[1][3])

function MMI.transform(::DBSCAN, fitresult, X)
    # table with assignments in first column and
    # point types in second column (core=1 vs. boundary=0)
    _, assignments, pointtypes = fitresult[1]
    X̃ = [assignments pointtypes]
    MMI.table(X̃, prototype=X)
end

function MMI.predict(::DBSCAN, fitresult, Xnew)
    X1, assignments, _ = fitresult[1]
    X2 = MMI.matrix(Xnew, transpose=true)

    labels = MMI.categorical(assignments)

    # construct KDtree with points in X1
    tree = KDTree(X1, Euclidean())

    # find nearest neighbor of X2 in X1
    inds, _ = nn(tree, X2)

    # return assignment of nearest neighbor
    labels[inds]
end

####
#### METADATA
####

metadata_pkg.(
    (KMeans, KMedoids),
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
    DBSCAN,
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    descr = DBSCANDescription,
    path = "$(PKG).DBSCAN"
)

end # module

