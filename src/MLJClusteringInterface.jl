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
const PKG = "MLJClusteringInterface"

####
#### KMeans
####

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
    human_name = "K-means clusterer",
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    path = "$(PKG).KMeans"
)

metadata_model(
    KMedoids,
    human_name = "K-medoids clusterer",
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    path = "$(PKG).KMedoids"
)
"""
$(MMI.doc_header(KMeans))

[K-means](http://en.wikipedia.org/wiki/K_means) is a classical method for
clustering or vector quantization. It produces a fixed number of clusters,
each associated with a *center* (also known as a *prototype*), and each data
point is assigned to a cluster with the nearest center.

From a mathematical standpoint, K-means is a coordinate descent
algorithm that solves the following optimization problem:

```math
\\text{minimize} \\ \\sum_{i=1}^n \\| \\mathbf{x}_i - \\boldsymbol{\\mu}_{z_i} \\|^2 \\ \\text{w.r.t.} \\ (\\boldsymbol{\\mu}, z)
```
Here, ``\\boldsymbol{\\mu}_k`` is the center of the ``k``-th cluster, and
``z_i`` is an index of the cluster for ``i``-th point ``\\mathbf{x}_i``.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check column  scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `k=3`: The number of centroids to use in clustering.

- `metric::SemiMetric=Distances.SqEuclidean`: The metric used to calculate the
  clustering. Must have type `PreMetric` from Distances.jl.


# Operations

- `predict(mach, Xnew)`: return cluster label assignments, given new
   features `Xnew` having the same Scitype as `X` above.

- `transform(mach, Xnew)`: instead return the mean pairwise distances from
   new samples to the cluster centers.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `centers`: The coordinates of the cluster centers.

# Report

The fields of `report(mach)` are:

- `assignments`: The cluster assignments of each point in the training data.

- `cluster_labels`: The labels assigned to each cluster.

# Examples

```
using MLJ
KMeans = @load KMeans pkg=Clustering

table = load_iris()
y, X = unpack(table, ==(:target), rng=123)
model = KMeans(k=3)
mach = machine(model, X) |> fit!

yhat = predict(mach, X)
@assert yhat == report(mach).assignments

compare = zip(yhat, y) |> collect;
compare[1:8] # clusters align with classes

center_dists = transform(mach, fitted_params(mach).centers')

@assert center_dists[1][1] == 0.0
@assert center_dists[2][2] == 0.0
@assert center_dists[3][3] == 0.0
```

See also
[`KMedoids`](@ref)
"""
KMeans

"""
$(MMI.doc_header(KMedoids))

[K-medoids](http://en.wikipedia.org/wiki/K-medoids) is a clustering algorithm that works by
finding ``k`` data points (called *medoids*) such that the total distance between each data
point and the closest *medoid* is minimal.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `k=3`: The number of centroids to use in clustering.

- `metric::SemiMetric=Distances.SqEuclidean`: The metric used to calculate the
  clustering. Must have type `PreMetric` from Distances.jl.

# Operations

- `predict(mach, Xnew)`: return cluster label assignments, given new
   features `Xnew` having the same Scitype as `X` above.

- `transform(mach, Xnew)`: instead return the mean pairwise distances from
   new samples to the cluster centers.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `medoids`: The coordinates of the cluster medoids.

# Report

The fields of `report(mach)` are:

- `assignments`: The cluster assignments of each point in the training data.

- `cluster_labels`: The labels assigned to each cluster.

# Examples

```
using MLJ
KMedoids = @load KMedoids pkg=Clustering

table = load_iris()
y, X = unpack(table, ==(:target), rng=123)
model = KMedoids(k=3)
mach = machine(model, X) |> fit!

yhat = predict(mach, X)
@assert yhat == report(mach).assignments

compare = zip(yhat, y) |> collect;
compare[1:8] # clusters align with classes

center_dists = transform(mach, fitted_params(mach).medoids')

@assert center_dists[1][1] == 0.0
@assert center_dists[2][2] == 0.0
@assert center_dists[3][3] == 0.0
```

See also
[`KMeans`](@ref)
"""
KMedoids


metadata_model(
    DBSCAN,
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    weights = false,
    descr = DBSCANDescription,
    path = "$(PKG).DBSCAN"
)

end # module
