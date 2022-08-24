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
export KMeans, KMedoids, DBSCAN

# ===================================================================
## CONSTANTS
# Define constants for easy referencing of packages
const MMI = MLJModelInterface
const Cl = Clustering
const PKG = "MLJClusteringInterface"


# # K_MEANS

@mlj_model mutable struct KMeans <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

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

# # K_MEDOIDS

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


# # PREDICT FOR K_MEANS AND K_MEDOIDS

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

# # DBSCAN

@mlj_model mutable struct DBSCAN <: MMI.Static
    radius::Real = 1.0::(_ > 0)
    leafsize::Int = 20::(_ > 0)
    min_neighbors::Int = 1::(_ > 0)
    min_cluster_size::Int = 1::(_ > 0)
end

# As DBSCAN is `Static`, there is no `fit` to implement.

function MMI.predict(model::DBSCAN, ::Nothing, X)

    Xarray   = MMI.matrix(X)'

    # output of core algorithm:
    clusters = Cl.dbscan(
        Xarray, model.radius;
        leafsize=model.leafsize,
        min_neighbors=model.min_neighbors,
        min_cluster_size=model.min_cluster_size,
    )
    nclusters = length(clusters)

    # assignments and point types
    npoints     = size(Xarray, 2)
    assignments = zeros(Int, npoints)
    raw_point_types  = fill('N', npoints)
    for (k, cluster) in enumerate(clusters)
        for i in cluster.core_indices
            assignments[i] = k
            raw_point_types[i] = 'C'
        end
        for i in cluster.boundary_indices
            assignments[i] = k
            raw_point_types[i] = 'B'
        end
    end
    point_types = MMI.categorical(raw_point_types)
    cluster_labels = unique(assignments)

    yhat = MMI.categorical(assignments)
    report = (; point_types, nclusters, cluster_labels, clusters)
    return yhat, report
end

MMI.reporting_operations(::Type{<:DBSCAN}) = (:predict,)


# # METADATA

metadata_pkg.(
    (KMeans, KMedoids, DBSCAN),
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
    path = "$(PKG).KMeans"
)

metadata_model(
    KMedoids,
    human_name = "K-medoids clusterer",
    input = MMI.Table(Continuous),
    output = MMI.Table(Continuous),
    path = "$(PKG).KMedoids"
)

metadata_model(
    DBSCAN,
    human_name = "DBSCAN clusterer (density-based spatial clustering of "*
    "applications with noise)",
    input = MMI.Table(Continuous),
    path = "$(PKG).DBSCAN"
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

"""
$(MMI.doc_header(DBSCAN))

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) is a clustering algorithm that groups
together points that are closely packed together (points with many nearby neighbors),
marking as outliers points that lie alone in low-density regions (whose nearest neighbors
are too far away). More information is available at the [Clustering.jl
documentation](https://juliastats.org/Clustering.jl/stable/index.html). Use `predict` to
get cluster assignments. Point types - core, boundary or noise - are accessed from the
machine report (see below).

This is a static implementation, i.e., it does not generalize to new data instances, and
there is no training data. For clusterers that do generalize, see [`KMeans`](@ref) or
[`KMedoids`](@ref).

In MLJ or MLJBase, create a machine with

    mach = machine(model)

# Hyper-parameters

- `radius=1.0`: query radius.

- `leafsize=20`: number of points binned in each leaf node of the nearest neighbor k-d
  tree.

- `min_neighbors=1`: minimum number of a core point neighbors.

- `min_cluster_size=1`: minimum number of points in a valid cluster.


# Operations

- `predict(mach, X)`: return cluster label assignments, as an unordered
  `CategoricalVector`. Here `X` is any table of input features (eg, a `DataFrame`) whose
  columns are of scitype `Continuous`; check column scitypes with `schema(X)`. Note that
  points of type `noise` will always get a label of `0`.


# Report

After calling `predict(mach)`, the fields of `report(mach)`  are:

- `point_types`: A `CategoricalVector` with the DBSCAN point type classification, one
  element per row of `X`. Elements are either `'C'` (core), `'B'` (boundary), or `'N'`
  (noise).

- `nclusters`: The number of clusters (excluding the noise "cluster")

- `cluster_labels`: The unique list of cluster labels

- `clusters`: A vector of `Clustering.DbscanCluster` objects from Clustering.jl, which
  have these fields:

  - `size`: number of points in a cluster (core + boundary)

  - `core_indices`: indices of points in the cluster core

  - `boundary_indices`: indices of points on the cluster boundary


# Examples

```
using MLJ

X, labels  = make_moons(400, noise=0.09, rng=1) # synthetic data with 2 clusters; X
y = map(labels) do label
    label == 0 ? "cookie" : "monster"
end;
y = coerce(y, Multiclass);

DBSCAN = @load DBSCAN pkg=Clustering
model = DBSCAN(radius=0.13, min_cluster_size=5)
mach = machine(model)

# compute and output cluster assignments for observations in `X`:
yhat = predict(mach, X)

# get DBSCAN point types:
report(mach).point_types
report(mach).nclusters

# compare cluster labels with actual labels:
compare = zip(yhat, y) |> collect;
compare[1:10] # clusters align with classes

# visualize clusters, noise in red:
points = zip(X.x1, X.x2) |> collect
colors = map(yhat) do i
   i == 0 ? :red :
   i == 1 ? :blue :
   i == 2 ? :green :
   i == 3 ? :yellow :
   :black
end
using Plots
scatter(points, color=colors)
```

"""
DBSCAN

end # module
