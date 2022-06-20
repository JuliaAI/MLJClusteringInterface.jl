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
export KMeans, KMedoids

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
"""
$(MMI.doc_header(KMeans))


`KMeans`: The K-Means algorithm finds K centroids corresponding to K clusters in
the data. The clusters are assumed to be elliptical, should be used with a euclidean distance metric

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Count`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `k=3`: The number of centroids to use in clustering.
- `metric::SemiMetric=SqEuclidean`: The metric used to calculate the clustering distance
  matrix

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
   features `Xnew` having the same Scitype as `X` above.
- `transform(mach, Xnew)`: instead return the mean pairwise distances from
   new samples to the cluster centers

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
using Distances
using Test
KMeans = @load KMeans pkg=Clustering

X, y = @load_iris
model = KMeans(k=3)
mach = machine(model, X) |> fit!

preds = predict(mach, X)
@test preds == report(mach).assignments

center_dists = transform(mach, fitted_params(mach).centers')

@test center_dists[1][1] == 0.0
@test center_dists[2][2] == 0.0
@test center_dists[3][3] == 0.0
```

See also
[`KMedoids`](@ref)
"""
KMeans
"""
$(MMI.doc_header(KMedoids))

`KMedoids`: The K-Medoids algorithm finds K centroids corresponding to K clusters in the
data. Unlike K-Means, the centroids are found among data points themselves. Clusters
are not assumed to be elliptical. Should be used with a non-euclidean distance metric

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Count`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `k=3`: The number of centroids to use in clustering.
- `metric::SemiMetric=SqEuclidean`: The metric used to calculate the clustering distance
  matrix

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
   features `Xnew` having the same Scitype as `X` above.
- `transform(mach, Xnew)`: instead return the mean pairwise distances from
   new samples to the cluster centers

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
using Test
KMeans = @load KMedoids pkg=Clustering

X, y = @load_iris
model = KMedoids(k=3)
mach = machine(model, X) |> fit!

preds = predict(mach, X)
@test preds == report(mach).assignments

center_dists = transform(mach, fitted_params(mach).medoids')

@test center_dists[1][1] == 0.0
@test center_dists[2][2] == 0.0
@test center_dists[3][3] == 0.0
```

See also
[`KMeans`](@ref)
"""
KMedoids


end # module
