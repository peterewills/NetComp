# NetComp Status and TODO

### September 24, 2017

## STATUS

Currently the major metrics are implemented. Resistance and lambda distances have been verified to be accurate; I have moderate confidence in the accuracy of DeltaCon and NetSimile metrics.

The next large and essential component of the package is the implementation of the fast (linear time) distances. I would also like to add more distances, as I learn of more useful ones.

The handling of sparse matrices is, as it stands, spotty and not very intelligent. To improve performance, I need to ensure that we are using sparse matrices wherever possible and that all functionality works smoothly with sparse (as well as dense) matrices.

## TODO

### Coding

- Code up fast resistance metric
- Code up fast DeltaCon metric
- Make $\lambda_k$ distances use fast solver. (Right now, we find all eigenvalues then throw out most of them)
- Think about how we handle sparse vs dense matrices. Can we do this more intelligently to speed up our calculations? 


### Tests

- Test speed on all metrics. Use "sparse" graphs, i.e. graphs with order $n \log(n)$ edges. Probably just do ER graphs, for simplicity.
- Test accuracy on approximate metrics
    - fast vs full resistance
    - fast vs full deltacon
    - partial vs total eigenvalue comparison