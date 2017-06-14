# NetComp, Current Status

## Big TODOs

### Coding

- Code up fast resistance metric
- Code up fast DeltaCon metric

### Tests

- Test speed on all metrics
	- use "sparse" graphs, i.e. graphs with order $n \log(n)$ edges. probably just do ER graphs, for simplicity.
- Test accuracy on approximate metrics
    - fast vs full resistance
    - fast vs full deltacon
    - partial vs total eigenvalue comparison (not sure if this is relevant anymore, we might calculate all eigenvalues anyways)

### Design    

- Think about how we handle sparse vs dense matrices.
	- can we do this more intelligently to speed up our calculations? 