# FEALPy manual

## Procedure

The procedure of FEALPy

```mermaid
graph LR;
  exampleRate --> exampleFEMModel
  exampleFEMModel --> exampleSpace
  exampleSpace --> exampleDof
```

For example, in the `PoissonFEMRate.py`,  

The important thing is to get the 

- (Gauss) integral points and weights.
- The values of basis-functions at integral points.
- The values of gradient-basis-functions at integral points.



```mermaid
graph LR
style A1 fill:#f9f,stroke:#333,stroke-width:0px
style A2 fill:#f9f,stroke:#333,stroke-width:0px
A1[Gauss integral points and weights] --- B1["basis function"]
A1 --- B2["gradient basis function"]

B1 --- C1["mass matrix"] 
A2["cell to dof"] --- C1
B2 --- C2["stiff mat"] 
A2 --- C2

```

