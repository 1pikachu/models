# Kubernetes Dev Ops

## MPI Operator Deployment

The following instructions explain how to deploy the MPI Operator to a Kubernetes
cluster. This deployment needs to be done by a devops user with Cluster RBAC permissions.
The steps below explain how to deploy the mpi-operator on your cluster using version `v0.2.3`.

1. Download the `mpi-operator.yaml` file
   ```
   curl -o mpi-operator.yaml https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
   ```

2. Change line #199 from `image: mpioperator/mpi-operator:latest` to `image: mpioperator/mpi-operator:v0.2.3`

3. Deploy the operator
   ```
   kubectl apply -f mpi-operator.yaml
   ```

For more information on deploying the mpi-operator to the k8s cluster, see the
[documentation in the mpi-operator repo](https://github.com/kubeflow/mpi-operator#mpi-operator).
