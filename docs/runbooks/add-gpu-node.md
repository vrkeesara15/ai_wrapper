# Adding a GPU node

**Owner:** platform engineer. **When:** queue time or requests waiting stay above budget while the pool is
at its KEDA `maxReplicas`, GPU memory is saturated, or the sizing worksheet (`aiready sizing worksheet`) says
so. Never add capacity on a guess: compare measured tokens/s per node with demand.

## Cloud (EKS, Terraform)

1. Raise the node group size in `terraform.tfvars` (`node_groups.gpu_a.max_size` / `desired_size`, or
   `gpu_b`), then `terraform plan` and `terraform apply`. In stage 3 it is usually enough to raise
   `max_size`, because KEDA plus the cluster autoscaler add nodes on demand.
2. Raise the pool's ceiling in values: `modelServing.pools.<pool>.models[].keda.maxReplicas`.
3. Run `helm upgrade`.

## On-prem

1. Rack the node and install the OS image that has the NVIDIA driver version pinned for the cluster. Join it
   to the cluster.
2. Label and taint it to match the pool:
   ```bash
   kubectl label node <node> aiready.io/pool=gpu-a      # or gpu-b
   kubectl taint node <node> nvidia.com/gpu=true:NoSchedule
   ```
3. Check that the GPU operator has advertised it: `kubectl describe node <node> | grep nvidia.com/gpu`. For
   `gpu-b`, apply the time-slicing / MIG profile the pool uses.
4. Warm the model cache (optional): the `verify-weights` init container copies and verifies the weights on
   first schedule. Large weights can take minutes, which is why each pool keeps one replica warm.

## Verify

* `kubectl -n aiready get pods -l aiready.io/pool=chat -o wide` shows the new replicas on the node.
* `vllm:num_requests_waiting` and queue time are back under budget, and `DCGM_FI_DEV_FB_USED` is below 95 %.
* Run a load test at the next change window and record the measured tokens/s per node in the sizing
  worksheet.
