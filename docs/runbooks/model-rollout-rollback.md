# Model rollout and rollback

**Owner:** AI engineer (evaluation, approval), with the platform engineer (serving).
**Principle:** every server speaks the same OpenAI-compatible API, so changing a model is a routing change.
The new version runs beside the old one; traffic moves to it only after it passes the gate; rollback is one
change.

## Roll out a new model version (or adapter)

1. **Weights into the model store.** Cloud: sync to S3, then the model-store PVC. On-prem: the signed bundle
   (`aiready bundle verify` then `aiready bundle import`). The store must contain `<name>/<revision>/SHA256SUMS`.
2. **Register.**
   `aiready registry register model <name> <version> --licence <id> --context-length <n> --base <base>`
3. **Evaluate offline against the current version:**
   `aiready eval gate <suites> --release <name>-<version> --baseline latest`
   Scores must not drop, safety cases must pass, and cost and latency must stay within budget. For adapters,
   also run the leak test: no training data may be echoed back.
4. **Approve.** A person approves in the registry: `aiready registry approve model <name> <version>`
   (`POST /v1/registry/models/{name}/{version}/approve`).
5. **Canary deployment.** Set `modelServing.pools.<pool>.models[].canary`:
   ```bash
   helm upgrade aiready deploy/helm/aiready -n aiready --reuse-values \
     --set 'modelServing.pools.chat.models[0].canary.enabled=true' \
     --set 'modelServing.pools.chat.models[0].canary.path=chat/v2'
   ```
   The canary Deployment `model-chat-canary` verifies its weights, then joins the `model-chat` Service.
   Traffic splits in proportion to replica counts. To set an exact percentage, use a weighted route in the
   gateway to the `model-chat-canary` Service (`config/models.yaml`; it is hot-reloaded).
6. **Watch** for at least one business cycle: TTFT, error rate, live quality scores, spend (Grafana
   "AI-Ready overview"), and the review queue.
7. **Promote.** Set the stable `path` to the new revision, disable the canary, and run `helm upgrade`. The
   rolling update keeps the old pods until the new ones are ready. This needs one spare GPU per model;
   otherwise promote during a quiet window.

## Roll back

* **Canary bad:** disable the canary (`canary.enabled=false`) or remove the weighted route. One change;
  traffic returns at once.
* **Promoted version bad:** run `helm rollback aiready <previous revision> -n aiready`, or set `path` back. The
  previous weights are still in the store and verified again on start. Also deprecate the version:
  `aiready registry deprecate model <name> <version>`.
* **Adapter bad:** remove it from `adapters:`, or route its task back to the base model (a gateway route
  change). The base model is untouched.

Record: the version, the gate scorecard link, the approver, the canary window, and the decision.
