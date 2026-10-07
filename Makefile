# AI-Ready developer and release tasks.
# `make help` lists targets. Variables can be overridden: `make build VERSION=1.2.0 REGISTRY=registry.internal/aiready`.

SHELL          := /usr/bin/env bash
.SHELLFLAGS    := -euo pipefail -c
.DEFAULT_GOAL  := help

PYTHON         ?= .venv/bin/python
VERSION        ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo 0.0.0-dev)
GIT_COMMIT     ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
BUILD_DATE     ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
REGISTRY       ?= ghcr.io/aiready
IMAGE          ?= $(REGISTRY)/aiready
COMPOSE        ?= docker compose -f deploy/compose/docker-compose.yml --env-file deploy/compose/.env
CHART          ?= deploy/helm/aiready
COSIGN_KEY     ?= cosign.key
BUNDLE_OUT     ?= dist/bundle
EVAL_SUITES    ?= claims
EVAL_BASELINE  ?= latest
AIREADY        ?= $(dir $(PYTHON))aiready
# helm is optional locally: HELM="docker run --rm -v $$PWD:/repo -w /repo alpine/helm:3.16.2" make helm-lint
HELM           ?= helm

.PHONY: help install test lint fmt policy-test run-pilot run-pilot-gpu run-demo stop-pilot logs build push sign sbom scan \
        bundle verify-bundle helm-lint helm-template eval eval-gate tf-validate clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------------------ develop
install: ## Create .venv and install the package with all extras and dev tools
	test -x $(PYTHON) || python3.12 -m venv .venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[all,dev]"

test: ## Run the unit and contract tests
	$(PYTHON) -m pytest -q

lint: ## Static checks: ruff, mypy, shellcheck on deployment scripts, YAML parse of deploy assets
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests
	-$(PYTHON) -m mypy src
	@if command -v shellcheck >/dev/null; then shellcheck -x deploy/onprem/*.sh deploy/onprem/backup/*.sh deploy/compose/init-secrets.sh; fi
	$(PYTHON) deploy/scripts/check_yaml.py

fmt: ## Auto-format Python code
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

policy-test: ## Run the policy-as-code test cases
	$(AIREADY) policy test config/policies/tests

# ------------------------------------------------------------------------------------------ stage 1
deploy/compose/.env:
	cp deploy/compose/.env.example deploy/compose/.env
	@echo "Created deploy/compose/.env from the example - review it before exposing the pilot to users."

run-pilot: deploy/compose/.env ## Stage 1: start the pilot on this host (mock models, no GPU)
	./deploy/compose/init-secrets.sh
	$(COMPOSE) up -d --build
	@echo "Gateway http://localhost:8080  Assistant http://localhost:8130  Grafana http://localhost:3000"

run-pilot-gpu: deploy/compose/.env ## Stage 1 with one GPU node: vLLM chat + TEI embeddings/reranker
	./deploy/compose/init-secrets.sh
	AIREADY_MODEL_CHAT_URL=http://vllm-chat:8000/v1 AIREADY_MODEL_SMALL_URL=http://vllm-chat:8000/v1 \
	AIREADY_MODEL_EMBED_URL=http://tei-embed:80 AIREADY_MODEL_RERANK_URL=http://tei-rerank:80 \
	  $(COMPOSE) --profile gpu up -d --build

run-demo: deploy/compose/.env ## Pilot plus the sample legacy application
	./deploy/compose/init-secrets.sh
	$(COMPOSE) --profile demo up -d --build

stop-pilot: ## Stop the pilot (volumes are kept)
	$(COMPOSE) --profile gpu --profile demo down

logs: ## Follow pilot logs
	$(COMPOSE) logs -f --tail=100

# ------------------------------------------------------------------------------------------ release
build: ## Build the platform image
	docker build \
	  --build-arg VERSION=$(VERSION) --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg BUILD_DATE=$(BUILD_DATE) \
	  -t $(IMAGE):$(VERSION) .

push: build ## Push the image to $(REGISTRY)
	docker push $(IMAGE):$(VERSION)

sbom: ## Generate an SPDX SBOM for the image (needs syft)
	mkdir -p dist
	syft $(IMAGE):$(VERSION) -o spdx-json=dist/sbom-$(VERSION).spdx.json

scan: ## Vulnerability scan (needs trivy); fails on HIGH/CRITICAL fixable findings
	trivy image --exit-code 1 --ignore-unfixed --severity HIGH,CRITICAL $(IMAGE):$(VERSION)

sign: ## Sign the pushed image by digest and attach the SBOM attestation (needs cosign)
	DIGEST=$$(docker inspect --format '{{index .RepoDigests 0}}' $(IMAGE):$(VERSION)); \
	cosign sign --yes --key $(COSIGN_KEY) "$$DIGEST"; \
	if [ -f dist/sbom-$(VERSION).spdx.json ]; then \
	  cosign attest --yes --key $(COSIGN_KEY) --type spdxjson --predicate dist/sbom-$(VERSION).spdx.json "$$DIGEST"; fi

bundle: ## Build a signed air-gapped release bundle (images, chart, config, model manifest)
	./deploy/onprem/build-bundle.sh --version $(VERSION) --out $(BUNDLE_OUT) --platform-image $(IMAGE):$(VERSION) \
	  $(if $(MODELS),--models $(MODELS),)

verify-bundle: ## Verify a bundle: make verify-bundle BUNDLE=dist/bundle/aiready-bundle-X.tar PUBKEY=cosign.pub
	./deploy/onprem/verify-bundle.sh --bundle $(BUNDLE) --key $(PUBKEY)

# ------------------------------------------------------------------------------------------ stage 2/3
helm-lint: ## Lint and render the Helm chart with every values file
	$(HELM) lint $(CHART)
	for f in values-pilot.yaml values-production.yaml values-platform.yaml; do \
	  $(HELM) lint $(CHART) -f $(CHART)/$$f; \
	  $(HELM) template aiready $(CHART) -n aiready -f $(CHART)/$$f > /dev/null; \
	done

helm-template: ## Render the production manifests to dist/manifests.yaml
	mkdir -p dist
	$(HELM) template aiready $(CHART) -n aiready -f $(CHART)/values-production.yaml > dist/manifests.yaml

tf-validate: ## Format-check and validate the AWS Terraform
	terraform -chdir=deploy/terraform/aws fmt -check -recursive
	terraform -chdir=deploy/terraform/aws init -backend=false -input=false
	terraform -chdir=deploy/terraform/aws validate

# ------------------------------------------------------------------------------------------ quality gate
eval: ## Run the evaluation suites (golden + safety) and write scorecards
	$(AIREADY) eval run $(EVAL_SUITES) --release $(VERSION) --safety

eval-gate: ## Release gate: exits non-zero if the score drops vs the baseline or a safety case fails
	$(AIREADY) eval gate $(EVAL_SUITES) --release $(VERSION) --baseline $(EVAL_BASELINE)

clean: ## Remove build outputs
	rm -rf dist build .pytest_cache .ruff_cache
