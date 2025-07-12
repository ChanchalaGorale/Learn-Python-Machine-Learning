## AI‑Driven Social‑Media Platform for Short‑Form Video, Images & Chat

---

#### 1 . Business Context

A start‑up wants to launch **SocialPulse**, a TikTok‑style mobile/web app where users post 15‑60 sec videos, images, and text updates. To compete, the platform must:

* **Moderate content** (NSFW, violence, hate speech) in real time.
* **Personalise the “For You” feed** per user across modalities.
* **Generate smart captions & alt‑text** automatically to boost accessibility and SEO.
* **Detect trending topics** quickly and recommend hashtags.
* **Offer creator analytics dashboards** (reach, engagement, predicted virality).

Target launch: 6 months. Expected load: 50 k DAU at launch, scaling to 1 M within a year.

---

#### 2 . Requirements & Assumptions

| Category           | Details                                                                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Functional**     | Upload & stream video (≤150 MB), images, text; like/share/comment; personalised feed; live search; real‑time moderation pipeline; LLM‑generated captions; push notifications. |
| **Non‑Functional** | P99 feed latency < 300 ms; 99.9 % uptime; GDPR/CCPA compliant; autoscale to 5× daily traffic; blue‑green or canary deployment.                                                |
| **Assumptions**    | Cloud budget allows managed GPU nodes; multi‑region (US‑East, EU‑West, AP‑South); initial team of 8 engineers (2 ML, 2 backend, 2 frontend, 1 DevOps, 1 PM).                  |

---

#### 3 . High‑Level Architecture

```
 Clients (iOS/Android/Web: React Native + React)
          │
   ┌──────┴────────────────┐
   │ Global API Gateway    │  (GraphQL over HTTP/2, gRPC for internal calls)
   └──────┬────────────────┘
          │
┌─────────┴─────────────────────────────────────────────────────────────────┐
│  Microservices on Kubernetes (EKS / GKE)                                 │
│                                                                         │
│  • Auth Service (OAuth2 / OpenID)                                        │
│  • Media Uploader  ──┐                                                   │
│  • Content Moderation│(Kafka topic: media_uploaded)                      │
│  • Caption Gen (LLM) │                                                   │
│  • Feed Ranker (online)                                                  │
│  • Notification Service                                                  │
│  • Analytics API                                                         │
│  • Admin Dashboard (Next.js)                                             │
└─────────┬─────────────────────────────────────────────────────────────────┘
          │
┌─────────┴───────────────┐    ┌──────────────────────┐
│   Object Storage (S3)   │    │  Feature / Vector DB │
│  (videos, images)       │    │   (Weaviate / Pinecone) │
└─────────┬───────────────┘    └─────────┬────────────┘
          │                               │
┌─────────┴─────────┐          ┌──────────┴─────────┐
│ Relational DB     │          │ Time‑Series DB     │
│ PostgreSQL (users │          │ TimescaleDB / Influx│
│ , follows, posts) │          │ (views, likes)      │
└───────────────────┘          └─────────────────────┘
```

Event backbone: **Apache Kafka** (Confluent Cloud) with topics such as `media_uploaded`, `moderation_passed`, `user_interaction`, `model_metrics`.

---

#### 4 . Tech‑Stack Decisions & Rationale

| Layer                  | Choice                                                                               | Why                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Frontend**           | React Native (mobile), React + Next.js (web)                                         | One code‑base, SSR for SEO; Next.js Edge functions for low latency                                    |
| **API**                | **GraphQL** (Hasura) + **FastAPI** fallback                                          | GraphQL gives flexible feed queries; FastAPI for ML micro‑APIs                                        |
| **Streaming**          | Apache Kafka (managed)                                                               | High‑throughput pub/sub; decouples upload, moderation, ranking                                        |
| **Databases**          | PostgreSQL (OLTP), S3, Weaviate (vector), TimescaleDB (metrics)                      | Strong consistency for social graph; cheap immutable media; ANN search; efficient time‑series queries |
| **ML Frameworks**      | PyTorch 2.0 + Hugging Face + LightGBM                                                | GPU acceleration; hub of vision & NLP models; fast GBM for tabular ranking                            |
| **Content Moderation** | Pre‑trained CLIP, ViT, and DistilBERT fine‑tuned on open NSFW & hate‑speech datasets | Covers image, video frames, and text captions                                                         |
| **Feed Ranking**       | Multi‑task Wide & Deep (TensorFlow Recommenders) or DeepFM                           | Handles sparse categorical + dense behavioural features                                               |
| **Caption/Alt‑Text**   | Llama‑3‑Instruct 8 B fine‑tuned with LoRA                                            | On‑premise model avoids per‑call costs; LoRA keeps VRAM lower                                         |
| **Feature Store**      | Feast on Snowflake / BigQuery                                                        | Online + offline parity; push features to Redis for low‑latency                                       |
| **MLOps**              | **Kubeflow Pipelines** (training), **MLflow** (tracking), **Evidently AI** (drift)   | Reproducible DAGs; version models; monitor quality                                                    |
| **CI/CD**              | GitHub Actions → Docker → ArgoCD                                                     | Declarative roll‑outs; preview environments per PR                                                    |
| **Observability**      | Prometheus + Grafana + Loki                                                          | Unified infra & app metrics/logs                                                                      |
| **Security**           | AWS Cognito / Auth0, Vault, IAM least‑privilege                                      | Managed identity; secret rotation                                                                     |

---

#### 5 . Data & ML Pipelines

| Stage                   | Tool / Lib                                           | Key Steps                                                                                                     |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Ingestion**           | Kafka Connect, S3 Events                             | Push video frames to `moderation` topic                                                                       |
| **Moderation Pipeline** | Triton Inference Server                              | 1️⃣ Split video → frames (ffmpeg) → ViT NSFW classifier  2️⃣ Text → DistilBERT toxicity  3️⃣ Aggregator rules |
| **Feature Engineering** | Spark Structured Streaming → Feast                   | Compute 30‑day engagement rates, user embeddings (Word2Vec on follow graph)                                   |
| **Model Training**      | Kubeflow → PyTorch/TFRS                              | Hourly incremental training; store artefacts in MLflow                                                        |
| **Online Serving**      | BentoML container (Feed Ranker) ✓ Autoscales via HPA | Feature look‑ups from Redis; returns ordered post IDs in < 50 ms                                              |
| **Caption LLM Service** | Text‑generation‑inference (TGI)                      | Async; generates alt‑text & trending hashtags                                                                 |
| **Feedback Loop**       | DataHub + Evidently AI                               | Compare prod vs train distribution; trigger retrain if drift > 5 %                                            |

---

#### 6 . Deployment Strategy

* **Docker‑files per microservice** (multi‑stage builds; distroless).
* **Helm charts** for each service; environment values (dev/stage/prod).
* **ArgoCD** watches `gitops/prod` branch → applies Helm → canary 10 % → 50 % → 100 % using Argo Rollouts.
* **GPU node‑pools** for Triton and TGI pods; CPU pools for others.
* **Autoscaling**:
  \* HPA on CPU (Feed Ranker) and GPU‑util (Moderation).
  \* Cluster‑Autoscaler adds nodes.

---

#### 7 . Monitoring & Maintenance

| Aspect             | Tool                          | Alert Examples                   |
| ------------------ | ----------------------------- | -------------------------------- |
| **Infrastructure** | Prometheus                    | CPU > 80 % 5 min, pod restarts   |
| **Model Perf**     | MLflow dashboards + Evidently | AUC drop > 3 % vs baseline       |
| **User KPIs**      | Amplitude / Mixpanel          | Retention D1, avg session length |
| **Cost**           | Kubecost                      | GPU spend > \$200/day            |
| **Incidents**      | PagerDuty                     | P0: feed latency > 500 ms        |

---

#### 8 . Q & A – Design Decisions Explained

| Question                                       | Answer / Rationale                                                                                                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Why GraphQL?**                               | Feed & profile pages require flexible, nested fetches. GraphQL prevents over/under‑fetching and lets clients evolve without new endpoints.                                  |
| **Why Kafka vs SQS/PubSub?**                   | Exactly‑once semantics with idempotent producers, large backlogs for media; replay for retraining; mature ecosystem (Kafka‑Streams, ksqlDB).                                |
| **Why separate vector DB?**                    | Personalized ranking needs fast ANN search over user/content embeddings. PostgreSQL or Redis alone can’t meet latency at 1 M TPS.                                           |
| **Why Feast as feature store?**                | Guarantees offline–online feature parity; integrates with both Spark (offline) and Redis (online) out of the box.                                                           |
| **Why GPU Triton vs plain Flask + PyTorch?**   | Triton batches requests across users, supports model ensembles, and exports Prometheus metrics; 40–60 % cheaper at scale.                                                   |
| **Why LoRA fine‑tuning?**                      | Keeps base weights frozen; only \~0.1 % parameters trained ⇒ lower compute, faster iteration; easier rollback.                                                              |
| **Why canary over blue‑green?**                | Gradual ramp‑up lets us watch real‑user metrics (CTR, dwell time) and abort if degradation; smaller infra footprint than parallel full stacks.                              |
| **Why TimescaleDB over DynamoDB for metrics?** | Native time‑series functions, continuous aggregates, powerful SQL; write‑heavy ingest (>100 k events/s) handled with hypertables + compression.                             |
| **How do we ensure privacy & compliance?**     | Encrypt data at rest (S3 SSE‑KMS, RDS AES‑256) and in transit (TLS 1.3). Fine‑grained IAM; audit logs (AWS CloudTrail). Data deletion API for GDPR “right to be forgotten”. |

---

#### 9 . Roadmap (Next 12 Months)

1. **M 3:** MVP feed, moderation, caption LLM.
2. **M 5:** Creator analytics, trending detection.
3. **M 7:** Multi‑region fail‑over, AB‑testing harness.
4. **M 9:** On‑device lightweight recommender for offline mode.
5. **M 12:** Real‑time live‑stream moderation & voice‑to‑text search.

---

### ✅ Outcome

The design provides:

* **Scalable, low‑latency feed & upload flow** with event streaming.
* **Robust ML/GenAI services** that moderate, caption, and rank content.
* **End‑to‑end MLOps** for drift detection, retraining, and safe roll‑outs.
* **Clear separation of concerns** (microservices) yet cost‑efficient (shared K8s cluster).
* **Compliance & observability** baked in from day 1.