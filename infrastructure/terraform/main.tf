# =============================================================================
# GCP Cloud Run deployment for the Structured Extraction Pipeline
#
# Resources:
#   - Cloud Run service with auto-scaling (0–10 instances)
#   - Secret Manager secrets for API keys
#   - IAM bindings for secret access
#
# Usage:
#   terraform init
#   terraform plan -var="project_id=my-project" -var="openai_api_key=sk-..."
#   terraform apply
# =============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "image" {
  description = "Container image URL (e.g. gcr.io/project/extraction-pipeline:latest)"
  type        = string
}

variable "database_url" {
  description = "PostgreSQL connection string"
  type        = string
  sensitive   = true
}

variable "redis_url" {
  description = "Redis connection string"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "courtlistener_api_key" {
  description = "CourtListener API key"
  type        = string
  sensitive   = true
  default     = ""
}

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Enable required APIs
# ---------------------------------------------------------------------------

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Secret Manager — API keys
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "openai_key" {
  secret_id = "extraction-pipeline-openai-key-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "openai_key" {
  secret      = google_secret_manager_secret.openai_key.id
  secret_data = var.openai_api_key
}

resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "extraction-pipeline-anthropic-key-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "anthropic_key" {
  secret      = google_secret_manager_secret.anthropic_key.id
  secret_data = var.anthropic_api_key
}

resource "google_secret_manager_secret" "courtlistener_key" {
  secret_id = "extraction-pipeline-courtlistener-key-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "courtlistener_key" {
  secret      = google_secret_manager_secret.courtlistener_key.id
  secret_data = var.courtlistener_api_key
}

# ---------------------------------------------------------------------------
# Cloud Run service
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "pipeline" {
  name     = "extraction-pipeline-${var.environment}"
  location = var.region

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = var.image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      # Health / liveness probes
      startup_probe {
        http_get {
          path = "/api/v1/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/api/v1/health"
          port = 8000
        }
        period_seconds = 30
      }

      # Environment variables
      env {
        name  = "API_HOST"
        value = "0.0.0.0"
      }
      env {
        name  = "API_PORT"
        value = "8000"
      }
      env {
        name  = "LOG_FORMAT"
        value = "json"
      }
      env {
        name  = "LOG_LEVEL"
        value = var.environment == "prod" ? "INFO" : "DEBUG"
      }
      env {
        name  = "DEBUG"
        value = var.environment == "prod" ? "false" : "true"
      }

      # Secrets from Secret Manager
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = "extraction-pipeline-db-url-${var.environment}"
            version = "latest"
          }
        }
      }
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "COURTLISTENER_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.courtlistener_key.secret_id
            version = "latest"
          }
        }
      }

      # Non-secret config
      env {
        name  = "REDIS_URL"
        value = var.redis_url
      }
    }

    service_account = google_service_account.pipeline.email
  }

  depends_on = [google_project_service.run]
}

# ---------------------------------------------------------------------------
# Service account + IAM
# ---------------------------------------------------------------------------

resource "google_service_account" "pipeline" {
  account_id   = "extraction-pipeline-${var.environment}"
  display_name = "Extraction Pipeline (${var.environment})"
}

resource "google_secret_manager_secret_iam_member" "openai_access" {
  secret_id = google_secret_manager_secret.openai_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_secret_manager_secret_iam_member" "anthropic_access" {
  secret_id = google_secret_manager_secret.anthropic_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_secret_manager_secret_iam_member" "courtlistener_access" {
  secret_id = google_secret_manager_secret.courtlistener_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipeline.email}"
}

# Allow unauthenticated access (public API) — remove for internal-only services
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.environment == "dev" ? 1 : 0
  name     = google_cloud_run_v2_service.pipeline.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.pipeline.uri
}

output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.pipeline.name
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.pipeline.email
}
