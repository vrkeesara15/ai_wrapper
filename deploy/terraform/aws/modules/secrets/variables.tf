variable "name" {
  description = "Name prefix (secrets are stored as <name>/<secret>)."
  type        = string
}

variable "data_kms_key_arn" {
  description = "KMS key encrypting the secrets."
  type        = string
}

variable "extra_secret_names" {
  description = "Additional empty secrets to create (filled out of band), e.g. connector credentials."
  type        = list(string)
  default     = []
}

variable "recovery_window_days" {
  description = "Days a deleted secret can still be recovered."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
