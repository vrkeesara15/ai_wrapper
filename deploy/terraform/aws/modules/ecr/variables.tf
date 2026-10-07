variable "name" {
  description = "Repository name prefix."
  type        = string
}

variable "repositories" {
  description = "Repository names."
  type        = list(string)
}

variable "data_kms_key_arn" {
  description = "KMS key for image encryption."
  type        = string
}

variable "keep_images" {
  description = "Tagged images kept per repository."
  type        = number
  default     = 50
}

variable "tags" {
  description = "Tags."
  type        = map(string)
  default     = {}
}
