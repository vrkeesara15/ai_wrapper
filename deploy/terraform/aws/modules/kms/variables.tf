variable "name" {
  description = "Name prefix."
  type        = string
}

variable "region" {
  description = "AWS region (for the CloudWatch Logs service principal)."
  type        = string
}

variable "account_id" {
  description = "AWS account id that administers the keys."
  type        = string
}

variable "deletion_window_days" {
  description = "Waiting period before a scheduled key deletion takes effect."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags applied to every key."
  type        = map(string)
  default     = {}
}
