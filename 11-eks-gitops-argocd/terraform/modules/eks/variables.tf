variable "cluster_name"        { type = string }
variable "kubernetes_version"  { type = string; default = "1.30" }
variable "vpc_id"              { type = string }
variable "subnet_ids"          { type = list(string) }
variable "private_subnet_ids"  { type = list(string) }
variable "node_instance_types" { type = list(string); default = ["t3.medium"] }
variable "desired_nodes"       { type = number; default = 2 }
variable "min_nodes"           { type = number; default = 1 }
variable "max_nodes"           { type = number; default = 5 }
variable "environment"         { type = string }
variable "tags"                { type = map(string); default = {} }
