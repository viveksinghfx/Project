variable "vpc_name"       { type = string }
variable "vpc_cidr"       { type = string }
variable "azs"            { type = list(string) }
variable "public_subnets" { type = list(string) }
variable "private_subnets"{ type = list(string) }
variable "cluster_name"   { type = string }
variable "tags"           { type = map(string); default = {} }
