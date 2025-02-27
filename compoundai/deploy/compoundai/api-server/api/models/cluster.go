package models

type Cluster struct {
	Resource

	Description string `json:"description"`
	KubeConfig  string `json:"kube_config"`
}
