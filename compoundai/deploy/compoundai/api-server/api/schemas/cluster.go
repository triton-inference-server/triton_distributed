package schemas

type ClusterSchema struct {
	ResourceSchema
	Description string `json:"description"`
}

type ClusterListSchema struct {
	BaseListSchema
	Items []*ClusterSchema `json:"items"`
}

type ClusterFullSchema struct {
	ClusterSchema
	KubeConfig *string `json:"kube_config"`
}

type UpdateClusterSchema struct {
	Description *string `json:"description"`
	KubeConfig  *string `json:"kube_config"`
}

type CreateClusterSchema struct {
	Description string `json:"description"`
	KubeConfig  string `json:"kube_config"`
	Name        string `json:"name"`
}

type GetClusterSchema struct {
	ClusterName string `uri:"clusterName" binding:"required"`
}
