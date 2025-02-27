package schemasv2

import "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"

type DeploymentSchema struct {
	schemas.ResourceSchema
	Creator        *schemas.UserSchema               `json:"creator"`
	Cluster        *ClusterSchema                    `json:"cluster"`
	Status         schemas.DeploymentStatus          `json:"status" enum:"unknown,non-deployed,running,unhealthy,failed,deploying"`
	URLs           []string                          `json:"urls"`
	LatestRevision *schemas.DeploymentRevisionSchema `json:"latest_revision"`
	KubeNamespace  string                            `json:"kube_namespace"`
}

type GetDeploymentSchema struct {
	DeploymentName string `uri:"deploymentName" binding:"required"`
}

type CreateDeploymentSchema struct {
	UpdateDeploymentSchema
	Name string `json:"name"`
}

type UpdateDeploymentSchema struct {
	DeploymentConfigSchema
	Bento string `json:"bento"`
}

type DeploymentConfigSchema struct {
	AccessAuthorization bool                   `json:"access_authorization"`
	Envs                interface{}            `json:"envs,omitempty"`
	Secrets             interface{}            `json:"secrets,omitempty"`
	Services            map[string]ServiceSpec `json:"services"`
}

type ServiceSpec struct {
	Scaling         ScalingSpec         `json:"scaling"`
	ConfigOverrides ConfigOverridesSpec `json:"config_overrides"`
}

type ScalingSpec struct {
	MinReplicas int `json:"min_replicas"`
	MaxReplicas int `json:"max_replicas"`
}

type ConfigOverridesSpec struct {
	Resources schemas.Resources `json:"resources"`
}
