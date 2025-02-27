package models

import (
	"time"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
)

type Deployment struct {
	Resource
	ClusterAssociate

	Description     string                   `json:"description"`
	Status          schemas.DeploymentStatus `json:"status"`
	StatusSyncingAt *time.Time               `json:"status_syncing_at"`
	StatusUpdatedAt *time.Time               `json:"status_updated_at"`
	KubeDeployToken string                   `json:"kube_deploy_token"`
	KubeNamespace   string                   `json:"kube_namespace"`
}

func (d *Deployment) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeDeployment
}
