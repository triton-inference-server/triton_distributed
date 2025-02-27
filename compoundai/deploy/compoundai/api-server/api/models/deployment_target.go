package models

import "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"

type DeploymentTarget struct {
	BaseModel
	DeploymentAssociate
	DeploymentRevisionAssociate
	CompoundAINimAssociate
	DMSAssociate

	Config *schemas.DeploymentTargetConfig `json:"config"`
}

func (s *DeploymentTarget) GetName() string {
	return s.Uid.String()
}

func (s *DeploymentTarget) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeDeploymentRevision
}
