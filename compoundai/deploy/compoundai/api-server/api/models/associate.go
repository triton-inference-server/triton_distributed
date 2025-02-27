package models

type ClusterAssociate struct {
	ClusterId              uint     `json:"cluster_id"`
	AssociatedClusterCache *Cluster `gorm:"foreignkey:ClusterId"`
}

type DeploymentAssociate struct {
	DeploymentId              uint        `json:"deployment_id"`
	AssociatedDeploymentCache *Deployment `gorm:"foreignkey:DeploymentId;constraint:OnDelete:CASCADE;"`
}

type DeploymentRevisionAssociate struct {
	DeploymentRevisionId              uint                `json:"deployment_revision_id"`
	AssociatedDeploymentRevisionCache *DeploymentRevision `gorm:"foreignkey:DeploymentRevisionId;constraint:OnDelete:CASCADE;"`
}

type CompoundAINimAssociate struct {
	CompoundAINimId  uint   `json:"compoundai_nim_id"`
	CompoundAINimTag string `json:"compoundai_nim_tag"`
}

type DMSAssociate struct {
	KubeRequestId    string
	KubeDeploymentId string
}
