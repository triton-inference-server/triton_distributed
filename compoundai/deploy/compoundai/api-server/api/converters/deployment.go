package converters

import (
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemasv2"
)

func ToDeploymentSchemaV2(cluster *models.Cluster, deployment *models.Deployment, creator *schemas.UserSchema) *schemasv2.DeploymentSchema {
	clusterSchema := ToClusterSchemaV2(cluster, creator)

	return &schemasv2.DeploymentSchema{
		ResourceSchema: schemas.ResourceSchema{
			Name:         deployment.Resource.Name,
			Labels:       []string{},
			ResourceType: deployment.GetResourceType(),
			BaseSchema: schemas.BaseSchema{
				Uid:       deployment.Resource.BaseModel.GetUid(),
				CreatedAt: deployment.Resource.BaseModel.CreatedAt,
				UpdatedAt: deployment.Resource.BaseModel.UpdatedAt,
				DeletedAt: nil, // Can assume that this is nil during creation
			},
		}, // Assuming ResourceSchema can be copied directly
		Creator:        creator,
		Cluster:        clusterSchema,
		Status:         deployment.Status,
		URLs:           []string{},
		LatestRevision: nil,
		KubeNamespace:  deployment.KubeNamespace,
	}
}
