package converters

import (
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemasv2"
)

func ToClusterSchemaList(clusters []*models.Cluster) []*schemas.ClusterSchema {
	clusterSchemas := make([]*schemas.ClusterSchema, 0)

	for _, cluster := range clusters {
		clusterSchemas = append(clusterSchemas, ToClusterSchema(cluster))
	}

	return clusterSchemas
}

func ToClusterSchema(cluster *models.Cluster) *schemas.ClusterSchema {
	return &schemas.ClusterSchema{
		Description: cluster.Description,
		ResourceSchema: schemas.ResourceSchema{
			Name:         cluster.Name,
			ResourceType: schemas.ResourceTypeCluster,
			BaseSchema: schemas.BaseSchema{
				Uid:       cluster.GetUid(),
				CreatedAt: cluster.CreatedAt,
				UpdatedAt: cluster.UpdatedAt,
				DeletedAt: &cluster.DeletedAt.Time,
			},
		},
	}
}

func ToClusterFullSchema(cluster *models.Cluster) *schemas.ClusterFullSchema {
	clusterSchema := ToClusterSchema(cluster)

	return &schemas.ClusterFullSchema{
		ClusterSchema: *clusterSchema,
		KubeConfig:    &cluster.KubeConfig,
	}
}

func ToClusterSchemaV2(cluster *models.Cluster, creator *schemas.UserSchema) *schemasv2.ClusterSchema {
	return &schemasv2.ClusterSchema{
		Description:      cluster.Description,
		OrganizationName: "nvidia",
		Creator:          creator,
		ResourceSchema: schemas.ResourceSchema{
			Name:   cluster.Name,
			Labels: []string{},
			BaseSchema: schemas.BaseSchema{
				Uid:       cluster.Resource.BaseModel.GetUid(),
				CreatedAt: cluster.Resource.BaseModel.CreatedAt,
				UpdatedAt: cluster.Resource.BaseModel.UpdatedAt,
				DeletedAt: nil, // Can assume that this is nil during creation
			},
		},
	}
}
