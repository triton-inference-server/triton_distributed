package controllers

import (
	"fmt"

	"github.com/gin-gonic/gin"

	"github.com/rs/zerolog/log"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/converters"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/services"
)

type clusterController struct{}

var ClusterController = clusterController{}

func (c *clusterController) Create(ctx *gin.Context) {
	var schema schemas.CreateClusterSchema
	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := services.ClusterService.Create(ctx, services.CreateClusterOption{
		Name:        schema.Name,
		Description: schema.Description,
		KubeConfig:  schema.KubeConfig,
	})

	if err != nil {
		log.Info().Msgf("Failed to create cluster: %s", err.Error())
		ctx.JSON(400, gin.Error{Err: err})
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) Update(ctx *gin.Context) {
	var schema schemas.UpdateClusterSchema
	clusterName := ctx.Param("clusterName")

	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := services.ClusterService.GetByName(ctx, clusterName)
	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find cluster with the name %s", clusterName))
		return
	}

	cluster, err = services.ClusterService.Update(ctx, cluster, services.UpdateClusterOption{
		Description: schema.Description,
		KubeConfig:  schema.KubeConfig,
	})

	if err != nil {
		log.Info().Msgf("Failed to update cluster: %s", err.Error())
		ctx.JSON(400, "Error updating cluster.")
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) Get(ctx *gin.Context) {
	clusterName := ctx.Param("clusterName")

	cluster, err := services.ClusterService.GetByName(ctx, clusterName)
	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find cluster with the name %s", clusterName))
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) List(ctx *gin.Context) {
	var schema schemas.ListQuerySchema

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	clusters, total, err := services.ClusterService.List(ctx, services.ListClusterOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
	})
	if err != nil {
		log.Info().Msgf("Failed to list clusters: %s", err.Error())
		ctx.JSON(400, gin.H{"Error": fmt.Sprintf("List clusters %s", err.Error())})
		return
	}

	clusterList := schemas.ClusterListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Start: schema.Start,
			Count: schema.Count,
			Total: total,
		},
		Items: converters.ToClusterSchemaList(clusters),
	}

	ctx.JSON(200, clusterList)
}
