package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/converters"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/database"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemasv2"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/services"
)

type deploymentController struct{}

var DeploymentController = deploymentController{}

type CreateDeploymentSchema struct {
	schemas.CreateDeploymentSchema
}

func (c *deploymentController) Create(ctx *gin.Context) {
	clusterName := ctx.Param("clusterName")
	var schema CreateDeploymentSchema

	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := services.ClusterService.GetByName(ctx, clusterName)
	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find cluster with the name %s", clusterName))
		return
	}

	deployment, err := c.createDeploymentHelper(ctx, cluster, schema)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deployment)
}

func (c *deploymentController) createDeploymentHelper(ctx *gin.Context, cluster *models.Cluster, schema CreateDeploymentSchema) (*models.Deployment, error) {
	description := ""
	if schema.Description != nil {
		description = *schema.Description
	}

	_, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err := services.DeploymentService.Create(ctx_, services.CreateDeploymentOption{
		ClusterId:     cluster.ID,
		Name:          schema.Name,
		Description:   description,
		KubeNamespace: schema.KubeNamespace,
	})
	if err != nil {
		log.Error().Msgf("Creating deployment failed: %s", err.Error())
		return nil, fmt.Errorf("creating deployment failed: %s", err.Error())
	}

	_, err = c.updateDeploymentEntities(ctx_, schema.UpdateDeploymentSchema, deployment)
	if err != nil {
		log.Error().Msgf("Failed to update deployment %s entities %s", deployment.Name, err.Error())
		return nil, fmt.Errorf("failed to update deployment %s entities %s", deployment.Name, err.Error())
	}

	log.Info().Msgf("CREATED DEPLOYMENT: %+v", *deployment)
	return deployment, nil
}

func (c *deploymentController) SyncStatus(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	status, err := services.DeploymentService.SyncStatus(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to sync deployment %s status: %s", deployment.Name, err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment.Status = status

	ctx.JSON(200, deployment)
}

func (c *deploymentController) Update(ctx *gin.Context) {
	var updateSchema schemas.UpdateDeploymentSchema
	var getSchema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&getSchema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindJSON(&updateSchema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &getSchema)

	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find deployment with the name %s", getSchema.DeploymentName))
		return
	}

	_, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err = services.DeploymentService.Update(ctx_, deployment, services.UpdateDeploymentOption{
		Description: updateSchema.Description,
	})

	if err != nil {
		log.Error().Msgf("Could not update deployment with the name %s: %s", getSchema.DeploymentName, err.Error())
		ctx.JSON(400, fmt.Sprintf("Could not update deployment with the name %s", getSchema.DeploymentName))
		return
	}

	if updateSchema.DoNotDeploy {
		deployment, err = c.updateDeploymentInformation(ctx_, updateSchema, deployment)
		if err != nil {
			log.Error().Msgf("Could not update deployment information %s: %s", getSchema.DeploymentName, err.Error())
			ctx.JSON(400, err.Error())
			return
		}
	} else {
		deployment, err = c.updateDeploymentEntities(ctx_, updateSchema, deployment)
		if err != nil {
			log.Error().Msgf("Could not update deployment entities %s: %s", getSchema.DeploymentName, err.Error())
			ctx.JSON(400, err.Error())
			return
		}
	}

	ctx.JSON(200, deployment)
}

func (c *deploymentController) updateDeploymentEntities(ctx context.Context, schema schemas.UpdateDeploymentSchema, deployment *models.Deployment) (*models.Deployment, error) {
	compoundAINims := map[string]uint{}
	for _, target := range schema.Targets {
		compoundAINims[fmt.Sprintf("%s:%s", target.BentoRepository, target.Bento)] = 123 // TODO: call API to fetch the real bento IDS
	}

	// Mark previous revisions as inactive...
	status_ := schemas.DeploymentRevisionStatusActive
	oldDeploymentRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		DeploymentId: &deployment.ID,
		Status:       &status_,
	})
	if err != nil {
		return nil, err
	}

	var oldDeploymentTargets = make([]*models.DeploymentTarget, 0)

	for _, oldDeploymentRevision := range oldDeploymentRevisions {
		_, err = services.DeploymentRevisionService.Update(ctx, oldDeploymentRevision, services.UpdateDeploymentRevisionOption{
			Status: schemas.DeploymentRevisionStatusPtr(schemas.DeploymentRevisionStatusInactive),
		})

		if err != nil {
			return nil, err
		}

		_oldDeploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{
			DeploymentRevisionId: &oldDeploymentRevision.ID,
		})

		oldDeploymentTargets = append(oldDeploymentTargets, _oldDeploymentTargets...)

		if err != nil {
			return nil, err
		}
	}

	// Create a new revision
	deploymentRevision, err := services.DeploymentRevisionService.Create(ctx, services.CreateDeploymentRevisionOption{
		DeploymentId: deployment.ID,
		Status:       schemas.DeploymentRevisionStatusActive,
	})
	if err != nil {
		return nil, err
	}

	// Create deployment targets
	deploymentTargets := make([]*models.DeploymentTarget, 0, len(schema.Targets))
	for _, createDeploymentTargetSchema := range schema.Targets {
		compoundAINimTag := fmt.Sprintf("%s:%s", createDeploymentTargetSchema.BentoRepository, createDeploymentTargetSchema.Bento)
		deploymentTarget, err := services.DeploymentTargetService.Create(ctx, services.CreateDeploymentTargetOption{
			DeploymentId:         deployment.ID,
			DeploymentRevisionId: deploymentRevision.ID,
			CompoundAINimId:      compoundAINims[compoundAINimTag],
			CompoundAINimTag:     compoundAINimTag,
			Config:               createDeploymentTargetSchema.Config,
		})
		if err != nil {
			return nil, err
		}
		deploymentTargets = append(deploymentTargets, deploymentTarget)
	}

	for _, oldDeploymentTarget := range oldDeploymentTargets {
		_, err := services.DeploymentTargetService.Terminate(ctx, oldDeploymentTarget)

		if err != nil {
			return nil, err
		}
	}

	// Deploy new revision
	err = services.DeploymentRevisionService.Deploy(ctx, deploymentRevision, deploymentTargets, false)
	if err != nil {
		return nil, err
	}

	return deployment, nil
}

func (c *deploymentController) updateDeploymentInformation(ctx context.Context, schema schemas.UpdateDeploymentSchema, deployment *models.Deployment) (*models.Deployment, error) {
	status_ := schemas.DeploymentRevisionStatusActive
	activeReploymentRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		DeploymentId: &deployment.ID,
		Status:       &status_,
	})
	if err != nil {
		return nil, err
	}

	targetSchemaCompoundAINims := map[string]*schemas.CreateDeploymentTargetSchema{}
	for _, targetSchema := range schema.Targets {
		targetSchemaCompoundAINims[fmt.Sprintf("%s:%s", targetSchema.BentoRepository, targetSchema.Bento)] = targetSchema
	}

	var activeDeploymentTargets = make([]*models.DeploymentTarget, 0)

	for _, activeReploymentRevision := range activeReploymentRevisions {
		_activeDeploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{
			DeploymentRevisionId: &activeReploymentRevision.ID,
		})

		activeDeploymentTargets = append(activeDeploymentTargets, _activeDeploymentTargets...)

		if err != nil {
			return nil, err
		}
	}

	for _, activeDeploymentTarget := range activeDeploymentTargets {
		if createDeploymentTargetSchema, ok := targetSchemaCompoundAINims[activeDeploymentTarget.CompoundAINimTag]; ok {
			config := activeDeploymentTarget.Config
			config.KubeResourceUid = createDeploymentTargetSchema.Config.KubeResourceUid
			config.KubeResourceVersion = createDeploymentTargetSchema.Config.KubeResourceVersion

			_, err = services.DeploymentTargetService.Update(ctx, activeDeploymentTarget, services.UpdateDeploymentTargetOption{
				Config: &config,
			})
			if err != nil {
				return nil, err
			}
		}
	}

	return deployment, nil
}

func (c *deploymentController) Get(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		ctx.JSON(400, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	ctx.JSON(200, deployment)
}

func (c *deploymentController) Terminate(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(400, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	_, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err = services.DeploymentService.Terminate(ctx_, deployment)
	if err != nil {
		log.Error().Msgf("Could not terminate deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(400, fmt.Sprintf("Could not terminate deployment with the name %s", schema.DeploymentName))
		return
	}

	ctx.JSON(200, deployment)
}

func (c *deploymentController) Delete(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(400, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	deployment, err = services.DeploymentService.Delete(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Could not delete deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(400, gin.H{"error": fmt.Sprintf("Could not delete deployment with the name %s", schema.DeploymentName)})
		return
	}

	ctx.JSON(200, deployment)
}

func (c *deploymentController) ListClusterDeployments(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	var getCluster schemas.GetClusterSchema

	if err := ctx.ShouldBindUri(&getCluster); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := services.ClusterService.GetByName(ctx, getCluster.ClusterName)

	if err != nil {
		log.Error().Msgf("Could not find cluster with the name %s: %s", getCluster.ClusterName, err.Error())
		ctx.JSON(400, fmt.Sprintf("Could not find cluster with the name %s", getCluster.ClusterName))
		return
	}

	listOpt := services.ListDeploymentOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		ClusterId: &cluster.ID,
	}

	deployments, _, err := services.DeploymentService.List(ctx, listOpt)
	if err != nil {
		log.Error().Msgf("Could not find deployments for the cluster %s with the following opts %+v: %s", getCluster.ClusterName, listOpt, err.Error())
		ctx.JSON(400, "Could not find deployments")
		return
	}

	ctx.JSON(200, deployments)
}

func getDeployment(ctx *gin.Context, s *schemas.GetDeploymentSchema) (*models.Deployment, error) {
	cluster, err := services.ClusterService.GetByName(ctx, s.ClusterName)

	if err != nil {
		return nil, err
	}

	deployment, err := services.DeploymentService.GetByName(ctx, cluster.ID, s.KubeNamespace, s.DeploymentName)

	if err != nil {
		return nil, err
	}

	return deployment, nil
}

// The start of the V2 deployment APIs
func (c *deploymentController) CreateV2(ctx *gin.Context) {
	clusterName := ctx.Query("cluster")
	if clusterName == "" {
		clusterName = "default"
	}
	log.Info().Msgf("Got clusterName: %s", clusterName)

	cluster, err := services.ClusterService.GetByName(ctx, clusterName)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	var schema schemasv2.CreateDeploymentSchema
	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Parse the Bento attribute
	var bentoRepo, bentoVersion string
	bentoParts := strings.Split(schema.Bento, ":")
	if len(bentoParts) == 2 {
		bentoRepo = bentoParts[0]
		bentoVersion = bentoParts[1]
	} else {
		ctx.JSON(400, fmt.Errorf("invalid Bento format, expected 'bentorepo:bentoversion'"))
		return
	}
	fmt.Println("Bento Repository:", bentoRepo)
	fmt.Println("Bento Version:", bentoVersion)

	// Determine the deployment name
	deploymentName := schema.Name
	if deploymentName == "" {
		deploymentName = fmt.Sprintf("dep-%s-%s--%s", bentoRepo, bentoVersion, uuid.New().String())
		deploymentName = deploymentName[:63] // Max label length for k8s
	}
	fmt.Println("Deployment Name:", deploymentName)

	// Extract the first service from Services map
	var firstServiceSpec schemasv2.ServiceSpec
	for _, serviceSpec := range schema.Services {
		firstServiceSpec = serviceSpec
		break
	}

	hpaMinReplica := int32(firstServiceSpec.Scaling.MinReplicas)
	hpaMaxRepica := int32(firstServiceSpec.Scaling.MaxReplicas)
	enableIngress := false

	// Convert service configuration into CreateDeploymentTargetSchema
	createDeploymentTarget := &schemas.CreateDeploymentTargetSchema{
		BentoRepository: bentoRepo,
		Bento:           bentoVersion,
		Config: &schemas.DeploymentTargetConfig{
			HPAConf: &schemas.DeploymentTargetHPAConf{
				MinReplicas: &hpaMinReplica,
				MaxReplicas: &hpaMaxRepica,
			},
			Resources: &schemas.Resources{
				Requests: &schemas.ResourceItem{
					CPU:    firstServiceSpec.ConfigOverrides.Resources.Requests.CPU,
					Memory: firstServiceSpec.ConfigOverrides.Resources.Requests.Memory,
				},
				Limits: &schemas.ResourceItem{
					CPU:    firstServiceSpec.ConfigOverrides.Resources.Limits.CPU,
					Memory: firstServiceSpec.ConfigOverrides.Resources.Limits.Memory,
				},
			},
			// Assuming Envs, Runners, EnableIngress, DeploymentStrategy are default values or nil
			EnableIngress:      &enableIngress, // Assuming false as default
			DeploymentStrategy: nil,            // Assuming no specific strategy as default
		},
	}

	body, _ := json.Marshal(createDeploymentTarget)
	log.Info().Msgf("Got the following target: %s", body)

	// Getting the k8s namespace
	kubeNamespace := os.Getenv("DEFAULT_KUBE_NAMESPACE")
	if kubeNamespace == "" {
		kubeNamespace = "yatai"
	}

	// Create the CreateDeploymentSchema instance
	createDeploymentSchema := CreateDeploymentSchema{
		CreateDeploymentSchema: schemas.CreateDeploymentSchema{
			Name:          deploymentName,
			KubeNamespace: kubeNamespace,
			UpdateDeploymentSchema: schemas.UpdateDeploymentSchema{
				Targets: []*schemas.CreateDeploymentTargetSchema{
					createDeploymentTarget,
				},
			},
		},
	}

	deployment, err := c.createDeploymentHelper(ctx, cluster, createDeploymentSchema)

	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Getting mocked creator
	creator := MockedController.getDefaultUserHelper()

	deploymentSchema := converters.ToDeploymentSchemaV2(cluster, deployment, creator)
	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) GetV2(ctx *gin.Context) {
	var schema schemasv2.GetDeploymentSchema
	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	clusterName := ctx.Query("cluster")
	if clusterName == "" {
		clusterName = "default"
	}
	log.Info().Msgf("Got clusterName: %s", clusterName)

	// Getting the k8s namespace
	kubeNamespace := os.Getenv("DEFAULT_KUBE_NAMESPACE")
	if kubeNamespace == "" {
		kubeNamespace = "compoundai"
	}

	cluster, err := services.ClusterService.GetByName(ctx, clusterName)

	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := services.DeploymentService.GetByName(ctx, cluster.ID, kubeNamespace, schema.DeploymentName)

	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Getting mocked creator
	creator := MockedController.getDefaultUserHelper()

	deploymentSchema := converters.ToDeploymentSchemaV2(cluster, deployment, creator)
	ctx.JSON(200, deploymentSchema)
}
