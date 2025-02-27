package services

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/common/consts"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/database"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"gorm.io/gorm"
	"k8s.io/apimachinery/pkg/util/validation"
)

type deploymentService struct{}

var DeploymentService = deploymentService{}

type CreateDeploymentOption struct {
	CreatorId     uint
	ClusterId     uint
	Name          string
	Description   string
	KubeNamespace string
}

type UpdateDeploymentOption struct {
	Description *string
	Status      *schemas.DeploymentStatus
}

type UpdateDeploymentStatusOption struct {
	Status    *schemas.DeploymentStatus
	SyncingAt **time.Time
	UpdatedAt **time.Time
}

type ListDeploymentOption struct {
	BaseListOption
	ClusterId        *uint
	CreatorId        *uint
	LastUpdaterId    *uint
	OrganizationId   *uint
	ClusterIds       *[]uint
	CreatorIds       *[]uint
	LastUpdaterIds   *[]uint
	OrganizationIds  *[]uint
	Ids              *[]uint
	CompoundAINimIds *[]uint
	Statuses         *[]schemas.DeploymentStatus
	Order            *string
}

func (s *deploymentService) Create(ctx context.Context, opt CreateDeploymentOption) (*models.Deployment, error) {
	errs := validation.IsDNS1035Label(opt.Name)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	errs = validation.IsDNS1035Label(opt.KubeNamespace)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	guid := uuid.New()

	deployment := models.Deployment{
		Resource: models.Resource{
			Name: opt.Name,
		},
		ClusterAssociate: models.ClusterAssociate{
			ClusterId: opt.ClusterId,
		},
		Description:     opt.Description,
		Status:          schemas.DeploymentStatusNonDeployed,
		KubeDeployToken: guid.String(),
		KubeNamespace:   opt.KubeNamespace,
	}

	db := s.getDB(ctx)

	err := db.Create(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to create deployment %s", err.Error())
		return nil, err
	}

	return &deployment, err
}

func (s *deploymentService) Update(ctx context.Context, b *models.Deployment, opt UpdateDeploymentOption) (*models.Deployment, error) {
	var err error
	updaters := make(map[string]interface{})
	if opt.Description != nil {
		updaters["description"] = *opt.Description
		defer func() {
			if err == nil {
				b.Description = *opt.Description
			}
		}()
	}

	if opt.Status != nil {
		updaters["status"] = *opt.Status
		defer func() {
			if err == nil {
				b.Status = *opt.Status
			}
		}()
	}

	if len(updaters) == 0 {
		return b, nil
	}

	log.Info().Msgf("Updating deployment with updaters %+v", updaters)
	err = s.getDB(ctx).Where("id = ?", b.ID).Updates(updaters).Error
	if err != nil {
		return nil, err
	}

	return b, err
}

func (s *deploymentService) Get(ctx context.Context, id uint) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("id = ?", id).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by id %d: %s", id, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) GetByUid(ctx context.Context, uid string) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("uid = ?", uid).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by uid %s: %s", uid, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) GetByName(ctx context.Context, clusterId uint, kubeNamespace, name string) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("cluster_id = ?", clusterId).Where("kube_namespace = ?", kubeNamespace).Where("name = ?", name).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by name %s: %s", name, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) Delete(ctx context.Context, deployment *models.Deployment) (*models.Deployment, error) {
	if deployment.Status != schemas.DeploymentStatusTerminated && deployment.Status != schemas.DeploymentStatusTerminating {
		return nil, errors.New("deployment is not terminated")
	}
	return deployment, s.getDB(ctx).Unscoped().Delete(deployment).Error
}

func (s *deploymentService) Terminate(ctx context.Context, deployment *models.Deployment) (*models.Deployment, error) {
	deployment, err := s.UpdateStatus(ctx, deployment, UpdateDeploymentStatusOption{
		Status: schemas.DeploymentStatusTerminating.Ptr(),
	})
	if err != nil {
		return nil, err
	}

	start := uint(0)
	count := uint(1)

	deploymentRevisions, _, err := DeploymentRevisionService.List(ctx, ListDeploymentRevisionOption{
		BaseListOption: BaseListOption{
			Start: &start,
			Count: &count,
		},
		DeploymentId: &deployment.ID,
		Status:       schemas.DeploymentRevisionStatusActive.Ptr(),
	})
	if err != nil {
		return nil, err
	}

	log.Info().Msgf("Fetched %d active deployment revisions to terminate", len(deploymentRevisions))
	for _, deploymentRevision := range deploymentRevisions {
		err = DeploymentRevisionService.Terminate(ctx, deploymentRevision)
		if err != nil {
			return nil, err
		}
	}
	// _, err = s.SyncStatus(ctx, deployment) // TODO: once implemented uncomment this
	return deployment, err
}

func (s *deploymentService) UpdateStatus(ctx context.Context, deployment *models.Deployment, opt UpdateDeploymentStatusOption) (*models.Deployment, error) {
	updater := map[string]interface{}{}
	if opt.Status != nil {
		deployment.Status = *opt.Status
		updater["status"] = *opt.Status
	}
	if opt.SyncingAt != nil {
		deployment.StatusSyncingAt = *opt.SyncingAt
		updater["status_syncing_at"] = *opt.SyncingAt
	}
	if opt.UpdatedAt != nil {
		deployment.StatusUpdatedAt = *opt.UpdatedAt
		updater["status_updated_at"] = *opt.UpdatedAt
	}
	log.Info().Msgf("Updating deployment with updaters %+v", updater)
	err := s.getDB(ctx).Where("id = ?", deployment.ID).Updates(updater).Error
	return deployment, err
}

func (s *deploymentService) SyncStatus(ctx context.Context, d *models.Deployment) (schemas.DeploymentStatus, error) {
	now := time.Now()
	nowPtr := &now
	_, err := s.UpdateStatus(ctx, d, UpdateDeploymentStatusOption{
		SyncingAt: &nowPtr,
	})
	if err != nil {
		log.Error().Msgf("Failed to update sync time for deployment %s: %s", d.Name, err.Error())
		return d.Status, err
	}
	currentStatus := schemas.DeploymentStatusDeploying

	// TODO: get status from DMS

	now = time.Now()
	nowPtr = &now
	_, err = s.UpdateStatus(ctx, d, UpdateDeploymentStatusOption{
		Status:    &currentStatus,
		UpdatedAt: &nowPtr,
	})
	if err != nil {
		return currentStatus, err
	}
	return currentStatus, nil
}

func (s *deploymentService) List(ctx context.Context, opt ListDeploymentOption) ([]*models.Deployment, uint, error) {
	query := s.getDB(ctx)

	if opt.Ids != nil {
		query = query.Where("deployment.id in (?)", *opt.Ids)
	}

	query = query.Joins("LEFT JOIN deployment_revision ON deployment_revision.deployment_id = deployment.id AND deployment_revision.status = ?", schemas.DeploymentRevisionStatusActive)
	if opt.CompoundAINimIds != nil {
		query = query.Joins("LEFT JOIN deployment_target ON deployment_target.deployment_revision_id = deployment_revision.id").Where("deployment_target.compoundai_nim_id IN (?)", *opt.CompoundAINimIds)
	}
	if opt.ClusterId != nil {
		query = query.Where("deployment.cluster_id = ?", *opt.ClusterId)
	}
	if opt.ClusterIds != nil {
		query = query.Where("deployment.cluster_id IN (?)", *opt.ClusterIds)
	}
	if opt.Statuses != nil {
		query = query.Where("deployment.status IN (?)", *opt.Statuses)
	}
	query = opt.BindQueryWithKeywords(query, "deployment")
	query = query.Select("deployment_revision.*, deployment.*")
	var total int64
	err := query.Count(&total).Error

	if err != nil {
		return nil, 0, err
	}
	query = opt.BindQueryWithLimit(query)
	if opt.Order != nil {
		query = query.Order(*opt.Order)
	} else {
		query.Order("deployment.id DESC")
	}
	deployments := make([]*models.Deployment, 0)
	err = query.Find(&deployments).Error
	if err != nil {
		return nil, 0, err
	}
	return deployments, uint(total), err
}

func (s *deploymentService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.Deployment{})
	return db
}
