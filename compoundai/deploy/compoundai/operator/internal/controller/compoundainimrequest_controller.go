/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"text/template"
	"time"

	"emperror.dev/errors"
	"github.com/apparentlymart/go-shquot/shquot"
	"github.com/ettle/strcase"
	"github.com/huandu/xstrings"
	"github.com/mitchellh/hashstructure/v2"
	"github.com/prometheus/common/version"
	"github.com/prune998/docker-registry-client/registry"
	"github.com/rs/xid"
	"github.com/sergeymakinen/go-quote/unix"
	"github.com/sirupsen/logrus"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/internal/controller_common"
	commonconfig "github.com/triton-inference-server/triton_distributed/deploy/compoundai/pkg/compoundai/config"
	commonconsts "github.com/triton-inference-server/triton_distributed/deploy/compoundai/pkg/compoundai/consts"
	"gopkg.in/yaml.v2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	compoundaiCommon "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/common"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/modelschemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/schemasv1"
	yataiclient "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/yatai-client"
	nvidiacomv1alpha1 "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/v1alpha1"
)

const (
	KubeAnnotationCompoundAINimRequestHash            = "yatai.ai/bento-request-hash"
	KubeAnnotationCompoundAINimRequestImageBuiderHash = "yatai.ai/bento-request-image-builder-hash"
	KubeAnnotationCompoundAINimRequestModelSeederHash = "yatai.ai/bento-request-model-seeder-hash"
	KubeLabelYataiImageBuilderSeparateModels          = "yatai.ai/yatai-image-builder-separate-models"
	KubeAnnotationCompoundAINimStorageNS              = "yatai.ai/bento-storage-namepsace"
	KubeAnnotationModelStorageNS                      = "yatai.ai/model-storage-namepsace"
	StoreSchemaAWS                                    = "aws"
	StoreSchemaGCP                                    = "gcp"
)

// CompoundAINimRequestReconciler reconciles a CompoundAINimRequest object
type CompoundAINimRequestReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   controller_common.Config
}

// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimrequests/finalizers,verbs=update
//+kubebuilder:rbac:groups=nvidia.com,resources=compoundainims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=nvidia.com,resources=compoundainims/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch
//+kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the CompoundAINimRequest object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *CompoundAINimRequestReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	compoundAINimRequest := &nvidiacomv1alpha1.CompoundAINimRequest{}

	err = r.Get(ctx, req.NamespacedName, compoundAINimRequest)

	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("compoundAINimRequest resource not found. Ignoring since object must be deleted")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get compoundAINimRequest")
		return
	}

	for _, condition := range compoundAINimRequest.Status.Conditions {
		if condition.Type == nvidiacomv1alpha1.CompoundAIDeploymentConditionTypeAvailable && condition.Status == metav1.ConditionTrue {
			logs.Info("Skip available compoundAINimRequest")
			return
		}
	}

	if len(compoundAINimRequest.Status.Conditions) == 0 {
		compoundAINimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsSeeding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile compoundAINimRequest",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile compoundAINimRequest",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExists,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile compoundAINimRequest",
			},
		)
		if err != nil {
			return
		}
	}

	logs = logs.WithValues("compoundAINimRequest", compoundAINimRequest.Name, "compoundAINimRequestNamespace", compoundAINimRequest.Namespace)

	defer func() {
		if err == nil {
			logs.Info("Reconcile success")
			return
		}
		logs.Error(err, "Failed to reconcile compoundAINimRequest.")
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile compoundAINimRequest: %v", err)
		_, err_ := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: err.Error(),
			},
		)
		if err_ != nil {
			logs.Error(err_, "Failed to update compoundAINimRequest status")
			return
		}
	}()

	compoundAINimAvailableCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable)
	if compoundAINimAvailableCondition == nil || compoundAINimAvailableCondition.Status != metav1.ConditionUnknown {
		compoundAINimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Reconciling",
			},
		)
		if err != nil {
			return
		}
	}

	separateModels := isSeparateModels(compoundAINimRequest)

	modelsExists := false
	var modelsExistsResult ctrl.Result
	var modelsExistsErr error

	if separateModels {
		compoundAINimRequest, modelsExists, modelsExistsResult, modelsExistsErr = r.ensureModelsExists(ctx, ensureModelsExistsOption{
			compoundAINimRequest: compoundAINimRequest,
			req:                  req,
		})
	}

	compoundAINimRequest, imageInfo, imageExists, imageExistsResult, err := r.ensureImageExists(ctx, ensureImageExistsOption{
		compoundAINimRequest: compoundAINimRequest,
		req:                  req,
	})

	if err != nil {
		err = errors.Wrapf(err, "ensure image exists")
		return
	}

	if !imageExists {
		result = imageExistsResult
		compoundAINimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "CompoundAINim image is building",
			},
		)
		if err != nil {
			return
		}
		return
	}

	if modelsExistsErr != nil {
		err = errors.Wrap(modelsExistsErr, "ensure model exists")
		return
	}

	if separateModels && !modelsExists {
		result = modelsExistsResult
		compoundAINimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Model is seeding",
			},
		)
		if err != nil {
			return
		}
		return
	}

	compoundAINimCR := &nvidiacomv1alpha1.CompoundAINim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      compoundAINimRequest.Name,
			Namespace: compoundAINimRequest.Namespace,
		},
		Spec: nvidiacomv1alpha1.CompoundAINimSpec{
			Tag:         compoundAINimRequest.Spec.BentoTag,
			Image:       imageInfo.ImageName,
			ServiceName: compoundAINimRequest.Spec.ServiceName,
			Context:     compoundAINimRequest.Spec.Context,
			Models:      compoundAINimRequest.Spec.Models,
		},
	}

	if separateModels {
		compoundAINimCR.Annotations = map[string]string{
			commonconsts.KubeAnnotationYataiImageBuilderSeparateModels: commonconsts.KubeLabelValueTrue,
		}
		if isAddNamespacePrefix() { // deprecated
			compoundAINimCR.Annotations[commonconsts.KubeAnnotationIsMultiTenancy] = commonconsts.KubeLabelValueTrue
		}
		compoundAINimCR.Annotations[KubeAnnotationModelStorageNS] = compoundAINimRequest.Annotations[KubeAnnotationModelStorageNS]
	}

	err = ctrl.SetControllerReference(compoundAINimRequest, compoundAINimCR, r.Scheme)
	if err != nil {
		err = errors.Wrap(err, "set controller reference")
		return
	}

	if imageInfo.DockerConfigJSONSecretName != "" {
		compoundAINimCR.Spec.ImagePullSecrets = []corev1.LocalObjectReference{
			{
				Name: imageInfo.DockerConfigJSONSecretName,
			},
		}
	}

	if compoundAINimRequest.Spec.DownloadURL == "" {
		var compoundAINim *schemasv1.BentoFullSchema
		compoundAINim, err = r.getCompoundAINim(ctx, compoundAINimRequest)
		if err != nil {
			err = errors.Wrap(err, "get compoundAINim")
			return
		}
		compoundAINimCR.Spec.Context = &nvidiacomv1alpha1.BentoContext{
			BentomlVersion: compoundAINim.Manifest.BentomlVersion,
		}
	}

	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CompoundAINimImageBuilder", "Creating CompoundAINim CR %s in namespace %s", compoundAINimCR.Name, compoundAINimCR.Namespace)
	err = r.Create(ctx, compoundAINimCR)
	isAlreadyExists := k8serrors.IsAlreadyExists(err)
	if err != nil && !isAlreadyExists {
		err = errors.Wrap(err, "create CompoundAINim resource")
		return
	}
	if isAlreadyExists {
		oldCompoundAINimCR := &nvidiacomv1alpha1.CompoundAINim{}
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CompoundAINimImageBuilder", "Updating CompoundAINim CR %s in namespace %s", compoundAINimCR.Name, compoundAINimCR.Namespace)
		err = r.Get(ctx, types.NamespacedName{Name: compoundAINimCR.Name, Namespace: compoundAINimCR.Namespace}, oldCompoundAINimCR)
		if err != nil {
			err = errors.Wrap(err, "get CompoundAINim resource")
			return
		}
		if !reflect.DeepEqual(oldCompoundAINimCR.Spec, compoundAINimCR.Spec) {
			oldCompoundAINimCR.OwnerReferences = compoundAINimCR.OwnerReferences
			oldCompoundAINimCR.Spec = compoundAINimCR.Spec
			err = r.Update(ctx, oldCompoundAINimCR)
			if err != nil {
				err = errors.Wrap(err, "update CompoundAINim resource")
				return
			}
		}
	}

	compoundAINimRequest, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "CompoundAINim is generated",
		},
	)
	if err != nil {
		return
	}

	return
}

func isEstargzEnabled() bool {
	return os.Getenv("ESTARGZ_ENABLED") == commonconsts.KubeLabelValueTrue
}

type ensureImageExistsOption struct {
	compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
	req                  ctrl.Request
}

//nolint:gocyclo,nakedret
func (r *CompoundAINimRequestReconciler) ensureImageExists(ctx context.Context, opt ensureImageExistsOption) (compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, imageInfo ImageInfo, imageExists bool, result ctrl.Result, err error) { // nolint: unparam
	logs := log.FromContext(ctx)

	compoundAINimRequest = opt.compoundAINimRequest
	req := opt.req

	imageInfo, err = r.getImageInfo(ctx, GetImageInfoOption{
		CompoundAINimRequest: compoundAINimRequest,
	})
	if err != nil {
		err = errors.Wrap(err, "get image info")
		return
	}

	imageExistsCheckedCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExistsChecked)
	imageExistsCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExists)
	if imageExistsCheckedCondition == nil || imageExistsCheckedCondition.Status != metav1.ConditionTrue || imageExistsCheckedCondition.Message != imageInfo.ImageName {
		imageExistsCheckedCondition = &metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExistsChecked,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		}
		compoundAINimAvailableCondition := &metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: "Checking image exists",
		}
		compoundAINimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCheckedCondition, *compoundAINimAvailableCondition)
		if err != nil {
			return
		}
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CheckingImage", "Checking image exists: %s", imageInfo.ImageName)
		imageExists, err = checkImageExists(compoundAINimRequest, imageInfo.DockerRegistry, imageInfo.InClusterImageName)
		if err != nil {
			err = errors.Wrapf(err, "check image %s exists", imageInfo.ImageName)
			return
		}

		err = r.Get(ctx, req.NamespacedName, compoundAINimRequest)
		if err != nil {
			logs.Error(err, "Failed to re-fetch compoundAINimRequest")
			return
		}

		if imageExists {
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CheckingImage", "Image exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExistsChecked,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExists,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			compoundAINimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		} else {
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CheckingImage", "Image not exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExistsChecked,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image not exists: %s", imageInfo.ImageName),
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExists,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image %s is not exists", imageInfo.ImageName),
			}
			compoundAINimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		}
	}

	var compoundAINimRequestHashStr string
	compoundAINimRequestHashStr, err = r.getHashStr(compoundAINimRequest)
	if err != nil {
		err = errors.Wrapf(err, "get compoundAINimRequest %s/%s hash string", compoundAINimRequest.Namespace, compoundAINimRequest.Name)
		return
	}

	imageExists = imageExistsCondition != nil && imageExistsCondition.Status == metav1.ConditionTrue && imageExistsCondition.Message == imageInfo.ImageName
	if imageExists {
		return
	}

	jobLabels := map[string]string{
		commonconsts.KubeLabelBentoRequest:        compoundAINimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder: commonconsts.KubeLabelValueTrue,
	}

	if isSeparateModels(opt.compoundAINimRequest) {
		jobLabels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueTrue
	} else {
		jobLabels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueFalse
	}

	jobs := &batchv1.JobList{}
	err = r.List(ctx, jobs, client.InNamespace(req.Namespace), client.MatchingLabels(jobLabels))
	if err != nil {
		err = errors.Wrap(err, "list jobs")
		return
	}

	reservedJobs := make([]*batchv1.Job, 0)
	for _, job_ := range jobs.Items {
		job_ := job_

		oldHash := job_.Annotations[KubeAnnotationCompoundAINimRequestHash]
		if oldHash != compoundAINimRequestHashStr {
			logs.Info("Because hash changed, delete old job", "job", job_.Name, "oldHash", oldHash, "newHash", compoundAINimRequestHashStr)
			// --cascade=foreground
			err = r.Delete(ctx, &job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
			return
		} else {
			reservedJobs = append(reservedJobs, &job_)
		}
	}

	var job *batchv1.Job
	if len(reservedJobs) > 0 {
		job = reservedJobs[0]
	}

	if len(reservedJobs) > 1 {
		for _, job_ := range reservedJobs[1:] {
			logs.Info("Because has more than one job, delete old job", "job", job_.Name)
			// --cascade=foreground
			err = r.Delete(ctx, job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
		}
	}

	if job == nil {
		job, err = r.generateImageBuilderJob(ctx, GenerateImageBuilderJobOption{
			ImageInfo:            imageInfo,
			CompoundAINimRequest: compoundAINimRequest,
		})
		if err != nil {
			err = errors.Wrap(err, "generate image builder job")
			return
		}
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Creating image builder job: %s", job.Name)
		err = r.Create(ctx, job)
		if err != nil {
			err = errors.Wrapf(err, "create image builder job %s", job.Name)
			return
		}
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Created image builder job: %s", job.Name)
		return
	}

	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CheckingImageBuilderJob", "Found image builder job: %s", job.Name)

	err = r.Get(ctx, req.NamespacedName, compoundAINimRequest)
	if err != nil {
		logs.Error(err, "Failed to re-fetch compoundAINimRequest")
		return
	}
	imageBuildingCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding)

	isJobFailed := false
	isJobRunning := true

	if job.Spec.Completions != nil {
		if job.Status.Succeeded != *job.Spec.Completions {
			if job.Status.Failed > 0 {
				for _, condition := range job.Status.Conditions {
					if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
						isJobFailed = true
						break
					}
				}
			}
			isJobRunning = !isJobFailed
		} else {
			isJobRunning = false
		}
	}

	if isJobRunning {
		conditions := make([]metav1.Condition, 0)
		if job.Status.Active > 0 {
			conditions = append(conditions, metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is running", job.Name),
			})
		} else {
			conditions = append(conditions, metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is waiting", job.Name),
			})
		}
		if compoundAINimRequest.Spec.ImageBuildTimeout != nil {
			if imageBuildingCondition != nil && imageBuildingCondition.LastTransitionTime.Add(time.Duration(*compoundAINimRequest.Spec.ImageBuildTimeout)).Before(time.Now()) {
				conditions = append(conditions, metav1.Condition{
					Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
					Status:  metav1.ConditionFalse,
					Reason:  "Timeout",
					Message: fmt.Sprintf("Image building job %s is timeout", job.Name),
				})
				if _, err = r.setStatusConditions(ctx, req, conditions...); err != nil {
					return
				}
				err = errors.New("image build timeout")
				return
			}
		}

		if compoundAINimRequest, err = r.setStatusConditions(ctx, req, conditions...); err != nil {
			return
		}

		if imageBuildingCondition != nil && imageBuildingCondition.Status != metav1.ConditionTrue && isJobRunning {
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CompoundAINimImageBuilder", "Image is building now")
		}

		return
	}

	if isJobFailed {
		compoundAINimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is failed.", job.Name),
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is failed.", job.Name),
			},
		)
		if err != nil {
			return
		}
		return
	}

	compoundAINimRequest, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageBuilding,
			Status:  metav1.ConditionFalse,
			Reason:  "Reconciling",
			Message: fmt.Sprintf("Image building job %s is successed.", job.Name),
		},
		metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeImageExists,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		},
	)
	if err != nil {
		return
	}

	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CompoundAINimImageBuilder", "Image has been built successfully")

	imageExists = true

	return
}

type ensureModelsExistsOption struct {
	compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
	req                  ctrl.Request
}

//nolint:gocyclo,nakedret
func (r *CompoundAINimRequestReconciler) ensureModelsExists(ctx context.Context, opt ensureModelsExistsOption) (compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, modelsExists bool, result ctrl.Result, err error) { // nolint: unparam
	compoundAINimRequest = opt.compoundAINimRequest
	modelTags := make([]string, 0)
	for _, model := range compoundAINimRequest.Spec.Models {
		modelTags = append(modelTags, model.Tag)
	}

	modelsExistsCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsExists)
	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "SeparateModels", "Separate models are enabled")
	if modelsExistsCondition == nil || modelsExistsCondition.Status == metav1.ConditionUnknown {
		r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "ModelsExists", "Models are not ready")
		modelsExistsCondition = &metav1.Condition{
			Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsExists,
			Status:  metav1.ConditionFalse,
			Reason:  "Reconciling",
			Message: "Models are not ready",
		}
		compoundAINimRequest, err = r.setStatusConditions(ctx, opt.req, *modelsExistsCondition)
		if err != nil {
			return
		}
	}

	modelsExists = modelsExistsCondition != nil && modelsExistsCondition.Status == metav1.ConditionTrue && modelsExistsCondition.Message == fmt.Sprintf("%s:%s", getJuiceFSStorageClassName(), strings.Join(modelTags, ", "))
	if modelsExists {
		return
	}

	modelsMap := make(map[string]*nvidiacomv1alpha1.BentoModel)
	for _, model := range compoundAINimRequest.Spec.Models {
		model := model
		modelsMap[model.Tag] = &model
	}

	jobLabels := map[string]string{
		commonconsts.KubeLabelBentoRequest:  compoundAINimRequest.Name,
		commonconsts.KubeLabelIsModelSeeder: "true",
	}

	jobs := &batchv1.JobList{}
	err = r.List(ctx, jobs, client.InNamespace(compoundAINimRequest.Namespace), client.MatchingLabels(jobLabels))
	if err != nil {
		err = errors.Wrap(err, "list jobs")
		return
	}

	var compoundAINimRequestHashStr string
	compoundAINimRequestHashStr, err = r.getHashStr(compoundAINimRequest)
	if err != nil {
		err = errors.Wrapf(err, "get compoundAINimRequest %s/%s hash string", compoundAINimRequest.Namespace, compoundAINimRequest.Name)
		return
	}

	existingJobModelTags := make(map[string]struct{})
	for _, job_ := range jobs.Items {
		job_ := job_

		oldHash := job_.Annotations[KubeAnnotationCompoundAINimRequestHash]
		if oldHash != compoundAINimRequestHashStr {
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "DeleteJob", "Because hash changed, delete old job %s, oldHash: %s, newHash: %s", job_.Name, oldHash, compoundAINimRequestHashStr)
			// --cascade=foreground
			err = r.Delete(ctx, &job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
			continue
		}

		modelTag := fmt.Sprintf("%s:%s", job_.Labels[commonconsts.KubeLabelYataiModelRepository], job_.Labels[commonconsts.KubeLabelYataiModel])
		_, ok := modelsMap[modelTag]

		if !ok {
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "DeleteJob", "Due to the nonexistence of the model %s, job %s has been deleted.", modelTag, job_.Name)
			// --cascade=foreground
			err = r.Delete(ctx, &job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
		} else {
			existingJobModelTags[modelTag] = struct{}{}
		}
	}

	for _, model := range compoundAINimRequest.Spec.Models {
		if _, ok := existingJobModelTags[model.Tag]; ok {
			continue
		}
		model := model
		pvc := &corev1.PersistentVolumeClaim{}
		pvcName := r.getModelPVCName(compoundAINimRequest, &model)
		err = r.Get(ctx, client.ObjectKey{
			Namespace: compoundAINimRequest.Namespace,
			Name:      pvcName,
		}, pvc)
		isPVCNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isPVCNotFound {
			err = errors.Wrapf(err, "get PVC %s/%s", compoundAINimRequest.Namespace, pvcName)
			return
		}
		if isPVCNotFound {
			pvc = r.generateModelPVC(GenerateModelPVCOption{
				CompoundAINimRequest: compoundAINimRequest,
				Model:                &model,
			})
			err = r.Create(ctx, pvc)
			isPVCAlreadyExists := k8serrors.IsAlreadyExists(err)
			if err != nil && !isPVCAlreadyExists {
				err = errors.Wrapf(err, "create model %s/%s pvc", compoundAINimRequest.Namespace, model.Tag)
				return
			}
		}
		var job *batchv1.Job
		job, err = r.generateModelSeederJob(ctx, GenerateModelSeederJobOption{
			CompoundAINimRequest: compoundAINimRequest,
			Model:                &model,
		})
		if err != nil {
			err = errors.Wrap(err, "generate model seeder job")
			return
		}
		oldJob := &batchv1.Job{}
		err = r.Get(ctx, client.ObjectKeyFromObject(job), oldJob)
		oldJobIsNotFound := k8serrors.IsNotFound(err)
		if err != nil && !oldJobIsNotFound {
			err = errors.Wrap(err, "get job")
			return
		}
		if oldJobIsNotFound {
			err = r.Create(ctx, job)
			if err != nil {
				err = errors.Wrap(err, "create job")
				return
			}
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "CreateJob", "Job %s has been created.", job.Name)
		} else if !reflect.DeepEqual(job.Labels, oldJob.Labels) || !reflect.DeepEqual(job.Annotations, oldJob.Annotations) {
			job.Labels = oldJob.Labels
			job.Annotations = oldJob.Annotations
			err = r.Update(ctx, job)
			if err != nil {
				err = errors.Wrap(err, "update job")
				return
			}
			r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "UpdateJob", "Job %s has been updated.", job.Name)
		}
	}

	jobs = &batchv1.JobList{}
	err = r.List(ctx, jobs, client.InNamespace(compoundAINimRequest.Namespace), client.MatchingLabels(jobLabels))
	if err != nil {
		err = errors.Wrap(err, "list jobs")
		return
	}

	succeedModelTags := make(map[string]struct{})
	failedJobNames := make([]string, 0)
	notReadyJobNames := make([]string, 0)
	for _, job_ := range jobs.Items {
		if job_.Spec.Completions != nil && job_.Status.Succeeded == *job_.Spec.Completions {
			modelTag := fmt.Sprintf("%s:%s", job_.Labels[commonconsts.KubeLabelYataiModelRepository], job_.Labels[commonconsts.KubeLabelYataiModel])
			succeedModelTags[modelTag] = struct{}{}
			continue
		}
		if job_.Status.Failed > 0 {
			for _, condition := range job_.Status.Conditions {
				if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
					failedJobNames = append(failedJobNames, job_.Name)
					continue
				}
			}
		}
		notReadyJobNames = append(notReadyJobNames, job_.Name)
	}

	if len(failedJobNames) > 0 {
		msg := fmt.Sprintf("Model seeder jobs failed: %s", strings.Join(failedJobNames, ", "))
		r.Recorder.Event(compoundAINimRequest, corev1.EventTypeNormal, "ModelsExists", msg)
		compoundAINimRequest, err = r.setStatusConditions(ctx, opt.req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsExists,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: msg,
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeCompoundAINimAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: msg,
			},
		)
		if err != nil {
			return
		}
		err = errors.New(msg)
		return
	}

	modelsExists = true

	for _, model := range compoundAINimRequest.Spec.Models {
		if _, ok := succeedModelTags[model.Tag]; !ok {
			modelsExists = false
			break
		}
	}

	if modelsExists {
		compoundAINimRequest, err = r.setStatusConditions(ctx, opt.req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsExists,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("%s:%s", getJuiceFSStorageClassName(), strings.Join(modelTags, ", ")),
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsSeeding,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: "All models have been seeded.",
			},
		)
		if err != nil {
			return
		}
	} else {
		compoundAINimRequest, err = r.setStatusConditions(ctx, opt.req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.CompoundAINimRequestConditionTypeModelsSeeding,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Model seeder jobs are not ready: %s.", strings.Join(notReadyJobNames, ", ")),
			},
		)
		if err != nil {
			return
		}
	}
	return
}

func (r *CompoundAINimRequestReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, err error) {
	compoundAINimRequest = &nvidiacomv1alpha1.CompoundAINimRequest{}
	/*
		Please don't blame me when you see this kind of code,
		this is to avoid "the object has been modified; please apply your changes to the latest version and try again" when updating CR status,
		don't doubt that almost all CRD operators (e.g. cert-manager) can't avoid this stupid error and can only try to avoid this by this stupid way.
	*/
	for i := 0; i < 3; i++ {
		if err = r.Get(ctx, req.NamespacedName, compoundAINimRequest); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch compoundAINimRequest")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&compoundAINimRequest.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, compoundAINimRequest); err != nil {
			time.Sleep(100 * time.Millisecond)
		} else {
			break
		}
	}
	if err != nil {
		err = errors.Wrap(err, "Failed to update compoundAINimRequest status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, compoundAINimRequest); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch compoundAINimRequest")
		return
	}
	return
}

type CompoundAINimImageBuildEngine string

const (
	CompoundAINimImageBuildEngineKaniko           CompoundAINimImageBuildEngine = "kaniko"
	CompoundAINimImageBuildEngineBuildkit         CompoundAINimImageBuildEngine = "buildkit"
	CompoundAINimImageBuildEngineBuildkitRootless CompoundAINimImageBuildEngine = "buildkit-rootless"
)

const (
	EnvCompoundAINimImageBuildEngine = "BENTO_IMAGE_BUILD_ENGINE"
)

func getCompoundAINimImageBuildEngine() CompoundAINimImageBuildEngine {
	engine := os.Getenv(EnvCompoundAINimImageBuildEngine)
	if engine == "" {
		return CompoundAINimImageBuildEngineKaniko
	}
	return CompoundAINimImageBuildEngine(engine)
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) makeSureDockerConfigJSONSecret(ctx context.Context, namespace string, dockerRegistryConf *commonconfig.DockerRegistryConfig) (dockerConfigJSONSecret *corev1.Secret, err error) {
	if dockerRegistryConf.Username == "" {
		return
	}

	// nolint: gosec
	dockerConfigSecretName := commonconsts.KubeSecretNameRegcred
	dockerConfigObj := struct {
		Auths map[string]struct {
			Auth string `json:"auth"`
		} `json:"auths"`
	}{
		Auths: map[string]struct {
			Auth string `json:"auth"`
		}{
			dockerRegistryConf.Server: {
				Auth: base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", dockerRegistryConf.Username, dockerRegistryConf.Password))),
			},
		},
	}

	dockerConfigContent, err := json.Marshal(dockerConfigObj)
	if err != nil {
		err = errors.Wrap(err, "marshal docker config")
		return nil, err
	}

	dockerConfigJSONSecret = &corev1.Secret{}

	err = r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: dockerConfigSecretName}, dockerConfigJSONSecret)
	dockerConfigIsNotFound := k8serrors.IsNotFound(err)
	// nolint: gocritic
	if err != nil && !dockerConfigIsNotFound {
		err = errors.Wrap(err, "get docker config secret")
		return nil, err
	}
	err = nil
	if dockerConfigIsNotFound {
		dockerConfigJSONSecret = &corev1.Secret{
			Type: corev1.SecretTypeDockerConfigJson,
			ObjectMeta: metav1.ObjectMeta{
				Name:      dockerConfigSecretName,
				Namespace: namespace,
			},
			Data: map[string][]byte{
				".dockerconfigjson": dockerConfigContent,
			},
		}
		err_ := r.Create(ctx, dockerConfigJSONSecret)
		if err_ != nil {
			dockerConfigJSONSecret = &corev1.Secret{}
			err = r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: dockerConfigSecretName}, dockerConfigJSONSecret)
			dockerConfigIsNotFound = k8serrors.IsNotFound(err)
			if err != nil && !dockerConfigIsNotFound {
				err = errors.Wrap(err, "get docker config secret")
				return nil, err
			}
			if dockerConfigIsNotFound {
				err_ = errors.Wrap(err_, "create docker config secret")
				return nil, err_
			}
			if err != nil {
				err = nil
			}
		}
	} else {
		dockerConfigJSONSecret.Data[".dockerconfigjson"] = dockerConfigContent
		err = r.Update(ctx, dockerConfigJSONSecret)
		if err != nil {
			err = errors.Wrap(err, "update docker config secret")
			return nil, err
		}
	}

	return
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) getYataiClient(ctx context.Context) (yataiClient **yataiclient.YataiClient, yataiConf **commonconfig.YataiConfig, err error) {
	yataiConf_, err := commonconfig.GetYataiConfig(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      name,
		}, secret)
		return secret, errors.Wrap(err, "get secret")
	}, commonconsts.YataiImageBuilderComponentName, false)
	isNotFound := k8serrors.IsNotFound(err)
	if err != nil && !isNotFound {
		err = errors.Wrap(err, "get yatai config")
		return
	}

	if isNotFound {
		return
	}

	if yataiConf_.Endpoint == "" {
		return
	}

	if yataiConf_.ClusterName == "" {
		yataiConf_.ClusterName = "default"
	}

	yataiClient_ := yataiclient.NewYataiClient(yataiConf_.Endpoint, fmt.Sprintf("%s:%s:%s", commonconsts.YataiImageBuilderComponentName, yataiConf_.ClusterName, yataiConf_.ApiToken))

	yataiClient = &yataiClient_
	yataiConf = &yataiConf_
	return
}

func (r *CompoundAINimRequestReconciler) getYataiClientWithAuth(ctx context.Context, compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) (**yataiclient.YataiClient, **commonconfig.YataiConfig, error) {
	orgId, ok := compoundAINimRequest.Labels[commonconsts.NgcOrganizationHeaderName]
	if !ok {
		orgId = commonconsts.DefaultOrgId
	}

	userId, ok := compoundAINimRequest.Labels[commonconsts.NgcUserHeaderName]
	if !ok {
		userId = commonconsts.DefaultUserId
	}

	auth := yataiclient.CompoundAIAuthHeaders{
		OrgId:  orgId,
		UserId: userId,
	}

	client, yataiConf, err := r.getYataiClient(ctx)
	if err != nil {
		return nil, nil, err
	}

	(*client).SetAuth(auth)
	return client, yataiConf, err
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) getDockerRegistry(ctx context.Context, compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) (dockerRegistry modelschemas.DockerRegistrySchema, err error) {
	if compoundAINimRequest != nil && compoundAINimRequest.Spec.DockerConfigJSONSecretName != "" {
		secret := &corev1.Secret{}
		err = r.Get(ctx, types.NamespacedName{
			Namespace: compoundAINimRequest.Namespace,
			Name:      compoundAINimRequest.Spec.DockerConfigJSONSecretName,
		}, secret)
		if err != nil {
			err = errors.Wrapf(err, "get docker config json secret %s", compoundAINimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		configJSON, ok := secret.Data[".dockerconfigjson"]
		if !ok {
			err = errors.Errorf("docker config json secret %s does not have .dockerconfigjson key", compoundAINimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		var configObj struct {
			Auths map[string]struct {
				Auth string `json:"auth"`
			} `json:"auths"`
		}
		err = json.Unmarshal(configJSON, &configObj)
		if err != nil {
			err = errors.Wrapf(err, "unmarshal docker config json secret %s", compoundAINimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		imageRegistryURI, _, _ := xstrings.Partition(compoundAINimRequest.Spec.Image, "/")
		var server string
		var auth string
		if imageRegistryURI != "" {
			for k, v := range configObj.Auths {
				if k == imageRegistryURI {
					server = k
					auth = v.Auth
					break
				}
			}
			if server == "" {
				for k, v := range configObj.Auths {
					if strings.Contains(k, imageRegistryURI) {
						server = k
						auth = v.Auth
						break
					}
				}
			}
		}
		if server == "" {
			for k, v := range configObj.Auths {
				server = k
				auth = v.Auth
				break
			}
		}
		if server == "" {
			err = errors.Errorf("no auth in docker config json secret %s", compoundAINimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		dockerRegistry.Server = server
		var credentials []byte
		credentials, err = base64.StdEncoding.DecodeString(auth)
		if err != nil {
			err = errors.Wrapf(err, "cannot base64 decode auth in docker config json secret %s", compoundAINimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		dockerRegistry.Username, _, dockerRegistry.Password = xstrings.Partition(string(credentials), ":")
		if compoundAINimRequest.Spec.OCIRegistryInsecure != nil {
			dockerRegistry.Secure = !*compoundAINimRequest.Spec.OCIRegistryInsecure
		}
		return
	}

	dockerRegistryConfig, err := commonconfig.GetDockerRegistryConfig(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      name,
		}, secret)
		return secret, errors.Wrap(err, "get secret")
	})
	if err != nil {
		err = errors.Wrap(err, "get docker registry")
		return
	}

	compoundAINimRepositoryName := "yatai-bentos"
	modelRepositoryName := "yatai-models"
	if dockerRegistryConfig.BentoRepositoryName != "" {
		compoundAINimRepositoryName = dockerRegistryConfig.BentoRepositoryName
	}
	if dockerRegistryConfig.ModelRepositoryName != "" {
		modelRepositoryName = dockerRegistryConfig.ModelRepositoryName
	}
	compoundAINimRepositoryURI := fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.Server, "/"), compoundAINimRepositoryName)
	modelRepositoryURI := fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.Server, "/"), modelRepositoryName)
	if strings.Contains(dockerRegistryConfig.Server, "docker.io") {
		compoundAINimRepositoryURI = fmt.Sprintf("docker.io/%s", compoundAINimRepositoryName)
		modelRepositoryURI = fmt.Sprintf("docker.io/%s", modelRepositoryName)
	}
	compoundAINimRepositoryInClusterURI := compoundAINimRepositoryURI
	modelRepositoryInClusterURI := modelRepositoryURI
	if dockerRegistryConfig.InClusterServer != "" {
		compoundAINimRepositoryInClusterURI = fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.InClusterServer, "/"), compoundAINimRepositoryName)
		modelRepositoryInClusterURI = fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.InClusterServer, "/"), modelRepositoryName)
		if strings.Contains(dockerRegistryConfig.InClusterServer, "docker.io") {
			compoundAINimRepositoryInClusterURI = fmt.Sprintf("docker.io/%s", compoundAINimRepositoryName)
			modelRepositoryInClusterURI = fmt.Sprintf("docker.io/%s", modelRepositoryName)
		}
	}
	dockerRegistry = modelschemas.DockerRegistrySchema{
		Server:                       dockerRegistryConfig.Server,
		Username:                     dockerRegistryConfig.Username,
		Password:                     dockerRegistryConfig.Password,
		Secure:                       dockerRegistryConfig.Secure,
		BentosRepositoryURI:          compoundAINimRepositoryURI,
		BentosRepositoryURIInCluster: compoundAINimRepositoryInClusterURI,
		ModelsRepositoryURI:          modelRepositoryURI,
		ModelsRepositoryURIInCluster: modelRepositoryInClusterURI,
	}

	return
}

func isAddNamespacePrefix() bool {
	return os.Getenv("ADD_NAMESPACE_PREFIX_TO_IMAGE_NAME") == trueStr
}

func getCompoundAINimImagePrefix(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) string {
	if compoundAINimRequest == nil {
		return ""
	}
	prefix, exist := compoundAINimRequest.Annotations[KubeAnnotationCompoundAINimStorageNS]
	if exist && prefix != "" {
		return fmt.Sprintf("%s.", prefix)
	}
	if isAddNamespacePrefix() {
		return fmt.Sprintf("%s.", compoundAINimRequest.Namespace)
	}
	return ""
}

func getModelNamespace(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) string {
	if compoundAINimRequest == nil {
		return ""
	}
	prefix := compoundAINimRequest.Annotations[KubeAnnotationModelStorageNS]
	if prefix != "" {
		return prefix
	}
	if isAddNamespacePrefix() {
		return compoundAINimRequest.Namespace
	}
	return ""
}

func getCompoundAINimImageName(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, dockerRegistry modelschemas.DockerRegistrySchema, compoundAINimRepositoryName, compoundAINimVersion string, inCluster bool) string {
	if compoundAINimRequest != nil && compoundAINimRequest.Spec.Image != "" {
		return compoundAINimRequest.Spec.Image
	}
	var uri, tag string
	if inCluster {
		uri = dockerRegistry.BentosRepositoryURIInCluster
	} else {
		uri = dockerRegistry.BentosRepositoryURI
	}
	tail := fmt.Sprintf("%s.%s", compoundAINimRepositoryName, compoundAINimVersion)
	separateModels := isSeparateModels(compoundAINimRequest)
	if separateModels {
		tail += ".nomodels"
	}
	if isEstargzEnabled() {
		tail += ".esgz"
	}

	tag = fmt.Sprintf("yatai.%s%s", getCompoundAINimImagePrefix(compoundAINimRequest), tail)

	if len(tag) > 128 {
		hashStr := hash(tail)
		tag = fmt.Sprintf("yatai.%s%s", getCompoundAINimImagePrefix(compoundAINimRequest), hashStr)
		if len(tag) > 128 {
			tag = fmt.Sprintf("yatai.%s", hash(fmt.Sprintf("%s%s", getCompoundAINimImagePrefix(compoundAINimRequest), tail)))[:128]
		}
	}
	return fmt.Sprintf("%s:%s", uri, tag)
}

func isSeparateModels(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) (separateModels bool) {
	return compoundAINimRequest.Annotations[commonconsts.KubeAnnotationYataiImageBuilderSeparateModels] == commonconsts.KubeLabelValueTrue
}

func checkImageExists(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, dockerRegistry modelschemas.DockerRegistrySchema, imageName string) (bool, error) {
	if compoundAINimRequest.Annotations["yatai.ai/force-build-image"] == commonconsts.KubeLabelValueTrue {
		return false, nil
	}

	server, _, imageName := xstrings.Partition(imageName, "/")
	if strings.Contains(server, "docker.io") {
		server = "index.docker.io"
	}
	if dockerRegistry.Secure {
		server = fmt.Sprintf("https://%s", server)
	} else {
		server = fmt.Sprintf("http://%s", server)
	}
	hub, err := registry.New(server, dockerRegistry.Username, dockerRegistry.Password, logrus.Debugf)
	if err != nil {
		err = errors.Wrapf(err, "create docker registry client for %s", server)
		return false, err
	}
	imageName, _, tag := xstrings.LastPartition(imageName, ":")
	tags, err := hub.Tags(imageName)
	isNotFound := err != nil && strings.Contains(err.Error(), "404")
	if isNotFound {
		return false, nil
	}
	if err != nil {
		err = errors.Wrapf(err, "get tags for docker image %s", imageName)
		return false, err
	}
	for _, tag_ := range tags {
		if tag_ == tag {
			return true, nil
		}
	}
	return false, nil
}

type ImageInfo struct {
	DockerRegistry             modelschemas.DockerRegistrySchema
	DockerConfigJSONSecretName string
	ImageName                  string
	InClusterImageName         string
	DockerRegistryInsecure     bool
}

type GetImageInfoOption struct {
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) getImageInfo(ctx context.Context, opt GetImageInfoOption) (imageInfo ImageInfo, err error) {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(opt.CompoundAINimRequest.Spec.BentoTag, ":")
	dockerRegistry, err := r.getDockerRegistry(ctx, opt.CompoundAINimRequest)
	if err != nil {
		err = errors.Wrap(err, "get docker registry")
		return
	}
	imageInfo.DockerRegistry = dockerRegistry
	imageInfo.ImageName = getCompoundAINimImageName(opt.CompoundAINimRequest, dockerRegistry, compoundAINimRepositoryName, compoundAINimVersion, false)
	imageInfo.InClusterImageName = getCompoundAINimImageName(opt.CompoundAINimRequest, dockerRegistry, compoundAINimRepositoryName, compoundAINimVersion, true)

	imageInfo.DockerConfigJSONSecretName = opt.CompoundAINimRequest.Spec.DockerConfigJSONSecretName

	imageInfo.DockerRegistryInsecure = opt.CompoundAINimRequest.Annotations[commonconsts.KubeAnnotationDockerRegistryInsecure] == "true"
	if opt.CompoundAINimRequest.Spec.OCIRegistryInsecure != nil {
		imageInfo.DockerRegistryInsecure = *opt.CompoundAINimRequest.Spec.OCIRegistryInsecure
	}

	if imageInfo.DockerConfigJSONSecretName == "" {
		var dockerRegistryConf *commonconfig.DockerRegistryConfig
		dockerRegistryConf, err = commonconfig.GetDockerRegistryConfig(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
			secret := &corev1.Secret{}
			err := r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, secret)
			return secret, errors.Wrap(err, "get docker registry secret")
		})
		if err != nil {
			err = errors.Wrap(err, "get docker registry")
			return
		}
		imageInfo.DockerRegistryInsecure = !dockerRegistryConf.Secure
		var dockerConfigSecret *corev1.Secret
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Making sure docker config secret %s in namespace %s", commonconsts.KubeSecretNameRegcred, opt.CompoundAINimRequest.Namespace)
		dockerConfigSecret, err = r.makeSureDockerConfigJSONSecret(ctx, opt.CompoundAINimRequest.Namespace, dockerRegistryConf)
		if err != nil {
			err = errors.Wrap(err, "make sure docker config secret")
			return
		}
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Docker config secret %s in namespace %s is ready", commonconsts.KubeSecretNameRegcred, opt.CompoundAINimRequest.Namespace)
		if dockerConfigSecret != nil {
			imageInfo.DockerConfigJSONSecretName = dockerConfigSecret.Name
		}
	}
	return
}

func (r *CompoundAINimRequestReconciler) getCompoundAINim(ctx context.Context, compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) (compoundAINim *schemasv1.BentoFullSchema, err error) {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINimRequest.Spec.BentoTag, ":")

	yataiClient_, _, err := r.getYataiClient(ctx)
	if err != nil {
		err = errors.Wrap(err, "get yatai client")
		return
	}

	if yataiClient_ == nil {
		err = errors.New("can't get yatai client, please check yatai configuration")
		return
	}

	yataiClient := *yataiClient_

	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "FetchCompoundAINim", "Getting compoundAINim %s from yatai service", compoundAINimRequest.Spec.BentoTag)
	compoundAINim, err = yataiClient.GetBento(ctx, compoundAINimRepositoryName, compoundAINimVersion)
	if err != nil {
		err = errors.Wrap(err, "get compoundAINim")
		return
	}
	r.Recorder.Eventf(compoundAINimRequest, corev1.EventTypeNormal, "FetchCompoundAINim", "Got compoundAINim %s from yatai service", compoundAINimRequest.Spec.BentoTag)
	return
}

func (r *CompoundAINimRequestReconciler) getImageBuilderJobName() string {
	guid := xid.New()
	return fmt.Sprintf("yatai-compoundainim-image-builder-%s", guid.String())
}

func (r *CompoundAINimRequestReconciler) getModelSeederJobName() string {
	guid := xid.New()
	return fmt.Sprintf("yatai-model-seeder-%s", guid.String())
}

func (r *CompoundAINimRequestReconciler) getModelSeederJobLabels(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, model *nvidiacomv1alpha1.BentoModel) map[string]string {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINimRequest.Spec.BentoTag, ":")
	modelRepositoryName, _, modelVersion := xstrings.Partition(model.Tag, ":")
	return map[string]string{
		commonconsts.KubeLabelBentoRequest:         compoundAINimRequest.Name,
		commonconsts.KubeLabelIsModelSeeder:        "true",
		commonconsts.KubeLabelYataiModelRepository: modelRepositoryName,
		commonconsts.KubeLabelYataiModel:           modelVersion,
		commonconsts.KubeLabelYataiBentoRepository: compoundAINimRepositoryName,
		commonconsts.KubeLabelYataiBento:           compoundAINimVersion,
	}
}

func (r *CompoundAINimRequestReconciler) getModelSeederPodLabels(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, model *nvidiacomv1alpha1.BentoModel) map[string]string {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINimRequest.Spec.BentoTag, ":")
	modelRepositoryName, _, modelVersion := xstrings.Partition(model.Tag, ":")
	return map[string]string{
		commonconsts.KubeLabelBentoRequest:         compoundAINimRequest.Name,
		commonconsts.KubeLabelIsModelSeeder:        "true",
		commonconsts.KubeLabelIsBentoImageBuilder:  "true",
		commonconsts.KubeLabelYataiModelRepository: modelRepositoryName,
		commonconsts.KubeLabelYataiModel:           modelVersion,
		commonconsts.KubeLabelYataiBentoRepository: compoundAINimRepositoryName,
		commonconsts.KubeLabelYataiBento:           compoundAINimVersion,
	}
}

func (r *CompoundAINimRequestReconciler) getImageBuilderJobLabels(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) map[string]string {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINimRequest.Spec.BentoTag, ":")
	labels := map[string]string{
		commonconsts.KubeLabelBentoRequest:         compoundAINimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder:  "true",
		commonconsts.KubeLabelYataiBentoRepository: compoundAINimRepositoryName,
		commonconsts.KubeLabelYataiBento:           compoundAINimVersion,
	}

	if isSeparateModels(compoundAINimRequest) {
		labels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueTrue
	} else {
		labels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueFalse
	}
	return labels
}

func (r *CompoundAINimRequestReconciler) getImageBuilderPodLabels(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) map[string]string {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINimRequest.Spec.BentoTag, ":")
	return map[string]string{
		commonconsts.KubeLabelBentoRequest:         compoundAINimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder:  "true",
		commonconsts.KubeLabelYataiBentoRepository: compoundAINimRepositoryName,
		commonconsts.KubeLabelYataiBento:           compoundAINimVersion,
	}
}

func hash(text string) string {
	// nolint: gosec
	hasher := md5.New()
	hasher.Write([]byte(text))
	return hex.EncodeToString(hasher.Sum(nil))
}

func (r *CompoundAINimRequestReconciler) getModelPVCName(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, model *nvidiacomv1alpha1.BentoModel) string {
	storageClassName := getJuiceFSStorageClassName()
	var hashStr string
	ns := getModelNamespace(compoundAINimRequest)
	if ns == "" {
		hashStr = hash(fmt.Sprintf("%s:%s", storageClassName, model.Tag))
	} else {
		hashStr = hash(fmt.Sprintf("%s:%s:%s", storageClassName, ns, model.Tag))
	}
	pvcName := fmt.Sprintf("model-seeder-%s", hashStr)
	if len(pvcName) > 63 {
		pvcName = pvcName[:63]
	}
	return pvcName
}

func (r *CompoundAINimRequestReconciler) getJuiceFSModelPath(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest, model *nvidiacomv1alpha1.BentoModel) string {
	modelRepositoryName, _, modelVersion := xstrings.Partition(model.Tag, ":")
	ns := getModelNamespace(compoundAINimRequest)
	if isHuggingfaceModel(model) {
		modelVersion = "all"
	}
	var path string
	if ns == "" {
		path = fmt.Sprintf("models/.shared/%s/%s", modelRepositoryName, modelVersion)
	} else {
		path = fmt.Sprintf("models/%s/%s/%s", ns, modelRepositoryName, modelVersion)
	}
	return path
}

func isHuggingfaceModel(model *nvidiacomv1alpha1.BentoModel) bool {
	return strings.HasPrefix(model.DownloadURL, "hf://")
}

type GenerateModelPVCOption struct {
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
	Model                *nvidiacomv1alpha1.BentoModel
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) generateModelPVC(opt GenerateModelPVCOption) (pvc *corev1.PersistentVolumeClaim) {
	storageSize := resource.MustParse("100Gi")
	if opt.Model.Size != nil {
		storageSize = *opt.Model.Size
		minStorageSize := resource.MustParse("1Gi")
		if storageSize.Value() < minStorageSize.Value() {
			storageSize = minStorageSize
		}
		storageSize.Set(storageSize.Value() * 2)
	}
	path := r.getJuiceFSModelPath(opt.CompoundAINimRequest, opt.Model)
	pvcName := r.getModelPVCName(opt.CompoundAINimRequest, opt.Model)
	pvc = &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: opt.CompoundAINimRequest.Namespace,
			Annotations: map[string]string{
				"path": path,
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: storageSize,
				},
			},
			StorageClassName: ptr.To(getJuiceFSStorageClassName()),
		},
	}
	return
}

type GenerateModelSeederJobOption struct {
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
	Model                *nvidiacomv1alpha1.BentoModel
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) generateModelSeederJob(ctx context.Context, opt GenerateModelSeederJobOption) (job *batchv1.Job, err error) {
	// nolint: gosimple
	podTemplateSpec, err := r.generateModelSeederPodTemplateSpec(ctx, GenerateModelSeederPodTemplateSpecOption(opt))
	if err != nil {
		err = errors.Wrap(err, "generate model seeder pod template spec")
		return
	}
	kubeAnnotations := make(map[string]string)
	hashStr, err := r.getHashStr(opt.CompoundAINimRequest)
	if err != nil {
		err = errors.Wrap(err, "failed to get hash string")
		return
	}
	kubeAnnotations[KubeAnnotationCompoundAINimRequestHash] = hashStr
	job = &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        r.getModelSeederJobName(),
			Namespace:   opt.CompoundAINimRequest.Namespace,
			Labels:      r.getModelSeederJobLabels(opt.CompoundAINimRequest, opt.Model),
			Annotations: kubeAnnotations,
		},
		Spec: batchv1.JobSpec{
			Completions: ptr.To(int32(1)),
			Parallelism: ptr.To(int32(1)),
			PodFailurePolicy: &batchv1.PodFailurePolicy{
				Rules: []batchv1.PodFailurePolicyRule{
					{
						Action: batchv1.PodFailurePolicyActionFailJob,
						OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
							ContainerName: ptr.To(ModelSeederContainerName),
							Operator:      batchv1.PodFailurePolicyOnExitCodesOpIn,
							Values:        []int32{ModelSeederJobFailedExitCode},
						},
					},
				},
			},
			Template: *podTemplateSpec,
		},
	}
	err = ctrl.SetControllerReference(opt.CompoundAINimRequest, job, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for job %s", job.Name)
		return
	}
	return
}

type GenerateModelSeederPodTemplateSpecOption struct {
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
	Model                *nvidiacomv1alpha1.BentoModel
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) generateModelSeederPodTemplateSpec(ctx context.Context, opt GenerateModelSeederPodTemplateSpecOption) (pod *corev1.PodTemplateSpec, err error) {
	kubeLabels := r.getModelSeederPodLabels(opt.CompoundAINimRequest, opt.Model)

	volumes := make([]corev1.Volume, 0)

	volumeMounts := make([]corev1.VolumeMount, 0)

	yataiAPITokenSecretName := ""

	internalImages := commonconfig.GetInternalImages()
	logrus.Infof("Model seeder is using the images %v", *internalImages)

	downloaderContainerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1000m"),
			corev1.ResourceMemory: resource.MustParse("3000Mi"),
		},
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("100m"),
			corev1.ResourceMemory: resource.MustParse("1000Mi"),
		},
	}

	downloaderContainerEnvFrom := opt.CompoundAINimRequest.Spec.DownloaderContainerEnvFrom

	if yataiAPITokenSecretName != "" {
		downloaderContainerEnvFrom = append(downloaderContainerEnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: yataiAPITokenSecretName,
				},
			},
		})
	}

	containers := make([]corev1.Container, 0)

	model := opt.Model
	modelRepositoryName, _, modelVersion := xstrings.Partition(model.Tag, ":")
	modelDownloadURL := model.DownloadURL
	modelDownloadHeader := ""
	if modelDownloadURL == "" {
		var yataiClient_ **yataiclient.YataiClient
		var yataiConf_ **commonconfig.YataiConfig

		yataiClient_, yataiConf_, err = r.getYataiClient(ctx)
		if err != nil {
			err = errors.Wrap(err, "get yatai client")
			return
		}

		if yataiClient_ == nil || yataiConf_ == nil {
			err = errors.New("can't get yatai client, please check yatai configuration")
			return
		}

		yataiClient := *yataiClient_
		yataiConf := *yataiConf_

		var model_ *schemasv1.ModelFullSchema
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting model %s from yatai service", model.Tag)
		model_, err = yataiClient.GetModel(ctx, modelRepositoryName, modelVersion)
		if err != nil {
			err = errors.Wrap(err, "get model")
			return
		}
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Model %s is got from yatai service", model.Tag)

		if model_.TransmissionStrategy != nil && *model_.TransmissionStrategy == modelschemas.TransmissionStrategyPresignedURL {
			var model0 *schemasv1.ModelSchema
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for model %s from yatai service", model.Tag)
			model0, err = yataiClient.PresignModelDownloadURL(ctx, modelRepositoryName, modelVersion)
			if err != nil {
				err = errors.Wrap(err, "presign model download url")
				return
			}
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Presigned url for model %s is got from yatai service", model.Tag)
			modelDownloadURL = model0.PresignedDownloadUrl
		} else {
			modelDownloadURL = fmt.Sprintf("%s/api/v1/model_repositories/%s/models/%s/download", yataiConf.Endpoint, modelRepositoryName, modelVersion)
			modelDownloadHeader = fmt.Sprintf("%s: %s:%s:$%s", commonconsts.YataiApiTokenHeaderName, commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
		}
	}

	modelDirPath := "/juicefs-workspace"
	var modelSeedCommandOutput bytes.Buffer
	err = template.Must(template.New("script").Parse(`
set -e

mkdir -p {{.ModelDirPath}}
url="{{.ModelDownloadURL}}"

if [[ ${url} == hf://* ]]; then
	if [ -f "{{.ModelDirPath}}/{{.ModelVersion}}.exists" ]; then
		echo "Model {{.ModelDirPath}}/{{.ModelVersion}}.exists already exists, skip downloading"
		exit 0
	fi
else
	if [ -f "{{.ModelDirPath}}/.exists" ]; then
		echo "Model {{.ModelDirPath}} already exists, skip downloading"
		exit 0
	fi
fi

cleanup() {
	echo "Cleaning up..."
	rm -rf /tmp/model
	rm -f /tmp/downloaded.tar
}

trap cleanup EXIT

if [[ ${url} == hf://* ]]; then
	mkdir -p /tmp/model
	hf_url="${url:5}"
	model_id=$(echo "$hf_url" | awk -F '@' '{print $1}')
	revision=$(echo "$hf_url" | awk -F '@' '{print $2}')
	endpoint=$(echo "$hf_url" | awk -F '@' '{print $3}')
	export HF_ENDPOINT=${endpoint}

	echo "Downloading model ${model_id} (endpoint=${endpoint}, revision=${revision}) from Huggingface..."
	huggingface-cli download ${model_id} --revision ${revision} --cache-dir {{.ModelDirPath}}
else
	echo "Downloading model {{.ModelRepositoryName}}:{{.ModelVersion}} to /tmp/downloaded.tar..."
	if [[ ${url} == s3://* ]]; then
		echo "Downloading from s3..."
		aws s3 cp ${url} /tmp/downloaded.tar
	elif [[ ${url} == gs://* ]]; then
		echo "Downloading from GCS..."
		gsutil cp ${url} /tmp/downloaded.tar
	else
		curl --fail -L -H "{{.ModelDownloadHeader}}" ${url} --output /tmp/downloaded.tar --progress-bar
	fi
	cd {{.ModelDirPath}}
	echo "Extracting model tar file..."
	tar -xvf /tmp/downloaded.tar
fi

if [[ ${url} == hf://* ]]; then
	echo "Creating {{.ModelDirPath}}/{{.ModelVersion}}.exists file..."
	touch {{.ModelDirPath}}/{{.ModelVersion}}.exists
else
	echo "Creating {{.ModelDirPath}}/.exists file..."
	touch {{.ModelDirPath}}/.exists
fi

echo "Done"
`)).Execute(&modelSeedCommandOutput, map[string]interface{}{
		"ModelDirPath":        modelDirPath,
		"ModelDownloadURL":    modelDownloadURL,
		"ModelDownloadHeader": modelDownloadHeader,
		"ModelRepositoryName": modelRepositoryName,
		"ModelVersion":        modelVersion,
		"HuggingfaceModelDir": fmt.Sprintf("models--%s", strings.ReplaceAll(modelRepositoryName, "/", "--")),
	})
	if err != nil {
		err = errors.Wrap(err, "failed to generate download command")
		return
	}
	modelSeedCommand := modelSeedCommandOutput.String()
	pvcName := r.getModelPVCName(opt.CompoundAINimRequest, model)
	volumes = append(volumes, corev1.Volume{
		Name: pvcName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
	containers = append(containers, corev1.Container{
		Name:  ModelSeederContainerName,
		Image: internalImages.BentoDownloader,
		Command: []string{
			"bash",
			"-c",
			modelSeedCommand,
		},
		VolumeMounts: append(volumeMounts, corev1.VolumeMount{
			Name:      pvcName,
			MountPath: modelDirPath,
		}),
		Resources: downloaderContainerResources,
		EnvFrom:   downloaderContainerEnvFrom,
		Env: []corev1.EnvVar{
			{
				Name:  "AWS_EC2_METADATA_DISABLED",
				Value: "true",
			},
		},
	})

	kubeAnnotations := make(map[string]string)
	kubeAnnotations[KubeAnnotationCompoundAINimRequestModelSeederHash] = opt.CompoundAINimRequest.Annotations[KubeAnnotationCompoundAINimRequestModelSeederHash]

	pod = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      kubeLabels,
			Annotations: kubeAnnotations,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Volumes:       volumes,
			Containers:    containers,
		},
	}

	var globalExtraPodSpec *compoundaiCommon.ExtraPodSpec

	configNamespace, err := commonconfig.GetYataiImageBuilderNamespace(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      name,
		}, secret)
		return secret, errors.Wrap(err, "get secret")
	})
	if err != nil {
		err = errors.Wrap(err, "failed to get Yatai image builder namespace")
		return
	}

	configCmName := "yatai-image-builder-config"
	r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateModelSeederPod", "Getting configmap %s from namespace %s", configCmName, configNamespace)
	configCm := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: configCmName, Namespace: configNamespace}, configCm)
	configCmIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !configCmIsNotFound {
		err = errors.Wrap(err, "failed to get configmap")
		return
	}
	err = nil

	if !configCmIsNotFound {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateModelSeederPod", "Configmap %s is got from namespace %s", configCmName, configNamespace)

		globalExtraPodSpec = &compoundaiCommon.ExtraPodSpec{}

		if val, ok := configCm.Data["extra_pod_spec"]; ok {
			err = yaml.Unmarshal([]byte(val), globalExtraPodSpec)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_pod_spec, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}
	} else {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateModelSeederPod", "Configmap %s is not found in namespace %s", configCmName, configNamespace)
	}

	if globalExtraPodSpec != nil {
		pod.Spec.PriorityClassName = globalExtraPodSpec.PriorityClassName
		pod.Spec.SchedulerName = globalExtraPodSpec.SchedulerName
		pod.Spec.NodeSelector = globalExtraPodSpec.NodeSelector
		pod.Spec.Affinity = globalExtraPodSpec.Affinity
		pod.Spec.Tolerations = globalExtraPodSpec.Tolerations
		pod.Spec.TopologySpreadConstraints = globalExtraPodSpec.TopologySpreadConstraints
		pod.Spec.ServiceAccountName = globalExtraPodSpec.ServiceAccountName
	}

	injectPodAffinity(&pod.Spec, opt.CompoundAINimRequest)

	return
}

type GenerateImageBuilderJobOption struct {
	ImageInfo            ImageInfo
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
}

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) generateImageBuilderJob(ctx context.Context, opt GenerateImageBuilderJobOption) (job *batchv1.Job, err error) {
	// nolint: gosimple
	podTemplateSpec, err := r.generateImageBuilderPodTemplateSpec(ctx, GenerateImageBuilderPodTemplateSpecOption(opt))
	if err != nil {
		err = errors.Wrap(err, "generate image builder pod template spec")
		return
	}
	kubeAnnotations := make(map[string]string)
	hashStr, err := r.getHashStr(opt.CompoundAINimRequest)
	if err != nil {
		err = errors.Wrap(err, "failed to get hash string")
		return
	}
	kubeAnnotations[KubeAnnotationCompoundAINimRequestHash] = hashStr
	job = &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        r.getImageBuilderJobName(),
			Namespace:   opt.CompoundAINimRequest.Namespace,
			Labels:      r.getImageBuilderJobLabels(opt.CompoundAINimRequest),
			Annotations: kubeAnnotations,
		},
		Spec: batchv1.JobSpec{
			Completions: ptr.To(int32(1)),
			Parallelism: ptr.To(int32(1)),
			PodFailurePolicy: &batchv1.PodFailurePolicy{
				Rules: []batchv1.PodFailurePolicyRule{
					{
						Action: batchv1.PodFailurePolicyActionFailJob,
						OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
							ContainerName: ptr.To(BuilderContainerName),
							Operator:      batchv1.PodFailurePolicyOnExitCodesOpIn,
							Values:        []int32{BuilderJobFailedExitCode},
						},
					},
				},
			},
			Template: *podTemplateSpec,
		},
	}
	err = ctrl.SetControllerReference(opt.CompoundAINimRequest, job, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for job %s", job.Name)
		return
	}
	return
}

func injectPodAffinity(podSpec *corev1.PodSpec, compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) {
	if podSpec.Affinity == nil {
		podSpec.Affinity = &corev1.Affinity{}
	}

	if podSpec.Affinity.PodAffinity == nil {
		podSpec.Affinity.PodAffinity = &corev1.PodAffinity{}
	}

	podSpec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution = append(podSpec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution, corev1.WeightedPodAffinityTerm{
		Weight: 100,
		PodAffinityTerm: corev1.PodAffinityTerm{
			LabelSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					commonconsts.KubeLabelBentoRequest: compoundAINimRequest.Name,
				},
			},
			TopologyKey: corev1.LabelHostname,
		},
	})
}

const BuilderContainerName = "builder"
const BuilderJobFailedExitCode = 42
const ModelSeederContainerName = "seeder"
const ModelSeederJobFailedExitCode = 42

type GenerateImageBuilderPodTemplateSpecOption struct {
	ImageInfo            ImageInfo
	CompoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest
}

//nolint:gocyclo,nakedret
func (r *CompoundAINimRequestReconciler) generateImageBuilderPodTemplateSpec(ctx context.Context, opt GenerateImageBuilderPodTemplateSpecOption) (pod *corev1.PodTemplateSpec, err error) {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(opt.CompoundAINimRequest.Spec.BentoTag, ":")
	kubeLabels := r.getImageBuilderPodLabels(opt.CompoundAINimRequest)

	inClusterImageName := opt.ImageInfo.InClusterImageName

	dockerConfigJSONSecretName := opt.ImageInfo.DockerConfigJSONSecretName

	dockerRegistryInsecure := opt.ImageInfo.DockerRegistryInsecure

	volumes := []corev1.Volume{
		{
			Name: "yatai",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
		{
			Name: "workspace",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
	}

	volumeMounts := []corev1.VolumeMount{
		{
			Name:      "yatai",
			MountPath: "/yatai",
		},
		{
			Name:      "workspace",
			MountPath: "/workspace",
		},
	}

	if dockerConfigJSONSecretName != "" {
		volumes = append(volumes, corev1.Volume{
			Name: dockerConfigJSONSecretName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName: dockerConfigJSONSecretName,
					Items: []corev1.KeyToPath{
						{
							Key:  ".dockerconfigjson",
							Path: "config.json",
						},
					},
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      dockerConfigJSONSecretName,
			MountPath: "/kaniko/.docker/",
		})
	}

	var compoundAINim *schemasv1.BentoFullSchema
	yataiAPITokenSecretName := ""
	compoundAINimDownloadURL := opt.CompoundAINimRequest.Spec.DownloadURL
	compoundAINimDownloadHeader := ""

	if compoundAINimDownloadURL == "" {
		var yataiClient_ **yataiclient.YataiClient
		var yataiConf_ **commonconfig.YataiConfig

		yataiClient_, yataiConf_, err = r.getYataiClientWithAuth(ctx, opt.CompoundAINimRequest)
		if err != nil {
			err = errors.Wrap(err, "get yatai client")
			return
		}

		if yataiClient_ == nil || yataiConf_ == nil {
			err = errors.New("can't get yatai client, please check yatai configuration")
			return
		}

		yataiClient := *yataiClient_
		yataiConf := *yataiConf_

		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting compoundAINim %s from yatai service", opt.CompoundAINimRequest.Spec.BentoTag)
		compoundAINim, err = yataiClient.GetBento(ctx, compoundAINimRepositoryName, compoundAINimVersion)
		if err != nil {
			err = errors.Wrap(err, "get compoundAINim")
			return
		}
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got compoundAINim %s from yatai service", opt.CompoundAINimRequest.Spec.BentoTag)

		if compoundAINim.TransmissionStrategy != nil && *compoundAINim.TransmissionStrategy == modelschemas.TransmissionStrategyPresignedURL {
			var compoundAINim_ *schemasv1.BentoSchema
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for compoundAINim %s from yatai service", opt.CompoundAINimRequest.Spec.BentoTag)
			compoundAINim_, err = yataiClient.PresignBentoDownloadURL(ctx, compoundAINimRepositoryName, compoundAINimVersion)
			if err != nil {
				err = errors.Wrap(err, "presign compoundAINim download url")
				return
			}
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got presigned url for compoundAINim %s from yatai service", opt.CompoundAINimRequest.Spec.BentoTag)
			compoundAINimDownloadURL = compoundAINim_.PresignedDownloadUrl
		} else {
			compoundAINimDownloadURL = fmt.Sprintf("%s/api/v1/bento_repositories/%s/bentos/%s/download", yataiConf.Endpoint, compoundAINimRepositoryName, compoundAINimVersion)
			compoundAINimDownloadHeader = fmt.Sprintf("%s: %s:%s:$%s", commonconsts.YataiApiTokenHeaderName, commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
		}

		// nolint: gosec
		yataiAPITokenSecretName = "yatai-api-token"

		yataiAPITokenSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      yataiAPITokenSecretName,
				Namespace: opt.CompoundAINimRequest.Namespace,
			},
			StringData: map[string]string{
				commonconsts.EnvYataiApiToken: yataiConf.ApiToken,
			},
		}

		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s in namespace %s", yataiAPITokenSecretName, opt.CompoundAINimRequest.Namespace)
		_yataiAPITokenSecret := &corev1.Secret{}
		err = r.Get(ctx, types.NamespacedName{Namespace: opt.CompoundAINimRequest.Namespace, Name: yataiAPITokenSecretName}, _yataiAPITokenSecret)
		isNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrapf(err, "failed to get secret %s", yataiAPITokenSecretName)
			return
		}

		if isNotFound {
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is not found, so creating it in namespace %s", yataiAPITokenSecretName, opt.CompoundAINimRequest.Namespace)
			err = r.Create(ctx, yataiAPITokenSecret)
			isExists := k8serrors.IsAlreadyExists(err)
			if err != nil && !isExists {
				err = errors.Wrapf(err, "failed to create secret %s", yataiAPITokenSecretName)
				return
			}
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is created in namespace %s", yataiAPITokenSecretName, opt.CompoundAINimRequest.Namespace)
		} else {
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is found in namespace %s, so updating it", yataiAPITokenSecretName, opt.CompoundAINimRequest.Namespace)
			err = r.Update(ctx, yataiAPITokenSecret)
			if err != nil {
				err = errors.Wrapf(err, "failed to update secret %s", yataiAPITokenSecretName)
				return
			}
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is updated in namespace %s", yataiAPITokenSecretName, opt.CompoundAINimRequest.Namespace)
		}
	}
	internalImages := commonconfig.GetInternalImages()
	logrus.Infof("Image builder is using the images %v", *internalImages)

	buildEngine := getCompoundAINimImageBuildEngine()

	privileged := buildEngine != CompoundAINimImageBuildEngineBuildkitRootless

	compoundAINimDownloadCommandTemplate, err := template.New("downloadCommand").Parse(`
set -e

mkdir -p /workspace/buildcontext
url="{{.CompoundAINimDownloadURL}}"
echo "Downloading compoundAINim {{.CompoundAINimRepositoryName}}:{{.CompoundAINimVersion}} to /tmp/downloaded.tar..."
if [[ ${url} == s3://* ]]; then
	echo "Downloading from s3..."
	aws s3 cp ${url} /tmp/downloaded.tar
elif [[ ${url} == gs://* ]]; then
	echo "Downloading from GCS..."
	gsutil cp ${url} /tmp/downloaded.tar
else
	curl --fail -L -H "{{.CompoundAINimDownloadHeader}}" ${url} --output /tmp/downloaded.tar --progress-bar
fi
cd /workspace/buildcontext
echo "Extracting compoundAINim tar file..."
tar -xvf /tmp/downloaded.tar
echo "Removing compoundAINim tar file..."
rm /tmp/downloaded.tar
{{if not .Privileged}}
echo "Changing directory permission..."
chown -R 1000:1000 /workspace
{{end}}
echo "Done"
	`)

	if err != nil {
		err = errors.Wrap(err, "failed to parse download command template")
		return
	}

	var compoundAINimDownloadCommandBuffer bytes.Buffer

	err = compoundAINimDownloadCommandTemplate.Execute(&compoundAINimDownloadCommandBuffer, map[string]interface{}{
		"CompoundAINimDownloadURL":    compoundAINimDownloadURL,
		"CompoundAINimDownloadHeader": compoundAINimDownloadHeader,
		"CompoundAINimRepositoryName": compoundAINimRepositoryName,
		"CompoundAINimVersion":        compoundAINimVersion,
		"Privileged":                  privileged,
	})
	if err != nil {
		err = errors.Wrap(err, "failed to execute download command template")
		return
	}

	compoundAINimDownloadCommand := compoundAINimDownloadCommandBuffer.String()

	downloaderContainerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1000m"),
			corev1.ResourceMemory: resource.MustParse("3000Mi"),
		},
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("100m"),
			corev1.ResourceMemory: resource.MustParse("1000Mi"),
		},
	}

	downloaderContainerEnvFrom := opt.CompoundAINimRequest.Spec.DownloaderContainerEnvFrom

	if yataiAPITokenSecretName != "" {
		downloaderContainerEnvFrom = append(downloaderContainerEnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: yataiAPITokenSecretName,
				},
			},
		})
	}

	initContainers := []corev1.Container{
		{
			Name:  "compoundainim-downloader",
			Image: internalImages.BentoDownloader,
			Command: []string{
				"bash",
				"-c",
				compoundAINimDownloadCommand,
			},
			VolumeMounts: volumeMounts,
			Resources:    downloaderContainerResources,
			EnvFrom:      downloaderContainerEnvFrom,
			Env: []corev1.EnvVar{
				{
					Name:  "AWS_EC2_METADATA_DISABLED",
					Value: "true",
				},
			},
		},
	}

	containers := make([]corev1.Container, 0)

	separateModels := isSeparateModels(opt.CompoundAINimRequest)

	models := opt.CompoundAINimRequest.Spec.Models
	modelsSeen := map[string]struct{}{}
	for _, model := range models {
		modelsSeen[model.Tag] = struct{}{}
	}

	if compoundAINim != nil {
		for _, modelTag := range compoundAINim.Manifest.Models {
			if _, ok := modelsSeen[modelTag]; !ok {
				models = append(models, nvidiacomv1alpha1.BentoModel{
					Tag: modelTag,
				})
			}
		}
	}

	for idx, model := range models {
		if separateModels {
			continue
		}
		modelRepositoryName, _, modelVersion := xstrings.Partition(model.Tag, ":")
		modelDownloadURL := model.DownloadURL
		modelDownloadHeader := ""
		if modelDownloadURL == "" {
			if compoundAINim == nil {
				continue
			}

			var yataiClient_ **yataiclient.YataiClient
			var yataiConf_ **commonconfig.YataiConfig

			yataiClient_, yataiConf_, err = r.getYataiClientWithAuth(ctx, opt.CompoundAINimRequest)

			if err != nil {
				err = errors.Wrap(err, "get yatai client")
				return
			}

			if yataiClient_ == nil || yataiConf_ == nil {
				err = errors.New("can't get yatai client, please check yatai configuration")
				return
			}

			yataiClient := *yataiClient_
			yataiConf := *yataiConf_

			var model_ *schemasv1.ModelFullSchema
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting model %s from yatai service", model.Tag)
			model_, err = yataiClient.GetModel(ctx, modelRepositoryName, modelVersion)
			if err != nil {
				err = errors.Wrap(err, "get model")
				return
			}
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Model %s is got from yatai service", model.Tag)

			if model_.TransmissionStrategy != nil && *model_.TransmissionStrategy == modelschemas.TransmissionStrategyPresignedURL {
				var model0 *schemasv1.ModelSchema
				r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for model %s from yatai service", model.Tag)
				model0, err = yataiClient.PresignModelDownloadURL(ctx, modelRepositoryName, modelVersion)
				if err != nil {
					err = errors.Wrap(err, "presign model download url")
					return
				}
				r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Presigned url for model %s is got from yatai service", model.Tag)
				modelDownloadURL = model0.PresignedDownloadUrl
			} else {
				modelDownloadURL = fmt.Sprintf("%s/api/v1/model_repositories/%s/models/%s/download", yataiConf.Endpoint, modelRepositoryName, modelVersion)
				modelDownloadHeader = fmt.Sprintf("%s: %s:%s:$%s", commonconsts.YataiApiTokenHeaderName, commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
			}
		}
		modelRepositoryDirPath := fmt.Sprintf("/workspace/buildcontext/models/%s", modelRepositoryName)
		modelDirPath := filepath.Join(modelRepositoryDirPath, modelVersion)
		var modelDownloadCommandOutput bytes.Buffer
		err = template.Must(template.New("script").Parse(`
set -e

mkdir -p {{.ModelDirPath}}
url="{{.ModelDownloadURL}}"
echo "Downloading model {{.ModelRepositoryName}}:{{.ModelVersion}} to /tmp/downloaded.tar..."
if [[ ${url} == s3://* ]]; then
	echo "Downloading from s3..."
	aws s3 cp ${url} /tmp/downloaded.tar
elif [[ ${url} == gs://* ]]; then
	echo "Downloading from GCS..."
	gsutil cp ${url} /tmp/downloaded.tar
else
	curl --fail -L -H "{{.ModelDownloadHeader}}" ${url} --output /tmp/downloaded.tar --progress-bar
fi
cd {{.ModelDirPath}}
echo "Extracting model tar file..."
tar -xvf /tmp/downloaded.tar
echo -n '{{.ModelVersion}}' > {{.ModelRepositoryDirPath}}/latest
echo "Removing model tar file..."
rm /tmp/downloaded.tar
{{if not .Privileged}}
echo "Changing directory permission..."
chown -R 1000:1000 /workspace
{{end}}
echo "Done"
`)).Execute(&modelDownloadCommandOutput, map[string]interface{}{
			"ModelDirPath":           modelDirPath,
			"ModelDownloadURL":       modelDownloadURL,
			"ModelDownloadHeader":    modelDownloadHeader,
			"ModelRepositoryDirPath": modelRepositoryDirPath,
			"ModelRepositoryName":    modelRepositoryName,
			"ModelVersion":           modelVersion,
			"Privileged":             privileged,
		})
		if err != nil {
			err = errors.Wrap(err, "failed to generate download command")
			return
		}
		modelDownloadCommand := modelDownloadCommandOutput.String()
		initContainers = append(initContainers, corev1.Container{
			Name:  fmt.Sprintf("model-downloader-%d", idx),
			Image: internalImages.BentoDownloader,
			Command: []string{
				"bash",
				"-c",
				modelDownloadCommand,
			},
			VolumeMounts: volumeMounts,
			Resources:    downloaderContainerResources,
			EnvFrom:      downloaderContainerEnvFrom,
			Env: []corev1.EnvVar{
				{
					Name:  "AWS_EC2_METADATA_DISABLED",
					Value: "true",
				},
			},
		})
	}

	var globalExtraPodMetadata *compoundaiCommon.ExtraPodMetadata
	var globalExtraPodSpec *compoundaiCommon.ExtraPodSpec
	var globalExtraContainerEnv []corev1.EnvVar
	var globalDefaultImageBuilderContainerResources *corev1.ResourceRequirements
	var buildArgs []string
	var builderArgs []string

	configNamespace, err := commonconfig.GetYataiImageBuilderNamespace(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      name,
		}, secret)
		return secret, errors.Wrap(err, "get secret")
	})
	if err != nil {
		err = errors.Wrap(err, "failed to get Yatai image builder namespace")
		return
	}

	configCmName := "yatai-image-builder-config"
	r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting configmap %s from namespace %s", configCmName, configNamespace)
	configCm := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: configCmName, Namespace: configNamespace}, configCm)
	configCmIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !configCmIsNotFound {
		err = errors.Wrap(err, "failed to get configmap")
		return
	}
	err = nil // nolint: ineffassign

	if !configCmIsNotFound {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is got from namespace %s", configCmName, configNamespace)

		globalExtraPodMetadata = &compoundaiCommon.ExtraPodMetadata{}

		if val, ok := configCm.Data["extra_pod_metadata"]; ok {
			err = yaml.Unmarshal([]byte(val), globalExtraPodMetadata)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_pod_metadata, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		globalExtraPodSpec = &compoundaiCommon.ExtraPodSpec{}

		if val, ok := configCm.Data["extra_pod_spec"]; ok {
			err = yaml.Unmarshal([]byte(val), globalExtraPodSpec)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_pod_spec, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		globalExtraContainerEnv = []corev1.EnvVar{}

		if val, ok := configCm.Data["extra_container_env"]; ok {
			err = yaml.Unmarshal([]byte(val), &globalExtraContainerEnv)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_container_env, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		if val, ok := configCm.Data["default_image_builder_container_resources"]; ok {
			globalDefaultImageBuilderContainerResources = &corev1.ResourceRequirements{}
			err = yaml.Unmarshal([]byte(val), globalDefaultImageBuilderContainerResources)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal default_image_builder_container_resources, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		buildArgs = []string{}

		if val, ok := configCm.Data["build_args"]; ok {
			err = yaml.Unmarshal([]byte(val), &buildArgs)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal build_args, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		builderArgs = []string{}
		if val, ok := configCm.Data["builder_args"]; ok {
			err = yaml.Unmarshal([]byte(val), &builderArgs)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal builder_args, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}
		logrus.Info("passed in builder args: ", builderArgs)
	} else {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is not found in namespace %s", configCmName, configNamespace)
	}

	if buildArgs == nil {
		buildArgs = make([]string, 0)
	}

	if opt.CompoundAINimRequest.Spec.BuildArgs != nil {
		buildArgs = append(buildArgs, opt.CompoundAINimRequest.Spec.BuildArgs...)
	}

	dockerFilePath := "/workspace/buildcontext/env/docker/Dockerfile"

	builderContainerEnvFrom := make([]corev1.EnvFromSource, 0)
	builderContainerEnvs := []corev1.EnvVar{
		{
			Name:  "DOCKER_CONFIG",
			Value: "/kaniko/.docker/",
		},
		{
			Name:  "IFS",
			Value: "''",
		},
	}

	kanikoCacheRepo := os.Getenv("KANIKO_CACHE_REPO")
	if kanikoCacheRepo == "" {
		kanikoCacheRepo = opt.ImageInfo.DockerRegistry.BentosRepositoryURIInCluster
	}

	kubeAnnotations := make(map[string]string)
	kubeAnnotations[KubeAnnotationCompoundAINimRequestImageBuiderHash] = opt.CompoundAINimRequest.Annotations[KubeAnnotationCompoundAINimRequestImageBuiderHash]

	command := []string{
		"/kaniko/executor",
	}
	args := []string{
		"--context=/workspace/buildcontext",
		"--verbosity=info",
		"--image-fs-extract-retry=3",
		"--cache=false",
		fmt.Sprintf("--cache-repo=%s", kanikoCacheRepo),
		"--compressed-caching=false",
		"--compression=zstd",
		"--compression-level=-7",
		fmt.Sprintf("--dockerfile=%s", dockerFilePath),
		fmt.Sprintf("--insecure=%v", dockerRegistryInsecure),
		fmt.Sprintf("--destination=%s", inClusterImageName),
	}

	kanikoSnapshotMode := os.Getenv("KANIKO_SNAPSHOT_MODE")
	if kanikoSnapshotMode != "" {
		args = append(args, fmt.Sprintf("--snapshot-mode=%s", kanikoSnapshotMode))
	}

	var builderImage string
	switch buildEngine {
	case CompoundAINimImageBuildEngineKaniko:
		builderImage = internalImages.Kaniko
		if isEstargzEnabled() {
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name:  "GGCR_EXPERIMENT_ESTARGZ",
				Value: "1",
			})
		}
	case CompoundAINimImageBuildEngineBuildkit:
		builderImage = internalImages.Buildkit
	case CompoundAINimImageBuildEngineBuildkitRootless:
		builderImage = internalImages.BuildkitRootless
	default:
		err = errors.Errorf("unknown compoundAINim image build engine %s", buildEngine)
		return
	}

	isBuildkit := buildEngine == CompoundAINimImageBuildEngineBuildkit || buildEngine == CompoundAINimImageBuildEngineBuildkitRootless

	if isBuildkit {
		output := fmt.Sprintf("type=image,name=%s,push=true,registry.insecure=%v", inClusterImageName, dockerRegistryInsecure)
		buildkitdFlags := []string{}
		if !privileged {
			buildkitdFlags = append(buildkitdFlags, "--oci-worker-no-process-sandbox")
		}
		if isEstargzEnabled() {
			buildkitdFlags = append(buildkitdFlags, "--oci-worker-snapshotter=stargz")
			output += ",oci-mediatypes=true,compression=estargz,force-compression=true"
		}
		if len(buildkitdFlags) > 0 {
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name:  "BUILDKITD_FLAGS",
				Value: strings.Join(buildkitdFlags, " "),
			})
		}
		command = []string{"buildctl-daemonless.sh"}
		args = []string{
			"build",
			"--frontend",
			"dockerfile.v0",
			"--local",
			"context=/workspace/buildcontext",
			"--local",
			fmt.Sprintf("dockerfile=%s", filepath.Dir(dockerFilePath)),
			"--output",
			output,
		}
		cacheRepo := os.Getenv("BUILDKIT_CACHE_REPO")
		if cacheRepo == "" {
			cacheRepo = opt.ImageInfo.DockerRegistry.BentosRepositoryURIInCluster
		}
		args = append(args, "--export-cache", fmt.Sprintf("type=registry,ref=%s:buildcache,mode=max,compression=zstd,ignore-error=true", cacheRepo))
		args = append(args, "--import-cache", fmt.Sprintf("type=registry,ref=%s:buildcache", cacheRepo))
	}

	var builderContainerSecurityContext *corev1.SecurityContext

	if buildEngine == CompoundAINimImageBuildEngineBuildkit {
		builderContainerSecurityContext = &corev1.SecurityContext{
			Privileged: ptr.To(true),
		}
	} else if buildEngine == CompoundAINimImageBuildEngineBuildkitRootless {
		kubeAnnotations["container.apparmor.security.beta.kubernetes.io/builder"] = "unconfined"
		builderContainerSecurityContext = &corev1.SecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type: corev1.SeccompProfileTypeUnconfined,
			},
			RunAsUser:  ptr.To(int64(1000)),
			RunAsGroup: ptr.To(int64(1000)),
		}
	}

	// add build args to pass via --build-arg
	for _, buildArg := range buildArgs {
		quotedBuildArg := unix.SingleQuote.Quote(buildArg)
		if isBuildkit {
			args = append(args, "--opt", fmt.Sprintf("build-arg:%s", quotedBuildArg))
		} else {
			args = append(args, fmt.Sprintf("--build-arg=%s", quotedBuildArg))
		}
	}
	// add other arguments to builder
	args = append(args, builderArgs...)
	logrus.Info("yatai-image-builder args: ", args)

	// nolint: gosec
	buildArgsSecretName := "yatai-image-builder-build-args"
	r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s from namespace %s", buildArgsSecretName, configNamespace)
	buildArgsSecret := &corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{Name: buildArgsSecretName, Namespace: configNamespace}, buildArgsSecret)
	buildArgsSecretIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !buildArgsSecretIsNotFound {
		err = errors.Wrap(err, "failed to get secret")
		return
	}

	if !buildArgsSecretIsNotFound {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is got from namespace %s", buildArgsSecretName, configNamespace)
		if configNamespace != opt.CompoundAINimRequest.Namespace {
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is in namespace %s, but CompoundAINimRequest is in namespace %s, so we need to copy the secret to CompoundAINimRequest namespace", buildArgsSecretName, configNamespace, opt.CompoundAINimRequest.Namespace)
			r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s in namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
			_buildArgsSecret := &corev1.Secret{}
			err = r.Get(ctx, types.NamespacedName{Namespace: opt.CompoundAINimRequest.Namespace, Name: buildArgsSecretName}, _buildArgsSecret)
			localBuildArgsSecretIsNotFound := k8serrors.IsNotFound(err)
			if err != nil && !localBuildArgsSecretIsNotFound {
				err = errors.Wrapf(err, "failed to get secret %s from namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
				return
			}
			if localBuildArgsSecretIsNotFound {
				r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Copying secret %s from namespace %s to namespace %s", buildArgsSecretName, configNamespace, opt.CompoundAINimRequest.Namespace)
				err = r.Create(ctx, &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      buildArgsSecretName,
						Namespace: opt.CompoundAINimRequest.Namespace,
					},
					Data: buildArgsSecret.Data,
				})
				if err != nil {
					err = errors.Wrapf(err, "failed to create secret %s in namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
					return
				}
			} else {
				r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is already in namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
				r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Updating secret %s in namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
				err = r.Update(ctx, &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      buildArgsSecretName,
						Namespace: opt.CompoundAINimRequest.Namespace,
					},
					Data: buildArgsSecret.Data,
				})
				if err != nil {
					err = errors.Wrapf(err, "failed to update secret %s in namespace %s", buildArgsSecretName, opt.CompoundAINimRequest.Namespace)
					return
				}
			}
		}

		for key := range buildArgsSecret.Data {
			envName := fmt.Sprintf("BENTOML_BUILD_ARG_%s", strings.ReplaceAll(strings.ToUpper(strcase.ToKebab(key)), "-", "_"))
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name: envName,
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: buildArgsSecretName,
						},
						Key: key,
					},
				},
			})

			if isBuildkit {
				args = append(args, "--opt", fmt.Sprintf("build-arg:%s=$(%s)", key, envName))
			} else {
				args = append(args, fmt.Sprintf("--build-arg=%s=$(%s)", key, envName))
			}
		}
	} else {
		r.Recorder.Eventf(opt.CompoundAINimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is not found in namespace %s", buildArgsSecretName, configNamespace)
	}

	builderContainerArgs := []string{
		"-c",
		fmt.Sprintf("sleep 15; %s && exit 0 || exit %d", shquot.POSIXShell(append(command, args...)), BuilderJobFailedExitCode), // TODO: remove once functionality exists to wait for istio sidecar.
	}

	container := corev1.Container{
		Name:            BuilderContainerName,
		Image:           builderImage,
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"sh"},
		Args:            builderContainerArgs,
		VolumeMounts:    volumeMounts,
		Env:             builderContainerEnvs,
		EnvFrom:         builderContainerEnvFrom,
		TTY:             true,
		Stdin:           true,
		SecurityContext: builderContainerSecurityContext,
	}

	if globalDefaultImageBuilderContainerResources != nil {
		container.Resources = *globalDefaultImageBuilderContainerResources
	}

	if opt.CompoundAINimRequest.Spec.ImageBuilderContainerResources != nil {
		container.Resources = *opt.CompoundAINimRequest.Spec.ImageBuilderContainerResources
	}

	containers = append(containers, container)

	pod = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      kubeLabels,
			Annotations: kubeAnnotations,
		},
		Spec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			Volumes:        volumes,
			InitContainers: initContainers,
			Containers:     containers,
		},
	}

	if globalExtraPodMetadata != nil {
		for k, v := range globalExtraPodMetadata.Annotations {
			pod.Annotations[k] = v
		}

		for k, v := range globalExtraPodMetadata.Labels {
			pod.Labels[k] = v
		}
	}

	if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodMetadata != nil {
		for k, v := range opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodMetadata.Annotations {
			pod.Annotations[k] = v
		}

		for k, v := range opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodMetadata.Labels {
			pod.Labels[k] = v
		}
	}

	if globalExtraPodSpec != nil {
		pod.Spec.PriorityClassName = globalExtraPodSpec.PriorityClassName
		pod.Spec.SchedulerName = globalExtraPodSpec.SchedulerName
		pod.Spec.NodeSelector = globalExtraPodSpec.NodeSelector
		pod.Spec.Affinity = globalExtraPodSpec.Affinity
		pod.Spec.Tolerations = globalExtraPodSpec.Tolerations
		pod.Spec.TopologySpreadConstraints = globalExtraPodSpec.TopologySpreadConstraints
		pod.Spec.ServiceAccountName = globalExtraPodSpec.ServiceAccountName
	}

	if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec != nil {
		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.PriorityClassName != "" {
			pod.Spec.PriorityClassName = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.PriorityClassName
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.SchedulerName != "" {
			pod.Spec.SchedulerName = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.SchedulerName
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.NodeSelector != nil {
			pod.Spec.NodeSelector = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.NodeSelector
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.Affinity != nil {
			pod.Spec.Affinity = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.Affinity
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.Tolerations != nil {
			pod.Spec.Tolerations = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.Tolerations
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints != nil {
			pod.Spec.TopologySpreadConstraints = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints
		}

		if opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.ServiceAccountName != "" {
			pod.Spec.ServiceAccountName = opt.CompoundAINimRequest.Spec.ImageBuilderExtraPodSpec.ServiceAccountName
		}
	}

	injectPodAffinity(&pod.Spec, opt.CompoundAINimRequest)

	if pod.Spec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.CompoundAINimRequest.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelYataiImageBuilderPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.CompoundAINimRequest.Namespace)
			return
		}
		if len(serviceAccounts.Items) > 0 {
			pod.Spec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			pod.Spec.ServiceAccountName = "default"
		}
	}

	for i, c := range pod.Spec.InitContainers {
		env := c.Env
		if globalExtraContainerEnv != nil {
			env = append(env, globalExtraContainerEnv...)
		}
		env = append(env, opt.CompoundAINimRequest.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.InitContainers[i].Env = env
	}
	for i, c := range pod.Spec.Containers {
		env := c.Env
		if globalExtraContainerEnv != nil {
			env = append(env, globalExtraContainerEnv...)
		}
		env = append(env, opt.CompoundAINimRequest.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.Containers[i].Env = env
	}

	return
}

func (r *CompoundAINimRequestReconciler) getHashStr(compoundAINimRequest *nvidiacomv1alpha1.CompoundAINimRequest) (string, error) {
	var hash uint64
	hash, err := hashstructure.Hash(struct {
		Spec        nvidiacomv1alpha1.CompoundAINimRequestSpec
		Labels      map[string]string
		Annotations map[string]string
	}{
		Spec:        compoundAINimRequest.Spec,
		Labels:      compoundAINimRequest.Labels,
		Annotations: compoundAINimRequest.Annotations,
	}, hashstructure.FormatV2, nil)
	if err != nil {
		err = errors.Wrap(err, "get compoundAINimRequest CR spec hash")
		return "", err
	}
	hashStr := strconv.FormatUint(hash, 10)
	return hashStr, nil
}

func getJuiceFSStorageClassName() string {
	if v := os.Getenv("JUICEFS_STORAGE_CLASS_NAME"); v != "" {
		return v
	}
	return "juicefs-sc"
}

const (
	trueStr = "true"
)

//nolint:nakedret
func (r *CompoundAINimRequestReconciler) doRegisterCompoundComponent() (err error) {
	logs := log.Log.WithValues("func", "doRegisterYataiComponent")

	ctx, cancel := context.WithTimeout(context.TODO(), time.Minute*5)
	defer cancel()

	logs.Info("getting yatai client")
	yataiClient, yataiConf, err := r.getYataiClient(ctx)
	if err != nil {
		err = errors.Wrap(err, "get yatai client")
		return
	}

	if yataiClient == nil || yataiConf == nil {
		logs.Info("can't get yatai client, skip registering")
		return
	}

	yataiClient_ := *yataiClient
	yataiConf_ := *yataiConf

	namespace, err := commonconfig.GetYataiImageBuilderNamespace(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      name,
		}, secret)
		return secret, errors.Wrap(err, "get secret")
	})
	if err != nil {
		err = errors.Wrap(err, "get yatai image builder namespace")
		return
	}

	_, err = yataiClient_.RegisterYataiComponent(ctx, yataiConf_.ClusterName, &schemasv1.RegisterYataiComponentSchema{
		Name:          modelschemas.YataiComponentNameImageBuilder,
		KubeNamespace: namespace,
		Version:       version.Version,
		SelectorLabels: map[string]string{
			"app.kubernetes.io/name": "yatai-image-builder",
		},
		Manifest: &modelschemas.YataiComponentManifestSchema{
			SelectorLabels: map[string]string{
				"app.kubernetes.io/name": "yatai-image-builder",
			},
			LatestCRDVersion: "v1alpha1",
		},
	})

	err = errors.Wrap(err, "register yatai component")
	return err
}

func (r *CompoundAINimRequestReconciler) registerCompoundComponent() {
	logs := log.Log.WithValues("func", "registerYataiComponent")
	err := r.doRegisterCompoundComponent()
	if err != nil {
		logs.Error(err, "registerYataiComponent")
	}
	ticker := time.NewTicker(time.Minute * 5)
	for range ticker.C {
		err := r.doRegisterCompoundComponent()
		if err != nil {
			logs.Error(err, "registerYataiComponent")
		}
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *CompoundAINimRequestReconciler) SetupWithManager(mgr ctrl.Manager) error {
	logs := log.Log.WithValues("func", "SetupWithManager")

	if os.Getenv("DISABLE_YATAI_COMPONENT_REGISTRATION") != trueStr {
		go r.registerCompoundComponent()
	} else {
		logs.Info("yatai component registration is disabled")
	}

	err := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.CompoundAINimRequest{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&nvidiacomv1alpha1.CompoundAINim{}).
		Owns(&batchv1.Job{}).
		WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
	return errors.Wrap(err, "failed to setup CompoundAINimRequest controller")
}
