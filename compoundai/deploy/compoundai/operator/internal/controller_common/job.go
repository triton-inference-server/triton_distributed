package controller_common

import (
	"context"
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	vcbatchv1alpha1 "volcano.sh/apis/pkg/apis/batch/v1alpha1"
	vcschedulingv1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
)

const (
	EntityHandlerTypeJob = "job"

	// EntityHandlerCreated indicates that the EntityHandler is created.
	EntityHandlerCreated = "ENTITY_HANDLER_CREATED"
	// EntityHandlerCompleted indicates that the EntityHandler has completed.
	EntityHandlerCompleted = "ENTITY_HANDLER_COMPLETED"
	// EntityHandlerPending indicates that the EntityHandler is in pending state.
	EntityHandlerPending = "ENTITY_HANDLER_PENDING"

	EntityHandlerCreatedState   = "EntityHandlerCreated"
	EntityHandlerCompletedState = "EntityHandlerCompleted"
	EntityHandlerFailedState    = "EntityHandlerFailed"
	EntityHandlerPendingState   = "EntityHandlerPending"
	EntityHandlerRunningState   = "EntityHandlerRunning"

	// VolcanoPodGroupOwnerKey helps the client-go index Volcano PodGroups based on their owner.
	// This way we can easily find a PodGroup owned by a given Job.
	VolcanoPodGroupOwnerKey = "owner"
)

type EntityHandlerJobDefinition interface {
	GetAdditionalEnvVars() []corev1.EnvVar
	GetCustomCommand() []string
	GetArguments() []string
	GetImage() string
	GetImagePullSecret() string
	GetCPU() resource.Quantity
	GetMemory() resource.Quantity
	GetGPUs() int
	GetPodSecurityContext() *corev1.PodSecurityContext
	GetTolerations() []corev1.Toleration
	GetNodeSelector() map[string]string
}

type JobFactory func() WrappedJob

type JobDetail struct {
	Name                   string
	Namespace              string
	CreatedConditionType   string
	CompletedConditionType string
	PendingConditionType   string
	CreatedState           string
	CompletedState         string
	FailedState            string
	PendingState           string
	RunningState           string
	Wrapped                WrappedJob
}

type PendingStatusDetails struct {
	Reason  string
	Message string
}

type WrappedJob interface {
	client.Object
	IsFail() bool
	IsComplete() bool
	IsPending() bool
	FetchPendingStatusDetails(ctx context.Context, cl client.Client) (*PendingStatusDetails, error)
	GetWrapped() client.Object
}

func (w *JobDetail) IsComplete() bool {
	if w.Wrapped == nil {
		return false
	}
	return w.Wrapped.IsComplete()
}

func (w *JobDetail) IsFail() bool {
	if w.Wrapped == nil {
		return false
	}
	return w.Wrapped.IsFail()
}

func (w *JobDetail) IsPending() bool {
	if w.Wrapped == nil || w.IsComplete() || w.IsFail() {
		return false
	}
	return w.Wrapped.IsPending()
}

func (w *JobDetail) GetJobCondition(ctx context.Context, cl client.Client) (*metav1.Condition, error) {
	jobState := &metav1.Condition{}
	if w.IsComplete() {
		jobState.Type = w.CompletedConditionType
		jobState.Status = metav1.ConditionTrue
		jobState.Reason = w.CompletedState
		jobState.Message = "The job has completed"
		return jobState, nil
	}
	if w.IsFail() {
		jobState.Type = w.CompletedConditionType
		jobState.Status = metav1.ConditionFalse
		jobState.Reason = w.FailedState
		jobState.Message = "The job has failed"
		return jobState, nil
	}
	if w.IsPending() {
		jobState.Type = w.PendingConditionType
		jobState.Status = metav1.ConditionTrue
		jobState.Reason = w.PendingState
		jobState.Message = "The job is pending"
		// We want to know why a job is stuck in pending.
		// e.g. not enough resources
		if details, err := w.Wrapped.FetchPendingStatusDetails(ctx, cl); details != nil || err != nil {
			if err != nil {
				return nil, err
			}
			if details != nil {
				if details.Reason != "" {
					jobState.Reason = details.Reason
				}
				if details.Message != "" {
					jobState.Message = details.Message
				}
			}
		}
		return jobState, nil
	}
	// running state
	jobState.Type = w.PendingConditionType
	jobState.Status = metav1.ConditionFalse
	jobState.Reason = w.RunningState
	jobState.Message = "The job is running"
	return jobState, nil
}

type reconcileJobCustomResource interface {
	client.Object
	StatusConditionsMutator
	StatusConditionsProvider
	StateMutator
}

func ReconcileJob(ctx context.Context, cl client.Client, crd reconcileJobCustomResource, jobDetail *JobDetail, jobFactory JobFactory) (bool, error) {
	logger := log.FromContext(ctx)
	jobNamespacedName := types.NamespacedName{Name: jobDetail.Name, Namespace: crd.GetNamespace()}
	err := cl.Get(ctx, jobNamespacedName, jobDetail.Wrapped.GetWrapped())
	if err != nil && client.IgnoreNotFound(err) != nil {
		logger.Error(err, "Failed to retrieve job", "job", jobDetail.Name)
		return false, err
	}
	// If job does not exist, create a new one
	if err != nil {
		jobDetail.Wrapped = jobFactory()
		if err := controllerutil.SetControllerReference(crd, jobDetail.Wrapped.GetWrapped(), cl.Scheme()); err != nil {
			logger.Error(err, "Failed to set controller reference", "job", jobNamespacedName)
			return false, err
		}
		err = cl.Create(ctx, jobDetail.Wrapped.GetWrapped())
		if err != nil {
			logger.Error(err, "Failed to create job", "job", jobNamespacedName)
			return false, err
		}
		logger.Info("Job successfully created", "job", jobNamespacedName)
		UpdateCondition(crd, jobDetail.CreatedConditionType, metav1.ConditionTrue, jobDetail.CreatedState, "The job has been created")
	}
	// Get job message, most relevant for jobs that are stuck in pending, and we want to know why.
	state, err := jobDetail.GetJobCondition(ctx, cl)
	if err != nil {
		logger.Error(err, "Failed to get condition from JobDetail")
	}
	if UpdateCondition(crd, state.Type, state.Status, state.Reason, state.Message) {
		logger.Info(state.Message, "job", jobNamespacedName)
	}
	if jobDetail.IsFail() {
		crd.SetState(CrdFailed)
	}
	return jobDetail.IsComplete(), nil
}

type EntityHandler interface {
	GetType() string
	GetJobDefinition() EntityHandlerJobDefinition
}

func RetrieveEntityHandlerJobDetail(crd client.Object, entityHandler EntityHandler, index int) *JobDetail {
	jobDetail := &JobDetail{
		Name:                   fmt.Sprintf("%v-entity-handler-%v", crd.GetName(), index),
		CreatedConditionType:   fmt.Sprintf("%v_%v", EntityHandlerCreated, index),
		CompletedConditionType: fmt.Sprintf("%v_%v", EntityHandlerCompleted, index),
		PendingConditionType:   fmt.Sprintf("%v_%v", EntityHandlerPending, index),
		CreatedState:           EntityHandlerCreatedState,
		CompletedState:         EntityHandlerCompletedState,
		FailedState:            EntityHandlerFailedState,
		PendingState:           EntityHandlerPendingState,
		RunningState:           EntityHandlerRunningState,
	}
	if entityHandler.GetType() == EntityHandlerTypeJob {
		jobDetail.Wrapped = &JobWrapper{&batchv1.Job{}}
	}
	return jobDetail
}

func GenerateEntityHandlerJobFactory(operatorName string, crd metav1.Object, jobDetail *JobDetail, pvc *corev1.PersistentVolumeClaim, handler EntityHandler, entity Entity, basePath *string, additionalEnvVars []corev1.EnvVar, additionalLabels map[string]string) JobFactory {
	if handler.GetType() == EntityHandlerTypeJob && handler.GetJobDefinition() != nil {
		return GenerateK8sJobFactory(operatorName, crd, jobDetail.Name, pvc, handler.GetJobDefinition(), entity, basePath, additionalEnvVars, additionalLabels)
	}
	return nil
}

type Entity interface {
	GetType() string
	GetName() string
	GetCheckpointName() *string
	GetPath() *string
	GetExportPathToEnv() *string
	GetAdditionalEnvVars() []corev1.EnvVar
	GetCustomCommandOverride() []string
	GetArgumentsOverride() []string
	GetSecurityContextOverride() *corev1.PodSecurityContext
	GetTolerationsOverride() []corev1.Toleration
	GetNodeSelectorOverride() map[string]string
}

func GenerateK8sJobFactory(operatorName string, crd metav1.Object, jobName string, pvc *corev1.PersistentVolumeClaim, job EntityHandlerJobDefinition, entity Entity, basePath *string, additionalEnvVars []corev1.EnvVar, additionalLabels map[string]string) JobFactory {
	env := append([]corev1.EnvVar{
		{
			Name:  "ENTITY_TYPE",
			Value: entity.GetType(),
		},
		{
			Name:  "ENTITY_NAME",
			Value: entity.GetName(),
		},
		{
			Name:  "ENTITY_PATH",
			Value: RetrieveEntityPath(entity, basePath),
		},
	}, entity.GetAdditionalEnvVars()...)

	env = append(env, additionalEnvVars...)
	env = append(env, job.GetAdditionalEnvVars()...)

	// Filter Duplicates out of the Env list
	seen := map[string]bool{}
	filteredEnv := []corev1.EnvVar{}
	for _, e := range env {
		if seen[e.Name] {
			continue
		}
		seen[e.Name] = true
		filteredEnv = append(filteredEnv, e)
	}
	env = filteredEnv

	if entity.GetCheckpointName() != nil {
		env = append(env, corev1.EnvVar{
			Name:  "ENTITY_CHECKPOINT_NAME",
			Value: *entity.GetCheckpointName(),
		})
	}
	command := job.GetCustomCommand()
	if len(entity.GetCustomCommandOverride()) > 0 {
		command = entity.GetCustomCommandOverride()
	}
	arguments := job.GetArguments()
	if len(entity.GetArgumentsOverride()) > 0 {
		arguments = entity.GetArgumentsOverride()
	}
	podSecurityContext := job.GetPodSecurityContext()
	if entity.GetSecurityContextOverride() != nil {
		podSecurityContext = entity.GetSecurityContextOverride()
	}
	tolerations := job.GetTolerations()
	if len(entity.GetTolerationsOverride()) > 0 {
		tolerations = entity.GetTolerationsOverride()
	}
	nodeSelector := job.GetNodeSelector()
	if len(entity.GetNodeSelectorOverride()) > 0 {
		nodeSelector = entity.GetNodeSelectorOverride()
	}
	res := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: crd.GetNamespace(),
			Labels:    additionalLabels,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: GetLabels(crd, operatorName, additionalLabels),
					Annotations: map[string]string{
						"sidecar.istio.io/inject": "false",
					},
				},
				Spec: corev1.PodSpec{
					SecurityContext: podSecurityContext,
					Containers: []corev1.Container{
						{
							Name:            "main",
							Image:           job.GetImage(),
							ImagePullPolicy: corev1.PullAlways,
							Resources: corev1.ResourceRequirements{
								Limits: map[corev1.ResourceName]resource.Quantity{
									"cpu":    job.GetCPU(),
									"memory": job.GetMemory(),
								},
								Requests: map[corev1.ResourceName]resource.Quantity{
									"cpu":    job.GetCPU(),
									"memory": job.GetMemory(),
								},
							},
							Env: env,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "entity-volume",
									MountPath: PVCMountPath,
								},
							},
							Command: command,
							Args:    arguments,
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
					Volumes: []corev1.Volume{
						{
							Name: "entity-volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvc.Name,
								},
							},
						},
					},
					ImagePullSecrets: []corev1.LocalObjectReference{
						{
							Name: job.GetImagePullSecret(),
						},
					},
					Tolerations:  tolerations,
					NodeSelector: nodeSelector,
				},
			},
			BackoffLimit: ptr.To[int32](5),
		},
	}
	if job.GetGPUs() > 0 {
		res.Spec.Template.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = *resource.NewQuantity(int64(job.GetGPUs()), resource.DecimalExponent)
		res.Spec.Template.Spec.Containers[0].Resources.Requests["nvidia.com/gpu"] = *resource.NewQuantity(int64(job.GetGPUs()), resource.DecimalExponent)
	}
	return func() WrappedJob {
		return &JobWrapper{res}
	}
}

func RetrieveEntityPath(entity Entity, basePath *string) string {
	entityPath := PVCMountPath
	if basePath != nil && *basePath != "" {
		entityPath += "/" + *basePath
	}
	if entity.GetPath() == nil {
		entityPath += "/" + entity.GetType() + "/" + entity.GetName()
	} else {
		entityPath += "/" + *entity.GetPath()
	}
	return entityPath
}

func GetLabels(crd metav1.Object, operatorName string, additionalLabels map[string]string) map[string]string {
	res := map[string]string{
		"app":                          operatorName,
		"app.kubernetes.io/name":       crd.GetName(),
		"app.kubernetes.io/managed-by": operatorName,
	}
	for k, v := range additionalLabels {
		res[k] = v
	}
	return res
}

type JobWrapper struct {
	*batchv1.Job
}

func (j *JobWrapper) IsFail() bool {
	for _, condition := range j.Status.Conditions {
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func (j *JobWrapper) IsComplete() bool {
	for _, condition := range j.Status.Conditions {
		if condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func (j *JobWrapper) IsPending() bool {
	return j.Status.Active == 0
}

func (j *JobWrapper) FetchPendingStatusDetails(_ context.Context, _ client.Client) (*PendingStatusDetails, error) {
	return nil, nil
}

func (j *JobWrapper) GetWrapped() client.Object {
	return j.Job
}

type VolcanoJobWrapper struct {
	*vcbatchv1alpha1.Job
}

func (j *VolcanoJobWrapper) IsFail() bool {
	return j.Status.State.Phase == vcbatchv1alpha1.Failed
}

func (j *VolcanoJobWrapper) IsComplete() bool {
	return j.Status.State.Phase == vcbatchv1alpha1.Completed
}

func (j *VolcanoJobWrapper) IsPending() bool {
	return j.Status.State.Phase == vcbatchv1alpha1.Pending
}

func (j *VolcanoJobWrapper) GetWrapped() client.Object {
	return j.Job
}

func (j *VolcanoJobWrapper) FetchPendingStatusDetails(ctx context.Context, cl client.Client) (*PendingStatusDetails, error) {
	// currently volcano job's message is always empty.
	// however, once they do populate it in the future,
	// we can just rely on it directly
	if j.Status.State.Reason != "" || j.Status.State.Message != "" {
		return &PendingStatusDetails{
			Reason:  j.Status.State.Reason,
			Message: j.Status.State.Message,
		}, nil
	}
	// For now, the message is actually stored in the underlying podgroup
	// PodGroup has a status section that has a list of conditions, where a condition looks like
	//     type: Unschedulable
	//     status: "True"
	//     reason: NotEnoughResources
	//     message: '1/0 tasks in gang unschedulable: pod group is not ready, 1 minAvailable'
	podgroups := &vcschedulingv1beta1.PodGroupList{}
	err := cl.List(ctx, podgroups, client.InNamespace(j.Namespace), client.MatchingFields{VolcanoPodGroupOwnerKey: j.Name})
	if client.IgnoreNotFound(err) != nil {
		return nil, err
	}
	if len(podgroups.Items) == 0 || podgroups.Items[0].Status.Phase != vcschedulingv1beta1.PodGroupPending {
		// Between the time it took us to get the podgroup, the it's no longer in pending state
		// just return and let the next reconciliation handle the update
		return nil, nil
	}
	condition := podgroups.Items[0].Status.Conditions[0]
	return &PendingStatusDetails{
		Reason:  condition.Reason,
		Message: condition.Message,
	}, nil
}
