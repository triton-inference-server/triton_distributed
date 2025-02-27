package controller_common

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	PVCCreatedState = "PVCCreated"
	PVCCreated      = "PVC_CREATED"
	PVCMountPath    = "/pvc"
)

type PVC interface {
	GetName() *string
	GetCreate() *bool
	GetStorageClass() string
	GetVolumeAccessMode() corev1.PersistentVolumeAccessMode
	GetSize() resource.Quantity
}

type customResource interface {
	client.Object
	StatusConditionsProvider
	StatusConditionsMutator
}

func ReconcilePVC(ctx context.Context, crd customResource, cl client.Client, pvcConfig PVC) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)
	pvc := &corev1.PersistentVolumeClaim{}
	pvcName := types.NamespacedName{Name: GetPvcName(crd, pvcConfig.GetName()), Namespace: crd.GetNamespace()}
	err := cl.Get(ctx, pvcName, pvc)
	if err != nil && client.IgnoreNotFound(err) != nil {
		logger.Error(err, "Unable to retrieve PVC", "crd", crd.GetName())
		return nil, err
	}

	// If PVC does not exist, create a new one
	if err != nil {
		if pvcConfig.GetCreate() == nil || !*pvcConfig.GetCreate() {
			logger.Error(err, "Unknown PVC", "pvc", pvc.Name)
			return nil, err
		}
		pvc = ConstructPVC(crd, pvcConfig)
		if err := controllerutil.SetControllerReference(crd, pvc, cl.Scheme()); err != nil {
			logger.Error(err, "Failed to set controller reference", "pvc", pvc.Name)
			return nil, err
		}
		err = cl.Create(ctx, pvc)
		if err != nil {
			logger.Error(err, "Failed to create pvc", "pvc", pvc.Name)
			return nil, err
		}
		logger.Info("PVC created", "pvc", pvcName)
		UpdateCondition(crd, PVCCreated, metav1.ConditionTrue, PVCCreatedState, "The PVC has been created")
	}
	return pvc, nil
}

func ConstructPVC(crd metav1.Object, pvcConfig PVC) *corev1.PersistentVolumeClaim {
	storageClassName := pvcConfig.GetStorageClass()
	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GetPvcName(crd, pvcConfig.GetName()),
			Namespace: crd.GetNamespace(),
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{pvcConfig.GetVolumeAccessMode()},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: pvcConfig.GetSize(),
				},
			},
			StorageClassName: &storageClassName,
		},
	}
}

func GetPvcName(crd metav1.Object, defaultName *string) string {
	if defaultName != nil {
		return *defaultName
	}
	return crd.GetName()
}
