package system

import (
	"context"

	corev1 "k8s.io/api/core/v1"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/pkg/compoundai/consts"
)

func GetNetworkConfigConfigMap(ctx context.Context, configmapGetter func(ctx context.Context, namespace, name string) (*corev1.ConfigMap, error)) (configMap *corev1.ConfigMap, err error) {
	configMap, err = configmapGetter(ctx, GetNamespace(), consts.KubeConfigMapNameNetworkConfig)
	return
}
