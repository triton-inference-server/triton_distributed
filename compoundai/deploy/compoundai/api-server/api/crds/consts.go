package crds

var ApiVersion string = "nvidia.com/v1alpha1"

type CustomResourceType string

const (
	CompoundAINimRequest    CustomResourceType = "CompoundAINimRequest"
	CompoundAINimDeployment CustomResourceType = "CompoundAINimDeployment"
)
