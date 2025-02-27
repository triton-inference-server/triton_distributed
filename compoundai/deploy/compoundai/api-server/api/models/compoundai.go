package models

import (
	"time"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
)

type CompoundAINimVersion struct {
	BaseModel
	Version                   string                               `json:"version"`
	Description               string                               `json:"description"`
	FilePath                  string                               `json:"file_path"`
	ImageBuildStatusSyncingAt *time.Time                           `json:"image_build_status_syncing_at"`
	ImageBuildStatusUpdatedAt *time.Time                           `json:"image_build_status_updated_at"`
	UploadStartedAt           *time.Time                           `json:"upload_started_at"`
	UploadFinishedAt          *time.Time                           `json:"upload_finished_at"`
	UploadFinishedReason      string                               `json:"upload_finished_reason"`
	Manifest                  *schemas.CompoundAINimManifestSchema `json:"manifest" type:"jsonb"`
	BuildAt                   time.Time                            `json:"build_at"`
}

func (b *CompoundAINimVersion) GetName() string {
	return b.Version
}

func (b *CompoundAINimVersion) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeCompoundAINimArtifactVersion
}
