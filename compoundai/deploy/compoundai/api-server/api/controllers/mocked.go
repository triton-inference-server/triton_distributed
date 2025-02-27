package controllers

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
)

type mockedController struct{}

var MockedController = mockedController{}

var mockedUid = "nvid1a11-1234-5678-9abc-def012345678"

func (c *mockedController) GetDefaultUser(ctx *gin.Context) {
	user := c.getDefaultUserHelper()
	ctx.JSON(200, user)
}

func (c *mockedController) getDefaultUserHelper() *schemas.UserSchema {
	return &schemas.UserSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name: "nvidia-user",
		},
		FirstName: "Compound",
		LastName:  "AI",
		Email:     "compoundai@nvidia.com",
	}
}

func (c *mockedController) GetDefaultOrg(ctx *gin.Context) {
	defaultOrg := schemas.OrganizationSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name:         "nvidia-org",
			ResourceType: schemas.ResourceTypeOrganization,
			Labels:       make([]string, 0),
		},
		Description: "nvidia-org-desc",
	}

	ctx.JSON(200, defaultOrg)
}
