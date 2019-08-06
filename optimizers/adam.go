package optimizers

import "github.com/po3rin/gonlp/entity"

// SDG has setting for Stochastic Gradient Descent.
type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
}

func InitAdam(lr, beta1, beta2 float64) *Adam {
	return &Adam{
		LR:    lr,
		Beta1: beta1,
		Beta2: beta2,
	}
}

func (a *Adam) Update(params []entity.Param, grads []entity.Grad) []entity.Param {
	return nil
}
