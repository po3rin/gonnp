package optimizers

import "github.com/po3rin/gonlp/entity"

type Optimizer interface {
	Update(params []entity.Param, grads []entity.Grad) []entity.Param
}
