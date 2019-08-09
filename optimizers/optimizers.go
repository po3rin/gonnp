// Package optimizers updates prams (ex. weight, bias ...) using various algorism.
package optimizers

import "github.com/po3rin/gonnp/entity"

// Optimizer updates prams.
type Optimizer interface {
	Update(params []entity.Param, grads []entity.Grad) []entity.Param
}
