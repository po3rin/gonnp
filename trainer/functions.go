package trainer

import (
	"reflect"

	"github.com/po3rin/gonnp/entity"
	"gonum.org/v1/gonum/mat"
)

func rmDuplicate(params []entity.Param, grads []entity.Grad) ([]entity.Param, []entity.Grad) {
	for {
		var findFlg bool
		L := len(params)
		for i := 0; i < L; i++ {
			for j := i + 1; j < L; j++ {
				if reflect.DeepEqual(params[i].Weight, params[j].Weight) {
					r, c := grads[i].Weight.Dims()
					d := mat.NewDense(r, c, nil)
					d.Add(grads[i].Weight, grads[j].Weight)
					grads[i].Weight = d
					findFlg = true
					params = append(params[:j], params[j+1:]...)
					grads = append(grads[:j], grads[j+1:]...)
				}
				if findFlg {
					break
				}
			}
			if findFlg {
				break
			}
		}
		if !findFlg {
			break
		}
	}
	return params, grads
}
