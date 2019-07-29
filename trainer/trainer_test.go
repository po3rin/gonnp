package trainer_test

import (
	"testing"

	"github.com/po3rin/gonlp/nn"
	"github.com/po3rin/gonlp/optimizers"
	"github.com/po3rin/gonlp/trainer"
)

func Test_Fit(t *testing.T) {
	n := nn.NewTwoLayerNet(2, 30, 10)
	o := optimizers.InitSDG(1)
	trainer := trainer.InitTrainer(n, o)

	_ = trainer
}
