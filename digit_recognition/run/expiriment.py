import argparse
import importlib

import numpy as np
from pytorch_lightning.callbacks import model_checkpoint
import torch 
import pytorch_lightning as pl
import wandb

import sys



from digit_recognition import lit_model


np.random.seed(42)
torch.manual_seed(42)

def _import_class(module_and_class_name : str ) -> type:
    """
    import class from a module , 'digit_recognition.models.MLP

    """

    module_name , class_name =module_and_class_name.rsplit(".",1)
    module = importlib.import_module(module_name)
    class_ =  getattr(module,class_name)
    return class_


def _setup_parser():
    """set up python's argument parser with data , model , trainer and other"""
    parser = argparse.ArgumentParser(Add_help = False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help = False , parents=[trainer_parser])

    parser.add_argument("--data_class",type = str , defaults ="MNIST")
    parser.add_argument("--model_class",type= str , default = "MLP")
    parser.add_argument("load_checkpoint",type = str , default=None)


    temp_args , _ = parser.parser_known_args()
    data_class = _import_class(f"digit_recognition.data.{temp_args.data_class}")
    model_class = _import_class(f"digit_recognition.models.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)


    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)


    parser.add_argument("--help", "-h", action="help") 
    return parser


def main():
    """
    run an experiment 
    sample command:

    python/run.py --max_epochs=3 --gpus='0, num_workers = 20 --model_calss=MLP  data_class =MNIST

    """


    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"digit_recognition.data.{args.data_class}")
    model_class = _import_class(f"digit_recognition.model.{args.model_class}")
    data = data_class(args)
    model = model_class(args=args,data_config= data.config())

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_model.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_chekpoint(args.load_chekpoint , args = args , model = model)

    else:
        lit_model = lit_model_class(args=args,model=model)

    logger = pl.loggers.TensorBoardLogger("run/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor = "val_loss",mode = "min", patience = 10)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epch:03}-{val_loss:.3f}-{val_cer:.3f}" , monitor = "val_loss" , mode = "min"
    )

    callbacks = [early_stopping_callback,model_checkpoint_callback]

    args.weights_summary = "full"

    trainer = pl.Trainer.from_argparse_args(args,callbacks=callbacks,logger=logger,weights_save_path = "training/logs")

    trainer.tune(lit_model,datamodule=data)

    trainer.fit(lit_model,datamodel = data)

    trainer.test(lit_model,datamodule=data)

if __name__ == "__main__":
    main()




