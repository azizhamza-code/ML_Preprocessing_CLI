import argparse
import pytorch_lightning as pl
from typing import Tuple ,Collection,Dict,Optional, Union
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from pathlib import Path


def load_and_orint_info(data_module_class) -> None:
    """ Load digit number and print """
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print('h')
    print(dataset)


BATCH_SIZE = 128
NUM_WORKERS = 0

class BaseDataModule(pl.LightningDataModule):
    """base data module"""


    def __init__(self,args : argparse.Namespace=None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get('batch_size',BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

        self.on_gpu = False

        self.dims :Tuple[int,...]
        self.output_dims:Tuple[int,...]
        self.mapping:Collection
        self.data_train:Union[Dataset,ConcatDataset]   
        self.data_val: Union[Dataset, ConcatDataset]
        self.data_test: Union[Dataset, ConcatDataset]  

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parent.parent.parent/"data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",type = int , default=BATCH_SIZE,help = 'Number of exemple to operate per forward step '

        )
        parser.add_argument(
            "--num_workers",type = int, default=NUM_WORKERS , help = 'Number of additional processes to load data'
        )

    def config(self):
        """return important setting of the datasets , wich will b passsed to instantiate  model"""
        return {"input_dims":self.dims,"output_dims":self.output_dims,"mapping":self.mapping}

    
    def prepare_data(self,*args,**kwargs) -> None:
        """
        use this methode to do things that might write to disk or that need to be done only from a single GPU 
        in destributed sitting(so don't set state)
        
        """
    
    def setup(self,stage:Optional[str]=None)->None:
        """
        split into train,val,test,and set dims.
        Should assing 'torch Dataset' to self.data_train ,self.data_val and optionally self.data_val

        """
    def train_dataloader(self):

        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers = self.num_workers,  
            pin_memory=self.on_gpu,          
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size = self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle = False,
            batch_size = self.batch_size,
            numworkers = self.num_worksers,
            pin_memory = self.on_gpu

        )
        

    




