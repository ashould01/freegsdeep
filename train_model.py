import os
from datetime import datetime
import torch.multiprocessing as mp
from freegsdeep.train import Trainer_resi, Trainer_bdry, Trainer_bdry_deeponet
from freegsdeep.typing import Optional

def main(
    save_name: str, device: Optional[str], batch_size: int,
    resi_or_bdry: str, data_load_path: str, 
    epoch: int, load_model: Optional[dict] = None
    ) -> None:
    if resi_or_bdry == 'resi':
        trainer = Trainer_resi(
            Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nR=65, nZ=65, 
            len_data=100, batch_size=batch_size, 
            epoch=epoch, hard=True,
            data_load_path=data_load_path, device=device
            )
    elif resi_or_bdry == 'bdry_deeponet':
        trainer = Trainer_bdry_deeponet(
            Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nR=65, nZ=65, 
            len_data=100, batch_size=batch_size, 
            epoch=epoch, data_load_path=data_load_path, device=device
            )
    elif resi_or_bdry == 'bdry':
        trainer = Trainer_bdry(
            Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nR=65, nZ=65, 
            len_data=100, batch_size=batch_size, num_resi_pt=150,
            epoch=epoch, data_load_path=data_load_path, device=device
            )
    else:
        raise NotImplementedError
    now = datetime.now().strftime("%y%m%d_%H%M%S")
    save_name = save_name + '_' + now
    os.makedirs(os.path.join('logs', save_name, 'figures'), exist_ok=True)
    os.makedirs(os.path.join('logs', save_name, 'model'), exist_ok=True)

    trainer.train(save_name, load_model)

if __name__ == '__main__':
    # main(
    #     'plain_lbfgs_large_dataset_hard_bdry', device='cuda:0',
    #     batch_size=25, resi_or_bdry='resi', data_load_path='data_100_2',
    #     epoch_adam=0, epoch_lbfgs=50,
    #     load_model={'epoch': 0, 'name': 'plain_lbfgs_large_dataset_hard_bdry_251120_210834'}
    # )
    main(
        'deeponet_bdry', device='cuda:0',
        batch_size=75, resi_or_bdry='bdry_deeponet', data_load_path='data_100_2',
        epoch=50, 
        # load_model={'epoch': 22, 'name': 'deeponet_251224_111859'}
        )