from pinn.train import PINNTrainer

if __name__ == '__main__':
    # PINNTrainer(
    #     'SolovevTriangularity', 4_000, 1_000, 0.1, 1.45, -1.0, 1.0, device='cuda:5',
    #     adam_epoch=100, lbfgs_epoch=100
    #     ).train('debug')
    PINNTrainer(
        dataset_type='IterationDataset', nR=65, nZ=65, 
        Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, device='cuda:1',
        adam_epoch=0, lbfgs_epoch=500
        ).train('no_restriction_1')