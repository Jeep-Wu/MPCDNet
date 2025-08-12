import multiprocessing
import os

def worker(cuda_visible_device, fold, barrier):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_device)
    from nnunetv2.run.run_training import run_training_entry

    run_training_entry(fold, barrier)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPCDNet Training Script')
    parser.add_argument('-c', '--cuda', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--fold', type=str, required=False, default=None,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')

    args, unknow = parser.parse_known_args()
    cuda_visible_device_list = [int(i) for i in args.cuda.split(',')]

    barrier = multiprocessing.Barrier(len(cuda_visible_device_list))
    processes = [
        multiprocessing.Process(target=worker, args=(cuda_visible_device, args.fold if args.fold else i, barrier)) for i, cuda_visible_device in enumerate(cuda_visible_device_list)
    ]

    # start all processes
    for p in processes:
        p.start()

    # wait for all processes to finish
    for p in processes:
        p.join()