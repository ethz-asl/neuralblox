import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
import shutil
from src import layers

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = config.get_dataset('train', cfg)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['training']['batch_size'], num_workers=cfg['training']['n_workers'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Model
model, input_crop_size, query_crop_size, grid_reso = config.get_model(cfg, device=device, dataset=train_dataset)

# Model for merging
model_merging = layers.Conv3D_one_input().to(device)

if model_merging is not None:
    # Freeze ConvONet parameters
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = optim.Adam(list(model.parameters()) + list(model_merging.parameters()), lr=1e-4)
    trainer = config.get_trainer_sequence(model, model_merging, optimizer, cfg, device=device)
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io_merging = CheckpointIO(out_dir, model=model_merging, optimizer = optimizer)
try:
    checkpoint_io.load(os.path.join(os.getcwd(), cfg['training']['backbone_file']))
    load_dict = checkpoint_io_merging.load('model_merging.pt')
except FileExistsError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', -1)

metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
nparameters_merging = sum(p.numel() for p in model_merging.parameters())
print('Total number of parameters: %d' % nparameters)
print('Total number of parameters in merging model: %d' % nparameters_merging)

print('output path: ', cfg['training']['out_dir'])

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss = trainer.train_sequence_window(batch, input_crop_size, query_crop_size, grid_reso, window=cfg['training']['batch_size'])
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io_merging.save('model_merging.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io_merging.save('model_merging_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io_merging.save('model_merging.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
