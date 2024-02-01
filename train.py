from tqdm import tqdm
import numpy as np
import torch
from dataloaders import get_loader
from SoundingEarth.lib import get_optimizer, get_model, get_loss_function, Metrics
from SoundingEarth.lib.models import FullModelWrapper
#from SoundingEarth.lib.evaluation import calculate_embeddings
from SoundingEarth.config import cfg,state

def full_forward(model, key, img, snd, snd_split, points, metrics):
    img = img.to(dev)
    snd = snd.to(dev)
    points = points.to(dev)

    Z_img = model.img_encoder(img)
    
    if cfg.SoundEncoder == 'VisualTransformersImage':
      Z_snd = model.snd_encoder(Z_snd)
    else:
      Z_snd = model.snd_encoder(snd, snd_split)

    loss = model.loss_function(Z_img, Z_snd, points)
    res = {"Loss":loss}
    return res

def train_epoch(model, data_loader, metrics):
    #state.Epoch += 1
    model.train(True)
    metrics.reset()
    # torch.autograd.set_detect_anomaly(True)
    for iteration, data in enumerate(tqdm(data_loader)):
        res = full_forward(model, *data, metrics)
        opt.zero_grad()
        print("actual loss: ",res["Loss"])
        res['Loss'].backward()
        opt.step()
        # state.BoardIdx += data[0].shape[0]

    # metrics_vals, metrics_hist = metrics.evaluate()
    # logstr = ', '.join(f'{k}: {v:2f}' for k, v in metrics_vals.items())
    # print(f'Epoch {state.Epoch:03d} Trn: {metrics_vals}')
    # m = {f'trn/{met}': val for met, val in metrics_vals.items()}
    # m['_epoch'] = state.Epoch

@torch.no_grad()
def val_epoch(model, data_loader, metrics):
    model.train(False)
    metrics.reset()
    for iteration, data in enumerate(data_loader):
        res = full_forward(model, *data, metrics)

    metrics_vals, metrics_hist = metrics.evaluate()
    logstr = ', '.join(f'{k}: {v:2f}' for k, v in metrics_vals.items())
    print(f'Epoch {state.Epoch:03d} Val: {metrics_vals}')
    m = {f'val/{met}': val for met, val in metrics_vals.items()}
    m['_epoch'] = state.Epoch
  
    # Save model Checkpoint
    if state.Epoch % 20 == 0:
        torch.save(model.state_dict(), checkpoints / f'{state.Epoch:02d}.pt')
    torch.save(model.state_dict(), checkpoints / 'latest.pt')

    if metrics_vals['Loss'] < state.BestLoss:
        print(f'Saving Checkpoint at Epoch {state.Epoch} as best one yet!')
        state.BestLoss = metrics_vals['Loss']
        state.BestEpoch = state.Epoch
        torch.save(model.state_dict(), checkpoints / f'best.pt')

    return state.BestEpoch + cfg.EarlyStopping < state.Epoch

@torch.no_grad()
def calculate_embeddings(model, loader_type, device):
    loader = get_loader(mode=loader_type, batch_size=32, num_workers=2, max_samples=50)
    keys, Z_snd, Z_img = [], [], []
    tf = model.loss_function.distance_transform
    for key, img, snd, snd_split, distance in tqdm(loader, 'Embeddings'):
        snd = snd.to(device)
        img = img.to(device)

        Z_snd.append(tf(model.snd_encoder(snd, snd_split)))
        Z_img.append(tf(model.img_encoder(img)))
        keys.append(key)

    keys = np.concatenate(keys)
    Z_img = torch.cat(Z_img)
    Z_snd = torch.cat(Z_snd)
    print('Z_img shape: ',Z_img.shape)
    print('Z_snd shape:', Z_snd.shape)

    return keys, Z_img, Z_snd

if __name__ == "__main__":
    dev = torch.device('cpu')
    img_encoder   = get_model(cfg.ImageEncoder, reducer=cfg.ImageReducer,
        input_dim=3, output_dim=cfg.LatentDim, final_pool=False
    )
    snd_encoder   = get_model(cfg.SoundEncoder, reducer=cfg.SoundReducer,
        input_dim=1, output_dim=cfg.LatentDim, final_pool=True
    )

    loss_function = get_loss_function(cfg.LossFunction)(*cfg.LossArg)

    model = FullModelWrapper(img_encoder, snd_encoder, loss_function).to(dev)

    opt = get_optimizer(cfg.Optimizer.Name)(model.parameters(), lr=cfg.Optimizer.LearningRate)

    train_data = get_loader(32, num_workers=2, mode='toy', max_samples=100)
    val_data = get_loader(32, num_workers=2, mode='val', max_samples=100)

    metrics = Metrics()
    for epoch in range(3):
        print(f'Starting epoch "{epoch}"')
        # k,img_emb,snd_emb = calculate_embeddings(model, 'toy', dev)
        # print('img_emb shape',img_emb[0].shape)
        # print('snd_emb shape',snd_emb[0].shape)

        # print("loss_func: ",type(loss_function))

        train_epoch(model, train_data, metrics)
        # stop_early = val_epoch(model, val_data, metrics)
        # if stop_early:
        #     print(f'Stopping Early after {cfg.EarlyStopping} epochs without improvement')
        #     break

