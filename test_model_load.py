import json, joblib, torch, sys, os
sys.path.insert(0, '.')
from modelling.anomaly.vae_lstm import VAELSTM

with open('models/vae-svdd/config.json') as f:
    cfg = json.load(f)
print(f'Config: threshold={cfg.get("best_threshold", "?")}, features={cfg.get("n_features", "?")}')

ckpt = torch.load('models/vae-svdd/vae_model.pt', map_location='cpu')
vae = VAELSTM(ckpt['input_dim'], ckpt['window_size'], ckpt['hidden_dim'], ckpt['latent_dim'])
vae.load_state_dict(ckpt['model_state_dict'])
vae.eval()
print(f'VAE: input_dim={ckpt["input_dim"]}, window={ckpt["window_size"]}, latent={ckpt["latent_dim"]}')

svdd = joblib.load('models/vae-svdd/svdd_model.pkl')
scaler = joblib.load('models/vae-svdd/scaler.pkl')
print(f'SVDD: {len(svdd.support_)} support vectors')
print(f'Scaler: mean.shape={scaler.mean_.shape}')
print('MODEL OK!')
