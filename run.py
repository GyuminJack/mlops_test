import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

# https://github.com/Lightning-AI/lightning/issues/10182
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class CtrPredictionModel(pl.LightningModule):
    def __init__(self, user_len, item_len):
        super().__init__()
        emb_dim = 25
        self.user_emb = nn.Embedding(num_embeddings=user_len, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=item_len, embedding_dim=emb_dim)
        self.projection = nn.Linear(in_features=emb_dim*2, out_features=1)

    def forward(self, user, item):
        user_emb = self.user_emb(torch.LongTensor(user))
        item_emb = self.item_emb(torch.LongTensor(item))
        ctr = self.projection(torch.concat([user_emb, item_emb], dim=-1)).flatten()
        return ctr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, z = train_batch
        z_hat = self.forward(x, y)
        loss = F.mse_loss(z, z_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, z = val_batch
        z_hat = self.forward(x, y).flatten()
        loss = F.mse_loss(z, z_hat)
        self.log('val_loss', loss)

if __name__ == "__main__":
    # Load Data
    local_data_path = "./data/movielens.sample"

    movielens_df = pd.read_csv(local_data_path, usecols=['user_id', 'movie_id', 'rating'])
    sparse_features = ['user_id','movie_id']
    for feat in sparse_features:
        lbe = LabelEncoder()
        movielens_df[feat] = lbe.fit_transform(movielens_df[feat])
    movielens_array = movielens_df.to_numpy()

    # Data To Tensor
    # Using torch custom dataset is useful (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    user = torch.LongTensor(movielens_array[:, 0])
    item = torch.LongTensor(movielens_array[:, 1])
    rating = torch.FloatTensor(movielens_array[:, 2])
    dataset = TensorDataset(user, item, rating)

    # Make Dataloader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = None

    # Make Model
    n_user = movielens_df['user_id'].nunique()
    n_item = movielens_df['movie_id'].nunique()
    ctr_model = CtrPredictionModel(n_user, n_item)

    # training
    trainer = pl.Trainer(max_epochs=10, auto_select_gpus=True, log_every_n_steps=50)
    trainer.fit(ctr_model, train_loader, val_loader)