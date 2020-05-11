from ignite.engine import Engine
import torch

def create_clustering_engine(model, optimizer, loss, device):

    model.to(device)

    def _update(engine, batch):
        data, label = batch
        print('--- data ---')
        print(data)
        data.to(device)
        label.to(device)
        model.train()

        optimizer.zero_grad()
        output = model(data)
        _loss = loss(output, label)
        _loss.backward()
        optimizer.step()

        return _loss.item()

    return Engine(_update)