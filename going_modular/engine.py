import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    model.train()
    train_loss_sum, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        log = model(X)
        loss = loss_fn(log, y)
        y_pred = torch.softmax(log, dim=1).argmax(dim=1)
        train_loss_sum += loss.item() * X.size(0)
        train_acc += (y_pred == y).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    avg_train_loss = train_loss_sum / len(dataloader.dataset)
    avg_train_acc = train_acc / len(dataloader.dataset) * 100
    return avg_train_loss, avg_train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
):
    model.eval()
    test_loss_sum, test_acc = 0, 0
    with torch.inference_mode():
        model.eval()
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            log = model(X)
            test_loss_sum += loss_fn(log, y).item() * X.size(0)
            y_pred = torch.softmax(log, dim=1).argmax(dim=1)
            test_acc += (y_pred == y).sum().item()
    avg_test_loss = test_loss_sum / len(dataloader.dataset)
    avg_test_acc = test_acc / len(dataloader.dataset) * 100
    return avg_test_loss, avg_test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          ):
    writer = SummaryWriter()

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device)
        test_loss, test_acc = test_step(model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:

            writer.add_scalars(main_tag="Loss", 
                                tag_scalar_dict={"train_loss": train_loss,
                                                    "test_loss": test_loss},
                                global_step=epoch)

            writer.add_scalars(main_tag="Accuracy", 
                                tag_scalar_dict={"train_acc": train_acc,
                                                    "test_acc": test_acc}, 
                                global_step=epoch)
            writer.close()
        else:
            pass
    return results



