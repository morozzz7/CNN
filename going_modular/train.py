from model_builder import HybridModel
import pickle
import torch
import argparse
import data_setup, engine, utils
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_size', type=int, default=512)
    args = parser.parse_args()

    train_dir = 'part_cars/train'
    test_dir = 'part_cars/test'

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transforms=data_transform_train,
    test_transforms=data_transform_test,
    batch_size=args.batch_size
)

    model = HybridModel(hidden_units=args.hidden_units,
    num_transformer_layers=args.num_transformer_layers,
    embedding_dim=args.embedding_dim, 
    num_heads=args.num_heads,
    mlp_size=args.mlp_size).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=args.epochs,
                device=device
    )

    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    utils.save_model(
        model=model,
        target_dir='models',
        model_name='first_try.pth'
    )

if __name__ == '__main__':
    main()


