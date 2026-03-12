"""
Optimized model training with fast data loading.
Uses pre-packed .npz data and RAM caching for maximum throughput.

Usage:
    1. First run: python pack_data.py
    2. Then run: python model_optimized.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as tf
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import kornia as K
import pickle
import time
import argparse

from functions_optimized import (
    create_fast_dataloaders,
    AlexCNNEncoder,
    ViTCNNEncoder,
    FasterViTCNNEncoder,
    EfficientViTCNNEncoder,
    DecoderRNN,
    DecoderPool,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def onlineAugmentation(Batch_X):
    """Apply random flip augmentation to batch."""
    for index_b, batch_num in enumerate(Batch_X):
        random_H = random.choice([0, 1])
        random_V = random.choice([0, 1])
        for index_s, tensor in enumerate(batch_num):
            aug_tensor = tensor
            if random_H:
                aug_tensor = K.geometry.transform.hflip(aug_tensor)
            if random_V:
                aug_tensor = K.geometry.transform.vflip(aug_tensor)
            Batch_X[index_b][index_s] = aug_tensor
    return Batch_X


def train(
    log_interval,
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    user_index,
    grad_accum_steps=1,
):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0
    epoch_start_time = time.time()

    optimizer.zero_grad()
    for batch_idx, (X, y) in enumerate(train_loader):
        X = onlineAugmentation(X)
        X, y = X.to(device), y.to(device).view(
            -1,
        )
        N_count += X.size(0)

        output = rnn_decoder(cnn_encoder(X))
        loss = F.cross_entropy(output, y) / grad_accum_steps
        losses.append(loss.item() * grad_accum_steps)

        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(
            y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy()
        )
        scores.append(step_score)

        loss.backward()

        # Gradient accumulation: step optimizer every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % log_interval == 0:
            # Calculate ETA
            elapsed = time.time() - epoch_start_time
            batches_done = batch_idx + 1
            batches_total = len(train_loader)
            batches_remaining = batches_total - batches_done
            eta_seconds = (elapsed / batches_done) * batches_remaining
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_min}m {eta_sec}s"

            print(
                f"User_{user_index}   Train Epoch: {epoch + 1} "
                f"[{N_count}/{len(train_loader.dataset)} ({100. * batches_done / batches_total:.0f}%)]  "
                f"Loss: {losses[-1]:.6f}, Accu: {100 * step_score:.2f}%  "
                f"ETA: {eta_str}"
            )

    # Handle remaining gradients
    if (batch_idx + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return losses, scores


def validation(
    model, device, optimizer, test_loader, epoch, save_model_path, user_index
):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(
                -1,
            )
            output = rnn_decoder(cnn_encoder(X))
            loss = F.cross_entropy(output, y, reduction="sum")
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(
        all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy()
    )

    print(
        f"User_{user_index}\nTest set ({len(all_y)} samples): "
        f"Average loss: {test_loss:.4f}, Accuracy: {100 * test_score:.2f}%\n"
    )

    torch.save(
        cnn_encoder.state_dict(),
        os.path.join(save_model_path, f"cnn_encoder_epoch{epoch + 1}.pth"),
    )
    torch.save(
        rnn_decoder.state_dict(),
        os.path.join(save_model_path, f"rnn_decoder_epoch{epoch + 1}.pth"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(save_model_path, f"optimizer_epoch{epoch + 1}.pth"),
    )
    print(f"Epoch {epoch + 1} model saved!")

    return test_loss, test_score


def main():
    # ===== Argument parsing =====
    parser = argparse.ArgumentParser(description="Train SpeciFingers model")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: load few samples to verify code works",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume training from epoch N (loads checkpoint from ckpt_0)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["alexnet", "vit", "fastervit", "efficientvit"],
        default="alexnet",
        help="Encoder backbone: 'alexnet' (default), 'vit', or 'fastervit'",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["lstm", "pool"],
        default="lstm",
        help="Temporal decoder: 'lstm' (default) or 'pool' (global average pooling)",
    )
    args = parser.parse_args()

    # ===== Configuration =====
    PACKED_DATA_PATH = "packed_data"

    # Model architecture (optimized for RTX 4090 24GB)
    CNN_fc_hidden1, CNN_fc_hidden2 = 512, 256
    CNN_embed_dim = 256
    dropout_p = 0.3
    RNN_hidden_layers = 2
    RNN_hidden_nodes = 256
    RNN_FC_dim = 128

    # Training params (optimized for RTX 4090 24GB + 62GB RAM)
    k = 3  # number of classes (encoder output)
    epochs = 1 if args.test else 5
    # ViT uses more VRAM than AlexNet, adjust batch size accordingly
    # ViT uses more VRAM than AlexNet, adjust batch size accordingly
    if args.encoder == "vit":
        batch_size = 4 if args.test else 32  # ViT: reduced to 32 to prevent OOM
    elif args.encoder == "fastervit":
        batch_size = 4 if args.test else 48  # FasterViT-0: lighter than ViT-B
    elif args.encoder == "efficientvit":
        batch_size = 4 if args.test else 48  # EfficientViT-B1: lightweight
    else:
        batch_size = 4 if args.test else 96  # AlexNet: can go higher
    learning_rate = 1e-3
    log_interval = 1 if args.test else 10
    grad_accum_steps = (
        3 if args.encoder == "vit" else 2
    )  # Effective batch size control
    start_epoch = args.resume  # Resume from this epoch

    # Optimized loading params - Adjusted for 96GB RAM
    # Previously reduced for ViT/FasterViT to prevent OOM, now increased
    if args.encoder in ("vit", "fastervit", "efficientvit"):
        num_workers = 4  # Increased from 0 to utilize CPU
        max_cache_gb = 60  # Increased to 60GB (leaves ~36GB for system/models)
    else:
        num_workers = 0 if args.test else 8  # AlexNet: more workers OK
        max_cache_gb = 60  # Increased to 60GB
    cache_in_ram = False if args.test else True
    test_mode = args.test

    if test_mode:
        print("\n" + "=" * 50)
        print("TEST MODE: Loading minimal samples for verification")
        print("=" * 50 + "\n")

    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Train for each user (leave-one-out cross-validation)
    user_range = range(0, 1)  # Only train with user_0 as test set
    for user_index in user_range:
        print(f"\n{'='*50}")
        print(f"Training with User {user_index} as test set")
        print(f"{'='*50}")

        save_model_path = f"./ckpt_{user_index}"
        os.makedirs(save_model_path, exist_ok=True)

        # Create optimized dataloaders
        start_time = time.time()
        train_loader, valid_loader, le = create_fast_dataloaders(
            PACKED_DATA_PATH,
            user_index=user_index,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_in_ram=cache_in_ram,
            test_mode=test_mode,
            max_cache_gb=max_cache_gb,
        )
        load_time = time.time() - start_time
        print(f"Data loading setup: {load_time:.2f}s")

        # Create model
        if args.encoder == "vit":
            print("Using ViT-B/16 encoder")
            cnn_encoder = ViTCNNEncoder(
                fc_hidden1=CNN_fc_hidden1,
                fc_hidden2=CNN_fc_hidden2,
                drop_p=dropout_p,
                CNN_embed_dim=CNN_embed_dim,
            ).to(device)
        elif args.encoder == "fastervit":
            print("Using FasterViT-0-224 encoder")
            cnn_encoder = FasterViTCNNEncoder(
                fc_hidden1=CNN_fc_hidden1,
                fc_hidden2=CNN_fc_hidden2,
                drop_p=dropout_p,
                CNN_embed_dim=CNN_embed_dim,
                model_name="faster_vit_0_224",
            ).to(device)
        elif args.encoder == "efficientvit":
            print("Using EfficientViT-B1 encoder")
            cnn_encoder = EfficientViTCNNEncoder(
                fc_hidden1=CNN_fc_hidden1,
                fc_hidden2=CNN_fc_hidden2,
                drop_p=dropout_p,
                CNN_embed_dim=CNN_embed_dim,
            ).to(device)
        else:
            print("Using AlexNet encoder")
            cnn_encoder = AlexCNNEncoder(
                fc_hidden1=CNN_fc_hidden1,
                fc_hidden2=CNN_fc_hidden2,
                drop_p=dropout_p,
                CNN_embed_dim=CNN_embed_dim,
            ).to(device)

        if args.decoder == "pool":
            print("Using Global Average Pooling decoder")
            rnn_decoder = DecoderPool(
                CNN_embed_dim=CNN_embed_dim,
                h_FC_dim=RNN_FC_dim,
                drop_p=dropout_p,
                num_classes=k,
            ).to(device)
        else:
            print("Using LSTM decoder")
            rnn_decoder = DecoderRNN(
                CNN_embed_dim=CNN_embed_dim,
                h_RNN_layers=RNN_hidden_layers,
                h_RNN=RNN_hidden_nodes,
                h_FC_dim=RNN_FC_dim,
                drop_p=dropout_p,
                num_classes=k,
            ).to(device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            cnn_encoder = nn.DataParallel(cnn_encoder)
            rnn_decoder = nn.DataParallel(rnn_decoder)
            crnn_params = (
                list(cnn_encoder.module.fc1.parameters())
                + list(cnn_encoder.module.bn1.parameters())
                + list(cnn_encoder.module.fc2.parameters())
                + list(cnn_encoder.module.bn2.parameters())
                + list(cnn_encoder.module.fc3.parameters())
                + list(rnn_decoder.parameters())
            )
        else:
            print(f"Using {max(1, torch.cuda.device_count())} GPU(s)")
            crnn_params = (
                list(cnn_encoder.fc1.parameters())
                + list(cnn_encoder.bn1.parameters())
                + list(cnn_encoder.fc2.parameters())
                + list(cnn_encoder.bn2.parameters())
                + list(cnn_encoder.fc3.parameters())
                + list(rnn_decoder.parameters())
            )

        optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

        # Load checkpoint if resuming
        if start_epoch > 0:
            ckpt_path = f"./ckpt_{user_index}"
            cnn_ckpt = os.path.join(ckpt_path, f"cnn_encoder_epoch{start_epoch}.pth")
            rnn_ckpt = os.path.join(ckpt_path, f"rnn_decoder_epoch{start_epoch}.pth")
            opt_ckpt = os.path.join(ckpt_path, f"optimizer_epoch{start_epoch}.pth")
            if os.path.exists(cnn_ckpt) and os.path.exists(rnn_ckpt):
                print(f"Loading checkpoint from epoch {start_epoch}...")
                cnn_encoder.load_state_dict(torch.load(cnn_ckpt, map_location=device))
                rnn_decoder.load_state_dict(torch.load(rnn_ckpt, map_location=device))
                if os.path.exists(opt_ckpt):
                    optimizer.load_state_dict(torch.load(opt_ckpt, map_location=device))
                print(f"Resumed from epoch {start_epoch}")
            else:
                print(
                    f"Warning: Checkpoint for epoch {start_epoch} not found, starting from scratch"
                )
                start_epoch = 0

        # Training records
        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []

        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            train_losses, train_scores = train(
                log_interval,
                [cnn_encoder, rnn_decoder],
                device,
                train_loader,
                optimizer,
                epoch,
                user_index,
                grad_accum_steps,
            )
            epoch_test_loss, epoch_test_score = validation(
                [cnn_encoder, rnn_decoder],
                device,
                optimizer,
                valid_loader,
                epoch,
                save_model_path,
                user_index,
            )

            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.2f}s")

            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)

            # Save results
            np.save("./CRNN_epoch_training_losses.npy", np.array(epoch_train_losses))
            np.save("./CRNN_epoch_training_scores.npy", np.array(epoch_train_scores))
            np.save("./CRNN_epoch_test_loss.npy", np.array(epoch_test_losses))
            np.save("./CRNN_epoch_test_score.npy", np.array(epoch_test_scores))


if __name__ == "__main__":
    main()
