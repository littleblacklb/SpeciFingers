import numpy as np

# Load training results
test_loss = np.load("CRNN_epoch_test_loss.npy")
test_score = np.load("CRNN_epoch_test_score.npy")
training_losses = np.load("CRNN_epoch_training_losses.npy")
training_scores = np.load("CRNN_epoch_training_scores.npy")

# Export to text file
with open("training_results.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("CRNN Training Results\n")
    f.write("=" * 80 + "\n\n")

    f.write("Training Losses:\n")
    f.write("-" * 80 + "\n")
    for i, loss in enumerate(training_losses, 1):
        f.write(f"Epoch {i}: {loss}\n")

    f.write("\n" + "=" * 80 + "\n\n")
    f.write("Training Scores:\n")
    f.write("-" * 80 + "\n")
    for i, score in enumerate(training_scores, 1):
        f.write(f"Epoch {i}: {score}\n")

    f.write("\n" + "=" * 80 + "\n\n")
    f.write("Test Losses:\n")
    f.write("-" * 80 + "\n")
    for i, loss in enumerate(test_loss, 1):
        f.write(f"Epoch {i}: {loss}\n")

    f.write("\n" + "=" * 80 + "\n\n")
    f.write("Test Scores:\n")
    f.write("-" * 80 + "\n")
    for i, score in enumerate(test_score, 1):
        f.write(f"Epoch {i}: {score}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("Summary Statistics\n")
    f.write("=" * 80 + "\n")
    # Aggregate per-epoch (training_losses/scores are 2D: epochs x batches)
    epoch_avg_losses = np.array(
        [np.mean(epoch_losses) for epoch_losses in training_losses]
    )
    epoch_avg_scores = np.array(
        [np.mean(epoch_scores) for epoch_scores in training_scores]
    )
    f.write(
        f"Best Avg Training Loss: {epoch_avg_losses.min():.6f} (Epoch {epoch_avg_losses.argmin() + 1})\n"
    )
    f.write(
        f"Best Avg Training Score: {epoch_avg_scores.max():.6f} (Epoch {epoch_avg_scores.argmax() + 1})\n"
    )
    f.write(f"Best Test Loss: {test_loss.min():.6f} (Epoch {test_loss.argmin() + 1})\n")
    f.write(
        f"Best Test Score: {test_score.max():.6f} (Epoch {test_score.argmax() + 1})\n"
    )

print("Training results exported to training_results.txt")
