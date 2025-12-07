# Training on Subsampled Datasets (10% & 25%)

This guide explains how to train the GraphFlix model on the 10% and 25% subsamples of the MovieLens 1M dataset. These scripts are the best starting point for running training, as they are faster than training on the full dataset and provide a good baseline for model performance.

## Quick Start

To start a training run, simply execute the desired script from the project's root directory.

### Train on 10% of the Data
```bash
./train_10pct_fixed.sh
```

### Train on 25% of the Data
```bash
./train_25pct_proper.sh
```

The scripts will prompt you before starting. Press `y` to continue.

## What the Scripts Do

Both scripts follow the same automated, multi-step process:

1.  **Verify Data**: They first check if the required subsampled dataset and its corresponding train/val/test splits exist in `data/processed/ml1m_10pct/` or `data/processed/ml1m_25pct/`. If not, they will attempt to create them.

2.  **Build Graph & Features**: They verify that the graph file (`graph_pyg.pt`), metadata embeddings (`phi_matrix.pt`), and user profiles (`user_profiles.pt`) have been precomputed. If not, they will run the necessary scripts to generate them.

3.  **Verify Data Leakage Prevention**: A crucial step that runs a test (`test_data_leakage.py`) to ensure that the negative sampling process during training does not accidentally include positive items from the validation or test sets. This guarantees a fair evaluation.

4.  **Start Training**: The main training script `scripts/train_graphflix.py` is executed. All output is logged to both the console and a file.
    - **Output Location**: A new directory is created for each run in `runs/`, named with a timestamp (e.g., `runs/graphflix_10pct_20251206_103000/`).
    - **Inside this directory, you will find**:
        - `training.log`: A complete log of the console output.
        - `config.yaml`: A copy of the configuration used for the run.
        - `tensorboard/`: Data for visualizing metrics.
        - `best.pt`, `final.pt`: Saved model checkpoints.

5.  **Evaluate Model**: After training is complete, the script automatically runs `evaluate_proper_final.py` on the `best.pt` checkpoint. This performs a rigorous 1-vs-100 evaluation on the test set.

## Configuration

The training process is controlled by a central configuration file. While the shell scripts handle the workflow, the model's hyperparameters are defined in this YAML file.

-   **Location**: `configs/model/graphflix_full.yaml`

### Important Parameters to Know

You can edit this file to experiment with different settings. Here are the most important ones:

| Parameter | Section | Description | Default | Why Change It? |
| :--- | :--- | :--- | :--- | :--- |
| `lr` | `training` | **Learning Rate**: How large the steps are during optimization. | `1e-3` | The most impactful parameter. If loss stalls, you may need to adjust this. |
| `batch_size` | `training` | **Batch Size**: Number of samples processed in one step. | `16` | Decrease this (e.g., to `8` or `4`) if you get a `CUDA out of memory` error. |
| `epochs` | `training` | **Epochs**: Number of full passes over the training data. | `20` | Increase for longer training, decrease for quicker tests. |
| `scheduler` | `training` | **Scheduler**: How the learning rate changes over time. | `plateau` | `plateau` is adaptive; `cosine` follows a fixed schedule. |
| `eval_every` | `eval` | **Eval Frequency**: How many epochs between validation checks. | `1` | Set to `1` to enable the `plateau` scheduler to work effectively. |
| `patience` | `training` | **Patience**: (For `plateau` scheduler) Epochs to wait for improvement before reducing LR. | `2` | Lower patience makes the scheduler react faster to stalled training. |

## Evaluation Protocol

The final step of the script is a fair and rigorous evaluation:

-   **Method**: 1-vs-100 negative sampling.
-   **Process**: For each user in the test set, the model ranks the one "true" positive movie against 99 "negative" movies that the user has not interacted with.
-   **Metrics**: The script calculates `Recall@10`, `Recall@20`, `NDCG@10`, and `NDCG@20`. These metrics measure how well the model places the true positive item within the top K recommendations.
-   **Output**: The results are printed to the console and saved in a file named `test_results_proper.txt` inside the run directory (e.g., `runs/graphflix_10pct_.../test_results_proper.txt`).

## What a Successful Execution Looks Like

1.  **Initial Checks Pass**: You will see green "✅" checkmarks as the script verifies that the data and graph files exist.
2.  **Training Starts**: You will see the model architecture, optimizer, and scheduler details printed, followed by the training progress for each epoch.
    ```
    ================================================================================
    Starting Training
    ================================================================================
    Epochs: 20
    Batch size: 16
    Eval every: 1 epochs
    ================================================================================

    Epoch 1/20
    --------------------------------------------------------------------------------
      Batch 100/5844 | Loss: 0.6988 | Avg Loss: 1.3633 | Beta: 0.9975 | Time: 33.0s
      ...
      Train Loss: 0.7052 | Beta: 1.0967 | Time: 1965.9s
      Validating...
      Val Loss: 0.6937
      ✓ New best model saved!
    ```
3.  **Scheduler Triggers (Optional but good!)**: If training stalls, you may see a message from the scheduler indicating it has reduced the learning rate. This is a sign the system is working as intended.
    ```
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.0005.
    ```
4.  **Evaluation Runs**: After 20 epochs, the evaluation script will run.
    ```
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Step 7/6 (Bonus): Evaluating with proper 1-vs-100 protocol...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Running proper evaluation...
    ```
5.  **Final Summary**: The script concludes with a summary of the results and where to find them.
    ```
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                          ✅ ALL DONE!                                 ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Results:
      • Training checkpoint: runs/graphflix_10pct_.../best.pt
      • Evaluation results: runs/graphflix_10pct_.../test_results_proper.txt
    ```

By following this guide, a new team member should be able to successfully kick off a training run and understand the key components of the process.
