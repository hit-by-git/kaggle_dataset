# Thesis Experiments: Multi-Model Linguistic Evaluation

This repository contains the code to run the V2 Thesis Plan experiments on Kaggle using a free tier 2x T4 GPU setup and `vllm`.

## How to Run on Kaggle

1. **Create a New Notebook in Kaggle:**
   - Go to Kaggle -> Notebooks -> New Notebook.
   - On the right-side panel, under **Accelerator**, select **GPU T4 x2**.
   - Make sure **Internet** is toggled **ON**.

2. **Add Your Hugging Face Token:**
   - Gemma 2 is a gated model. You must accept the terms on Hugging Face first: https://huggingface.co/google/gemma-2-9b-it
   - Get your Hugging Face access token from your HF profile.
   - In Kaggle, go to **Add-ons -> Secrets**, add a new secret named `HF_TOKEN`, and paste your token.

3. **Upload your Dataset:**
   Upload your `result.jsonl` file as a Kaggle Dataset (via the "Add Data" button on the right panel). Note the folder name Kaggle assigns it.

4. **Install Dependencies & Clone Repo:**
   Create a code cell and run:
   ```bash
   # Uninstall problematic pre-installed Kaggle packages that clash with vLLM
   !pip uninstall -y torchaudio torchvision -q
   # Upgrade transformers to match the latest huggingface_hub
   !pip install -U transformers -q
   # Clone the experiment repo
   !git clone https://github.com/hit-by-git/kaggle_dataset.git
   # Install experiment requirements
   !pip install -r kaggle_dataset/requirements.txt -q
   ```

5. **Login to Hugging Face:**
   Create a code cell and run:
   ```python
   from kaggle_secrets import UserSecretsClient
   from huggingface_hub import login
   import os

   user_secrets = UserSecretsClient()
   hf_token = user_secrets.get_secret("HF_TOKEN")
   os.environ["HF_TOKEN"] = hf_token
   login(token=hf_token)
   ```

6. **Run the Main Experiment Script:**
   Create a code cell and run (replace `your-dataset-name` with your actual Kaggle dataset folder):
   ```bash
   !python kaggle_dataset/main.py \
       --data /kaggle/input/your-dataset-name/result.jsonl \
       --model google/gemma-2-9b-it \
       --output /kaggle/working/experiment_results.csv
   ```

7. **Evaluate the Results:**
   Create a code cell and run:
   ```bash
   !python kaggle_dataset/evaluate_results.py --results /kaggle/working/experiment_results.csv
   ```
   This prints Accuracy, Precision, Recall, F1, and Confusion Matrices for your thesis.
