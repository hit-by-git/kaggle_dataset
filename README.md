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

3. **Install Dependencies:**
   Create a code cell and run:
   ```bash
   # Uninstall problematic pre-installed Kaggle packages
   !pip uninstall -y torchaudio torchvision
   # Upgrade transformers to avoid huggingface_hub conflicts
   !pip install -U transformers
   # Install requirements
   !pip install -r requirements.txt
   ```

4. **Login to Hugging Face (in Notebook):**
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

5. **Upload your Dataset:**
   Upload your `result.jsonl` file to the Kaggle input directory (e.g., via the "Add Data" button).

6. **Run the Main Experiment Script:**
   Create a code cell and run (replace the path with your actual Kaggle dataset path):
   ```bash
   !python main.py --data /kaggle/input/your-dataset-name/result.jsonl --model google/gemma-2-9b-it
   ```
   *Note: This will output `experiment_results.csv` to the Kaggle working directory.*

7. **Evaluate the Results:**
   Create a code cell and run:
   ```bash
   !python evaluate_results.py --results experiment_results.csv
   ```
   This will print out all the Academic metrics (Accuracy, Precision, Recall, F1, and Confusion Matrices) needed for your thesis chapters.
