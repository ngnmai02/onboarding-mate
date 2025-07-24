# onboarding-mate
Example code for onboarding AI assistant
Dataset used in this: 'Salesforce/wikitext', 'wikitext-2-raw-v1' from HF

## Setup
```bash
pip install -r requirements.txt
```
or on LUMI
```bash
module load LUMI
module load lumi-containter-wrapper
mkdir <install_dir>
conda-containerize new --prefix <install_dir> env.yml
```

## Download Data

```bash
python download_data.py
```

## Fine-tune the Model

```bash
python tuning.py
```

## Run the Assistant

```bash
python assistant.py
```

Type your onboarding questions and get answers from the model