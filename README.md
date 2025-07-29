# onboarding-mate
Example code for onboarding AI assistant

**📌 Information**

This example code uses the GPT2 model from HuggingFace and the dataset "wikitext" from Salesforce as example dataset. 

The Saleforces wikitext dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License. 

For more information related to the dataset and its subset, please visit: https://huggingface.co/datasets/Salesforce/wikitext

Please not that the example code can be run smoothly but the performance is not optimized due to random hyperparameters.

**🚀 Prerequisites**

To install this starter code, you will need: 

- Python
- Pip

**🔧 Installation**

1. Clone this repository
2. Create an environment and install dependencies. Dependencies can be listed either in requirements.txt (for local implementation) or in env.yml (for Puhti implementation). 

For local usage, you can either use python virtual environment or conda environment. Example below is for python virtual environment.

```bash 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For usage on Puhti, it is recommended to use Tykky container wrapper.

```bash
module purge
module load tykky
mkdir <install_dir>
conda-containerize new --prefix <install_dir> env.yml
export PATH="<install_dir>/bin:$PATH"
```

**🧠 Usage**

## Download Data

```bash
python download_data.py
```

## Fine-tune the Model

```bash
python tuning.py
```

## Run the Assistant

To run the assistant, you will have to open an interactive shell on LUMI

```bash
python assistant.py
```

Type your onboarding questions and get answers from the model


**🧑‍🏫 What you have to-do?**

Comes up a creative full scaled solution accompanied with this Proof-of-Concept code to demonstrate an Onboarding AI assistant. 


**💼 TODOS**

Feel free to change anything!