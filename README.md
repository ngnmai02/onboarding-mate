# onboarding-mate
Example code for onboarding AI assistant
Dataset used in this: 'Salesforce/wikitext', 'wikitext-2-raw-v1' from HF

**🚀 Prerequisites**

To run this starter code, you will need: 

- Pytho, pip to install


**🔧 Installation**

1. Clone this repository
2. Create an environment and install dependencies

```bash 
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

```

For usage on LUMI, it is recommended to use container-wrapper.

```bash
module load LUMI
module load lumi-containter-wrapper
mkdir <install_dir>
conda-containerize new --prefix <install_dir> env.yml
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