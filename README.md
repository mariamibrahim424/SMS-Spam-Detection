# SMS Spam Detection

This project builds a machine learning model to classify SMS messages as either spam or ham (non-spam). The process involves cleaning and preprocessing raw SMS data, extracting features, training a model, evaluating its performance, and saving the trained model for future use.

## Installation

1. Clone the repository:

```bash
   git clone git@github.com:mariamibrahim424/SMS-Spam-Detection.git
   cd sms-spam-detection
```

2. Create a virtual environment (optional but recommended):

```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

3. Install the required libraries:

```bash
   pip install -r requirements.txt
```

4. Download necessary data and dependencies:

```bash
   python3 scripts/download_nltk_data.py
```

5. Run the model
   This will execute the model, allowing you to classify SMS messages as either spam or ham.

```bash
   python3 run.py
```
