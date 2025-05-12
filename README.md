# Advanced NLP Exercise 1: Fine-Tuning

This is the code base for the **ANLP HUJI** course Exercise 1, where we fine-tune pretrained models to perform sentence pair classification on the **MRPC dataset**.

---

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

---

## Fine-Tuning and Prediction on Test Set

### Command to Fine-Tune the Model

Run the following command to fine-tune the model on the MRPC dataset:

```bash
python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_train --model_path <path to pretrained model>
```

### Command to Predict on the Test Set

To make predictions on the test set using the trained model, run:

```bash
python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_predict --model_path <path to fine-tuned model>
```

---

## Explanation of Parameters

* `--max_train_samples`: Number of training samples to use (default: all).
* `--max_eval_samples`: Number of validation samples to use (default: all).
* `--max_predict_samples`: Number of test samples to use for predictions (default: all).
* `--lr`: Learning rate for training (default: 2e-5).
* `--num_train_epochs`: Number of epochs for training.
* `--batch_size`: Batch size for training and prediction.
* `--do_train`: Flag to fine-tune the model. Set this to train the model.
* `--do_predict`: Flag to run predictions on the test set. Set this to generate predictions after training.
* `--model_path`: Path to the pretrained or fine-tuned model.

---

## Output Files

### For Training

The model and tokenizer are saved after fine-tuning, and can be reloaded for future predictions.

### For Prediction

A `predictions.txt` file will be generated, containing the prediction results for all test samples, formatted as:

```php-template
<sentence1>###<sentence2>###<predicted_label>
```