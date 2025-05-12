# Advanced NLP Exercise 1: Fine Tuning

This is the code base for the *ANLP HUJI* course Exercise 1, where we fine-tune pretrained models to perform sentence pair classification on the *MRPC dataset*.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_train/--do_predict --model_path <path to prediction model>```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.
