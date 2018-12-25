# AllenNLP seq2seq playground


* [SyntheticQA.ipynb](SyntheticQA.ipynb) - Synthetic  data generator. Used to generate train and test data.
* [allen_test_conf.json](allen_test_conf.json) - AllenNLP experiment configuration.
* [predictor.py](predictor.py) - Script, used for inference with trained model


## Install & Run


Install dependencies
```bash
pip install -r requirements.txt
```

---

If you want to build your own configuration, you may use graphical configuration constructor.
```bash
allennlp configure
```
This command will start a web-server with a configuration tool.

---

Run training
```bash
allennlp train -f -s data/stats allen_test_conf.json
```

Observe various training statistics with Tensoroard
```bash
tensorboard --logdir data/stats/log
```

---

Apply trained model
```
python predictor.py
```