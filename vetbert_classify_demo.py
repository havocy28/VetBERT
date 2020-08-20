
import sys

if len(sys.argv) < 2:
    print('ERROR: you must specify the file formatted the same as the \'clinical_notes.xls\' demo file')
    exit()

import datetime
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from bert import modeling
from bert import optimization
from bert import run_classifier
from bert import tokenization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# loads files
DATA_FILE = sys.argv[1]

try:
    df_test = pd.read_excel(DATA_FILE)
except:
    print('ERROR: make sure you have an excel file formated like the example xls file included in 97-2003 format')
    exit()

if df_test.columns[0] != 'Text' or df_test.columns[1] != 'Label':
    print('ERROR: make sure you have your columns labeled \'Text\' and \'Label\'')
    exit()

df_test['Text'] = df_test['Text'].astype(str)

# location of
folder = './VetBERT_model'
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = f'{folder}'

# location of pretrained models
OUTPUT_DIR = f'./models'
print(f'>> Model output directory: {OUTPUT_DIR}')
print(f'>>  BERT pretrained directory: {BERT_PRETRAINED_DIR}')

# encodes labels

LabelTotals = 38  # based on nunique labels of formatted convenia df

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)
d = dict(zip(le.transform(le.classes_), le.classes_))
try:
    df_test["Label"] = le.transform(df_test["Label"])
except:
    print('ERROR: make sure you use one of the sample labels included in the labels.txt')
    exit()


X_test, y_test = df_test["Text"].values,  df_test["Label"].values


def create_examples(lines, set_type, labels=None):
    # Generate data for the BERT model
    guid = f'{set_type}'
    examples = []
    if guid == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(
                run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '0'
            examples.append(
                run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 512
# Model configs
SAVE_CHECKPOINTS_STEPS = 10000
KEEP_CHECKPOINT_MAX = 5
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'vetbert.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
use_tpu = False
# tpu_name = 'tputemp0'

label_list = [str(num) for num in range(LabelTotals)]
tokenizer = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE, do_lower_case=False)
#train_examples = create_examples(X_train, 'train', labels=y_train)

tpu_cluster_resolver = None
if use_tpu and tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_name)  # Since training will happen on GPU, we won't need a cluster resolver
# TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))


num_train_steps = 50000
# num_train_steps = int(
#    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


# allows for verbose mode if you uncomment

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=EVAL_BATCH_SIZE
)


print('>> Started running Test at {} '.format(datetime.datetime.now()))


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        print(params)
        batch_size = 8

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[
                            num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


predict_examples = create_examples(X_test, 'test')

predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

eval_batch_size = 8

predict_batch_size = eval_batch_size

predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False,
    #    predict_batch_size=eval_batch_size
    #    eval_steps= int(len(predict_examples) // eval_batch_size)
)

result = estimator.predict(input_fn=predict_input_fn)


preds = []
preds_score = []
top_3_lists = []
for prediction in result:
    predication_array = prediction['probabilities']
    preds.append(np.argmax(prediction['probabilities']))
    preds_score.append(np.amax(prediction['probabilities']))
    top_3_lists.append(list(predication_array.argsort()[-3:][::-1]))

accuracy = accuracy_score(y_test, preds)

# convert to labels
y_test_names = [d[num] for num in y_test]
pred_names = [d[num] for num in preds]

for i, top_3 in enumerate(top_3_lists):
    top_3_lists[i] = [d[num] for num in top_3]


print('calculated accuracy: which is: %s' % accuracy)
report = classification_report(y_test_names, pred_names)

print(report)
pred_list = []
for text, test, pred, score, top_3 in zip(X_test, y_test_names, pred_names, preds_score, top_3_lists):

    pred_list.append([text, test, pred, score, top_3])

df_pred_data = pd.DataFrame(
    pred_list, columns=['text', 'truth', 'predicted', 'score', 'top_3_lists'])

#pred_data = {'text': X_test, 'truth' : y_test_names, 'predicted' : pred_names}
# print(len(X_train))
# print(len(y_test_names))
# print(len(pred_names))

df_pred_data.to_excel('./output/predicted_outputs.xls', index=False)
print('results copied to output folder')
