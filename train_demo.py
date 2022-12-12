import os
from data import DATA_SET_DIR
from elmo.lm_generator import LMDataGenerator
from elmo.model import ELMo
import argparse
import time

parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'test_dataset': 'wikitext-2/wiki.test.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 2000,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 10,
    'patience': 2,
    'batch_size': 1,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 2,
    'n_highway_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 400,
    'hidden_units_size': 200,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="training.")
    parser.add_argument("--evaluate", action='store_true', help="evaluation.")
    parser.add_argument("--save_model", action='store_true', help="save model.")
    parser.add_argument("--profile", action='store_true', help="profile.")
    parser.add_argument("--tensorboard", action='store_true')
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--precision", type=str, default='float32', help="float32, int8 or float16")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("-i", "-n", "--num_iter", type=int, default=200)
    parser.add_argument("--epoch_warmup", type=int, default=3)
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

parameters['batch_size'] = args.batch_size
parameters['precision'] = args.precision

if args.precision == 'float16' :
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("---- with foat16 mix_precision")

# timeline
import pathlib
timeline_dir = str(pathlib.Path.cwd()) + '/timeline/' + str(os.getpid())

# Set-up Generators
train_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['train_dataset']),
                                  os.path.join(DATA_SET_DIR, parameters['vocab']),
                                  sentence_maxlen=parameters['sentence_maxlen'],
                                  token_maxlen=parameters['token_maxlen'],
                                  batch_size=parameters['batch_size'],
                                  shuffle=parameters['shuffle'],
                                  token_encoding=parameters['token_encoding'])

val_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['valid_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

test_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['test_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

# Compile ELMo
elmo_model = ELMo(parameters)
elmo_model.compile_elmo(print_summary=True)

# Train ELMo
if args.train:
    elmo_model.train(train_data=train_generator, valid_data=val_generator)

# Persist ELMo Bidirectional Language Model in disk
if args.save_model:
    elmo_model.save(sampled_softmax=False)

# Evaluate Bidirectional Language Model
if args.evaluate:
    print("## Evaluate Start:")
    total_time = 0.0
    total_sample = 0
    # already processed with batchSize
    # num_iter = int(len(test_generator) / args.batch_size)
    num_iter = len(test_generator)

    print("dataset length: {}, batch_size: {}".format(len(test_generator), args.batch_size))
    for i in range(args.epochs):
        if args.tensorboard and i == (args.epochs // 2):
            print("---- collect tensorboard")
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
            tf.profiler.experimental.start('./tensorboard_data', options = options)
        start_time = time.time()
        elmo_model.evaluate(test_generator, num_iter=num_iter, batch_size=args.batch_size)
        end_time = time.time()
        print("Iteration: {}, inference time: {}".format(i, end_time - start_time))
        if i > args.epoch_warmup:
            total_time += end_time - start_time
            total_sample += num_iter * args.batch_size
        if args.tensorboard and i == (args.epochs // 2):
            tf.profiler.experimental.stop()
            print("---- collect tensorboard end")
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("### Latency:: {:.2f} ms".format(latency))
    print("### Throughput: {:.3f} samples/s".format(throughput))

# Build ELMo meta-model to deploy for production and persist in disk
# elmo_model.wrap_multi_elmo_encoder(print_summary=True, save=True)

# Load ELMo encoder
# elmo_model.load_elmo_encoder()

# Get ELMo embeddings to feed as inputs for downstream tasks
# elmo_embeddings = elmo_model.get_outputs(test_generator, output_type='word', state='mean')

# BUILD & TRAIN NEW KERAS MODEL FOR DOWNSTREAM TASK (E.G., TEXT CLASSIFICATION)

