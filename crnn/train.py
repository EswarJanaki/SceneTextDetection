import argparse
import pprint
import shutil
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow import keras

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=Path, required=True, help="The config file path."
)
parser.add_argument(
    "--save_dir",
    type=Path,
    required=True,
    help="The path to save the models, logs, etc.",
)
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)["train"]
pprint.pprint(config)

args.save_dir.mkdir(exist_ok=True)
if list(args.save_dir.iterdir()):
    raise ValueError(f"{args.save_dir} is not a empty folder")
shutil.copy(args.config, args.save_dir / args.config.name)

strategy = tf.distribute.MirroredStrategy()
batch_size = config["batch_size_per_replica"] * strategy.num_replicas_in_sync

dataset_builder = DatasetBuilder(**config["dataset_builder"])
train_ds = dataset_builder(config["train_ann_paths"], batch_size, True)
val_ds = dataset_builder(config["val_ann_paths"], batch_size, False)

with strategy.scope():
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        **config["lr_schedule"]
    )
    model = build_model(
        dataset_builder.num_classes,
        weight=config.get("weight"),
        img_shape=config["dataset_builder"]["img_shape"],
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=CTCLoss(),
        metrics=[SequenceAccuracy()],
    )

model.summary()

model_prefix = "{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}"
model_path = f"{args.save_dir}/{model_prefix}.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True),
    keras.callbacks.TensorBoard(
        log_dir=f"{args.save_dir}/logs", **config["tensorboard"]
    ),
]
'''
import argparse
import pprint
import shutil
from keras.utils import custom_object_scope
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow import keras
import matplotlib.pyplot as plt
from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True,
                    help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True,
                    help='The path to save the models, logs, etc.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

args.save_dir.mkdir(exist_ok=True)
'''
'''if list(args.save_dir.iterdir()):
    raise ValueError(f'{args.save_dir} is not a empty folder')''''''
#shutil.copy(args.config, args.save_dir / args.config.name)

strategy = tf.distribute.MirroredStrategy()
batch_size = config['batch_size_per_replica'] * strategy.num_replicas_in_sync

dataset_builder = DatasetBuilder(**config['dataset_builder'])
train_ds = dataset_builder(config['train_ann_paths'], batch_size, True)
val_ds = dataset_builder(config['val_ann_paths'], batch_size, False)

with strategy.scope():
      lr_schedule = keras.optimizers.schedules.CosineDecay(
          **config['lr_schedule'])
      if not list(args.save_dir.iterdir()):
        model = build_model(dataset_builder.num_classes,
                          weight=config.get('weight'),
                          img_shape=config['dataset_builder']['img_shape'])
      else:
        with custom_object_scope({'CTCLoss': CTCLoss,'SequenceAccuracy':SequenceAccuracy}):
          model= tf.keras.models.load_model(args.save_dir, compile=False)
        shutil.rmtree(args.save_dir)
      model.compile(optimizer=keras.optimizers.Adam(lr_schedule),
                    loss=CTCLoss(), metrics=[SequenceAccuracy()])

model.summary()
model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
model_path = f'{args.save_dir}/{model_prefix}.h5'
'''
history=model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks,
          validation_data=val_ds)
# Plot the training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss Plot ',)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#print(plt.show())
model.save(args.save_dir)
plt.savefig(str(args.save_dir)+'/loss_plot.png')

