# Optimization hyperparameters.
local HIDDEN_SIZE = std.extVar("N_HID");

# Task parameters.
local NUM_TRAIN = 100000;
local NUM_VALID = 10000;
local LENGTH = 128;

{
  "dataset_reader": {
    "type": "max_difference",
    "seed": 2,
  },

  "train_data_path": NUM_TRAIN + ":" + LENGTH,
  "validation_data_path": NUM_VALID + ":" + LENGTH,
  
 "model": {
    "type": "sat_metrics_tagger",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 4
        }
      }
    },

    "encoder": {
      "type": "saturated_lstm",
      "input_size": 4,
      "hidden_size": HIDDEN_SIZE,
    }

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "tokens___tokens"]],
      "batch_size": 16,
  },
  "trainer": {
    "optimizer": "adam",
    "validation_metric": "-loss",
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
