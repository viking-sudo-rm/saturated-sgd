# Model hyperparameters.
local EMBEDDING_DIM = 400;  # 300;
local CHAR_EMBEDDING_DIM = 50;
local INPUT_DIM = EMBEDDING_DIM + CHAR_EMBEDDING_DIM;
local HIDDEN_DIM = 1150;
local NUM_LAYERS = 2;

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local OPTIMIZER = std.extVar("OPTIM");
local BATCH_SIZE = 20;  # 16;
local PATIENCE = 5;
local CHAR_DROPOUT = std.extVar("C_DROP");
local EMBED_DROPOUT = 0.5;  # Based on the language model, this shouldn't affect saturation.
# local DROPOUT = 0.5;
local WEIGHT_DECAY = std.extVar("L2");

# TODO: Get a strong baseline LSTM going using https://github.com/salesforce/awd-lstm-lm.

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET = "penn";


{
  "dataset_reader": {
    "type": "simple_lm",
    "end_token": "<eos>",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      },
      "characters": {
        "type": "characters",
      },
    },
  },

  "train_data_path": DATA_ROOT + "/" + DATASET + "/train.txt",
  "validation_data_path": DATA_ROOT + "/" + DATASET + "/valid.txt",
  
  "model": {
    "type": "sat_metrics_lm",

    "activation_metrics": {
        "sat-.5": {"type": "num_saturated", "delta": 0.5},
        "sat-.25": {"type": "num_saturated", "delta": 0.25},
        "sat-.1": {"type": "num_saturated", "delta": 0.1},
        "sat-.01": {"type": "num_saturated", "delta": 0.01},
        "sat-dist": {"type": "sat_dist"},
        "preact": {"type": "preactivation_abs"},
    },

    "parameter_metrics": {
        "norm": "param_norm",
    },

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
        "characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 8,
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 8,
            "num_filters": CHAR_EMBEDDING_DIM,
            "ngram_filter_sizes": [5],
          },
          "dropout": CHAR_DROPOUT,
        },
      },
    },

    "contextualizer": {
      "type": "percent_saturated_dropout",
      "percent": 0.0,
      "norm": "gru",
      "encoder": {
          "type": "gru",
          "input_size": INPUT_DIM,
          "hidden_size": HIDDEN_DIM,
          "num_layers": NUM_LAYERS,
      },
    },

    "dropout": EMBED_DROPOUT,

  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["source", "tokens___tokens"]],
    "batch_size": BATCH_SIZE,
  },
  "trainer": {
    "optimizer": {
      "type": OPTIMIZER,
      "weight_decay": WEIGHT_DECAY,
    },
    "num_epochs": 300,
    "patience": PATIENCE,
    "cuda_device": 0,
    "validation_metric": "-perplexity",
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
  }
}
