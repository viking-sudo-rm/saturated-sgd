local bool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

# Model hyperparameters.
local EMBEDDING_DIM = 100;
local HIDDEN_DIM = std.extVar("N_HID");

# Optimization hyperparameters.
local OPTIMIZER = "adamw";
local BATCH_SIZE = 16;
local PATIENCE = 10;
local EMBED_DROPOUT = 0.0;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";


// TODO: Use a bert encoder here.
{
  "dataset_reader": {
    "type": "agreement",
    "token_indexers": {
      "tokens": "single_id",
    },
  },

  "train_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.train",
  "validation_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.val",
  
  "model": {
    "type": "sat_metrics_classifier",

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
      },
    },

    "seq2seq_encoder": {
        "type": "rnn",
        "input_size": EMBEDDING_DIM,
        "hidden_size": HIDDEN_DIM,
        "bidirectional": false,
        "num_layers": 1,
    },

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": HIDDEN_DIM,
    },

    "dropout": EMBED_DROPOUT,

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "tokens___tokens"]],
      "batch_size": BATCH_SIZE,
  },
  "trainer": {
      "optimizer": {
        "type": OPTIMIZER,
      },
      "num_epochs": 300,
      "patience": PATIENCE,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "checkpointer": {
        "num_serialized_models_to_keep": 1,
      },
  }
}
