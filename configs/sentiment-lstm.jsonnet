# Architecture hyperparameters.
local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 200;
local N_LAYERS = 1;

# Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = std.extVar("OPTIM");
local LEARNING_RATE = 2e-5;  # 5e-5, 3e-5, 2e-5

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET_SIZE = 8544;


// TODO: Use a bert encoder here.
{
  "dataset_reader": {
    "type": "basic_sentiment",
    "binary_sentiment": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      }
    },
  },

  "train_data_path": DATA_ROOT + "/stanford/trees/train.txt",
  "validation_data_path": DATA_ROOT + "/stanford/trees/dev.txt",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
        "norm": "param_norm",
        "num_saturated": {
          "type": "num_saturated",
          "weight_delta": 0.01,
          "act_delta": 0.01,
          "act_norm": 2,
        },
        "mask_change": {
          "type": "mask_change",
          "percent": 0.5,
          "normalize": false,
        },
        "norm_mask_change": {
          "type": "mask_change",
          "percent": 0.5,
          "normalize": true,
        },
    },

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
      },
    },

    // "seq2seq_encoder": {
    //   "type": "stacked_self_attention",
    //   "input_dim": EMBEDDING_DIM,
    //   "hidden_dim": 1000,
    //   "projection_dim": 100,
    //   "feedforward_hidden_dim": 5000,
    //   "num_layers": 10,
    //   "num_attention_heads": 10,
    // },

    "seq2seq_encoder": {
      "type": "lstm",
      "input_size": EMBEDDING_DIM,
      "hidden_size": HIDDEN_DIM,
      "num_layers": N_LAYERS,
    },

    "seq2vec_encoder": {
      "type": "boe",
      // "embedding_dim": 1000,
      "embedding_dim": HIDDEN_DIM,
    },

  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": BATCH_SIZE,
    },
  },

  "trainer": {
    "optimizer": {
      "type": OPTIMIZER,
      "lr": LEARNING_RATE,
    },
    "num_epochs": 1000,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
