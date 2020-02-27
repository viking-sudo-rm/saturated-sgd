# Architecture hyperparameters.
local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 200;
local N_LAYERS = 3;

# Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = "huggingface_adamw";
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
        "angles": "outward_projection",
        "step": "mag_dir_step",
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
      "type": "lstm",
      "input_size": EMBEDDING_DIM,
      "hidden_size": HIDDEN_DIM,
      "num_layers": N_LAYERS,
    },

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": HIDDEN_DIM,
    },

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "tokens___tokens"]],
      "batch_size": BATCH_SIZE,
  },
  "trainer": {
    "optimizer": {
      "type": OPTIMIZER,
      "lr": LEARNING_RATE,
    },
    // This is the correct learning rate setup for finetuning BERT.
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 20,
      "num_steps_per_epoch": std.floor(DATASET_SIZE / BATCH_SIZE),
    },
    "num_epochs": 20,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
