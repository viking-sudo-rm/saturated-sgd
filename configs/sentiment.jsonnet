local utils = import "utils.jsonnet";

# Architecture hyperparameters.
local MODEL_NAME = std.extVar("MODEL");
local EMBEDDING_DIM = utils.getTransformerDim(MODEL_NAME);

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
      "roberta": {
        "type": "pretrained_transformer_mismatched",
        "model_name": MODEL_NAME,
      }
    },
  },

  "train_data_path": DATA_ROOT + "/stanford/trees/train.txt",
  "validation_data_path": DATA_ROOT + "/stanford/trees/dev.txt",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
        "norm": "param_norm",
        // The similarity between \theta_t and \theta_{t+1}, should go to 1 always.
        "alpha": "outward_projection",
        // The similarity between \theta_t and \Delta\theta. Does it go to 1?
        "beta": {
          "type": "outward_projection",
          "use_step": true,
        },
    },

    "text_field_embedder": {
      "token_embedders": {
        "roberta":{
            "type": "pretrained_transformer_mismatched",
            "model_name": MODEL_NAME,
        }
      }
     },

    "seq2seq_encoder": {
      "type": "pass_through",
      "input_dim": EMBEDDING_DIM,
    },

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": EMBEDDING_DIM,
    },

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "roberta___token_ids"]],
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
