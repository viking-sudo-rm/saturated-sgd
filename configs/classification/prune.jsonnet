local utils = import "../utils.jsonnet";

# Architecture hyperparameters.
local MODEL_NAME = std.extVar("MODEL");
local EMBEDDING_DIM = utils.getTransformerDim(MODEL_NAME);

# Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = "huggingface_adamw";
local LEARNING_RATE = 2e-5;  # 5e-5, 3e-5, 2e-5
local NUM_EPOCHS = 20;

# Pruning hyperparameters.
local PRUNE_PERCENT = std.parseJson(std.extVar("PERCENT"));
local PRUNE_WEIGHTS =
  if std.extVar("NORM") == "WEIGHTS" then true
  else false;
local PRUNE_MODE = std.extVar("MODE");
local PRUNE_EPOCH = 4;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET_SIZE = 8544;


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
    },

    "text_field_embedder": {
      "token_embedders": {
        "roberta":{
            // "type": "pretrained_transformer_mismatched",
            // "model_name": MODEL_NAME,
            "type": "pretrained_pruner_wrapper",
            "pretrained_embedder": {
              "type": "pretrained_transformer_mismatched",
              "model_name": MODEL_NAME,
            },
            "prune_weights": PRUNE_WEIGHTS,
            "prune_mode": PRUNE_MODE,
            "percent": PRUNE_PERCENT,
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
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": NUM_EPOCHS,
      "num_steps_per_epoch": std.floor(DATASET_SIZE / BATCH_SIZE),
    },
    "num_epochs": NUM_EPOCHS,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "epoch_callbacks": [
      {
        "type": "pruner",
        "epoch": PRUNE_EPOCH,
      },
    ],
  }
}
