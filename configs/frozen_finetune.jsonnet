local utils = import "utils.jsonnet";

# Architecture hyperparameters.
local MODEL_NAME = std.extVar("MODEL");
local EMBEDDING_DIM = utils.getTransformerDim(MODEL_NAME);

# Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = "huggingface_adamw";
local LEARNING_RATE = 2e-5;  # 5e-5, 3e-5, 2e-5
local EPOCHS = 10;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET_SIZE = 141948;


{
  "dataset_reader": {
    "type": "agreement",
    "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME,
    },
    "token_indexers": {
      "roberta": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME,
      }
    },
  },

  "train_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.train",
  "validation_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.val",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
        "norm": "param_norm",
    },

    "activation_metrics": {
        "agree": {
            "type": "accuracy",
            "key": "preds",
        },
    },

    "text_field_embedder": {
      "token_embedders": {
        "roberta":{
            "type": "pretrained_transformer_frozen",
            "embedder": {
                "model_name": MODEL_NAME,
            },
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
    // This is the correct learning rate setup for finetuning BERT.
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": EPOCHS,
      "num_steps_per_epoch": std.floor(DATASET_SIZE / BATCH_SIZE),
    },
    "num_epochs": EPOCHS,
    "cuda_device": 0,
    // "validation_metric": "+accuracy",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
