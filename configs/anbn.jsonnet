# Architecture hyperparameters.
local EMBEDDING_DIM = 2;
local HIDDEN_DIM = std.parseInt(std.extVar("N_HID"));

# Optimization hyperparameters.
local BATCH_SIZE = 16;
local OPTIMIZER = "adamw";
local NUM_EPOCHS = 50;
local PATIENCE = 10;


{
  "dataset_reader": {
    "type": "anbn",
  },

  "train_data_path": "1:1000",
  "validation_data_path": "1000:1100",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
        "norm": "param_norm",
    },

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": EMBEDDING_DIM,
        }
      }
     },

    "seq2seq_encoder": {
      "type": "rnn",
      "input_size": EMBEDDING_DIM,
      "hidden_size": HIDDEN_DIM,
    },

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": HIDDEN_DIM,
    }

  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": BATCH_SIZE,
    },
  },

  "trainer": {
    "optimizer": OPTIMIZER,
    "num_epochs": NUM_EPOCHS,
    "patience": PATIENCE,
    "cuda_device": 0,
    "validation_metric": "-loss",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    }
  }
}
