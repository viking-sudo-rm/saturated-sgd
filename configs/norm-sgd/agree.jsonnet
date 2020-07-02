// Architecture hyperparameters.
local RNN = "rnn";
local EMBEDDING_DIM = 50;
local HIDDEN_DIM = 50;
local N_LAYERS = 1;

// Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = std.extVar("OPTIMIZER");
local LEARNING_RATE = std.parseJson(std.extVar("LR"));
local MIN_STEP = std.parseJson(std.extVar("MIN_STEP"));
local DROPOUT = 0.0;
local PATIENCE = 50;


{
  "dataset_reader": {
    "type": "agreement",
    "token_indexers": {
      "tokens": "single_id",
    },
  },

  "train_data_path": "/net/nfs.corp/allennlp/willm/data/rnn_agr_simple/numpred.train",
  "validation_data_path": "/net/nfs.corp/allennlp/willm/data/rnn_agr_simple/numpred.val",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
      "norm": "param_norm",
    },

    "activation_metrics": {
      "sat_sim": {
        "type": "cosine_similarity",
        "key": "embedded_sequence",
      },
      "sat_acc": {
        "type": "accuracy",
        "key": "preds",
        "ignore_mask": true,
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

    "seq2seq_encoder": {
      "type": RNN,
      "input_size": EMBEDDING_DIM,
      "hidden_size": HIDDEN_DIM,
      "num_layers": N_LAYERS,
    },

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": HIDDEN_DIM,
    },

    "dropout": DROPOUT,

  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": BATCH_SIZE,
    },
  },

  "trainer": {
    "optimizer":
    if OPTIMIZER == "floor_sgd"
      then {
        "type": OPTIMIZER,
        "lr": LEARNING_RATE,
        "min_step": MIN_STEP,
      }
      else {
        "type": OPTIMIZER,
        "lr": LEARNING_RATE,
      },
    "num_epochs": 1000,
    "patience": PATIENCE,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },

    "batch_callbacks": [
        {
          "type": "parameter_metrics",
          "metrics": {
            "ortho": {
              "type": "ortho",
            },
          },
        },
    ],

  }
}
