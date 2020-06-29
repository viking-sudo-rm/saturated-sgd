// Architecture hyperparameters.
local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 650;
local N_LAYERS = 2;

// Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = "norm_sgd";
local LEARNING_RATE = 2e-5;  # 5e-5, 3e-5, 2e-5
local DROPOUT = 0.3;


{
  "dataset_reader": {
    "type": "demo_pos_tagging",
    "token_indexers": {
      "tokens": "single_id",
    },
  },

  "train_data_path": "https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/training.txt",
  "validation_data_path": "https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/validation.txt",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
      "norm": "param_norm",
    },

    "activation_metrics": {
      "sat_sim": {
        "type": "cosine_similarity",
        "key": "encoded_text",
      },
      "sat_acc": {
        "type": "accuracy",
        "key": "predictions",
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
      "type": "lstm",
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
