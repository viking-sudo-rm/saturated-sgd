# Architecture hyperparameters.
local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 200;
local N_LAYERS = 1;

# Optimization hyperparameters.
local BATCH_SIZE = 32; # 16, 32
local OPTIMIZER = std.extVar("OPTIM");
local LEARNING_RATE = 2e-5;  # 5e-5, 3e-5, 2e-5
local DROPOUT = 0.3;

# Path to the data on the file system.
local PATH = "/net/nfs.corp/allennlp/suching/acl_2019_datasets/";
local DATASET = std.extVar("DATASET");


// TODO: Use a bert encoder here.
{
  "dataset_reader": {
    "type": "text_classification_json_with_sampling",
    "lazy": false,
    // "max_sequence_length": 512,
    // "sample": 500,
    "token_indexers": {
      "tokens": "single_id",
    },
    "tokenizer": "spacy",
  },

  "train_data_path": PATH + DATASET + "/train.jsonl",
  "validation_data_path": PATH + DATASET + "/dev.jsonl",
  
  "model": {
    "type": "sat_metrics_classifier",

    "parameter_metrics": {
      // "norm": "param_norm",
    },

    "activation_metrics": {
      "sat_sim": {
        "type": "cosine_similarity",
        "key": "embedded_sequence",
      },
    },

    "prune_metrics": {
      "prune_sim": {
        "type": "cosine_similarity",
        "key": "embedded_sequence",
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
  }
}
