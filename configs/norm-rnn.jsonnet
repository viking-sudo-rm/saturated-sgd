# Optimization hyperparameters.
local BATCH_SIZE = 16;
local HIDDEN_SIZE = 16;
local SCALE = std.parseJson(std.extVar("SCALE"));


{
  "dataset_reader": {
    "type": "anbn_tagging",
  },

  "train_data_path": "1:1000",
  "validation_data_path": "1000:1100",
  
 "model": {
    "type": "anbnc_tagger",  # simple_tagger, sat_metrics_tagger

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 4
        }
      }
    },

    "encoder": {
      "type": "norm_rnn",
      "input_dim": 4,
      "hidden_dim": HIDDEN_SIZE,
      "scale": SCALE,
    }

  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": BATCH_SIZE,
    },
  },

  "trainer": {
    "optimizer": "adam",
    "validation_metric": "-loss",
    "num_epochs": 20,
    "patience": 5,
    "cuda_device": 0,
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
