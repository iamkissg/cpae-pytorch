{
  "random_seed": 1,
  "numpy_seed": 1,
  "pytorch_seed": 1,

  // dataset reader
  "dataset_reader": {
    "type": "cpae_reader",
    "max_definition_length": 100,
    "definition_namespace": "definition",
    "word_namespace": "word",
    "definition_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "definition_token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false,
        "namespace": "definition"
      }
    },
    "word_token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false,
        "namespace": "word"
      }
    }
  },

  // pre-defined vocabulary
  // if not using, just comment these lines
  "vocabulary": {
    "pretrained_files": {
      // vocab extracted by the original CPAE code
      "definition": "data/vocab.txt"
    },
    "only_include_pretrained_words": true,
    "max_vocab_size": {
      "definition": 45102  // do not include @@PADDING@@ and @@UNKNOWN@@
    },
  },
  "train_data_path": "data/en_wn_full_all.jsonl",

  "model": {
    "type": "cpae",
    "alpha": 1,
    "beta": 64,
    "definition_pooling": "last",  // last, max, mean, self-attentive
    "text_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          // num_embeddings should be no more than vocabulary.max_vocab_size + @@PADDING@@ + @@UNKNOWN@@,
          // and the resultant vocabulary size will be extended to the given vocabulary and specified size
          // if using pretrained_file, comment out num_embeddings
          // "num_embeddings": 2000,
          "pretrained_file": "/home/kissg/Dataset/word_vectors/GFPW/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.txt",
          "embedding_dim": 300,
          "trainable": true,
          "vocab_namespace": "definition"
        }
      }
    },
    "definition_encoder": {
      "type": "lstm",
      "bidirectional": false,
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "dropout": 0.0
    },
    "definition_feedforward": {
      "input_dim": 300,
      "num_layers": 1,
      "hidden_dims": [300],
      "activations": ["linear"],
      "dropout": [0.0]
    },
    "definition_decoder": {
      "input_dim": 300,
      "num_layers": 1,
      "hidden_dims": [17456],
      "activations": ["linear"],
      "dropout": [0.0]
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "should_log_parameter_statistics": false,
    "num_epochs": 50,
    "patience": 50,
    "num_serialized_models_to_keep": 1,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0003,
      "betas": [0.9, 0.999]
    }
  }
}