import elasticsearch
from pathlib import Path
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel

# Elastic configuration.
ELASTIC_ADDRESS = "https://3c816b713afd4cf08196bc3520542d99.us-central1.gcp.cloud.es.io:443"
# Uncomment the following lines if start ES with SECURITY ENABLED.
#ELASTIC_ADDRESS = "https://localhost:9200"
ELASTIC_PASSWORD = ""
CA_CERTS_PATH = "/Users/krishnan/ai-demos/vector-search-elastic-tutorial/A2525B64D8BFD084D946539261844AC9A3F7DBDC.crt"

def main():
        # Load a Hugging Face transformers model directly from the model hub
        tm = TransformerModel("sentence-transformers/all-MiniLM-L6-v2", "text_embedding")

        # Export the model in a TorchScript representation which Elasticsearch uses
        tmp_path = "models"
        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        model_path, config, vocab_path = tm.save(tmp_path)

        # Import model into Elasticsearch
        #client = elasticsearch.Elasticsearch(hosts=[ELASTIC_ADDRESS])
        # Use this instead, IF using SECURITY ENABLED.
        
        client = elasticsearch.Elasticsearch(hosts=[ELASTIC_ADDRESS], ca_certs=CA_CERTS_PATH, basic_auth=("elastic", ELASTIC_PASSWORD))

        ptm = PyTorchModel(client, tm.elasticsearch_model_id())
        ptm.import_model(model_path=model_path, config_path=None, vocab_path=vocab_path, config=config)


if __name__ == "__main__":
    main()
