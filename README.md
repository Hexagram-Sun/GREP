Codes for the GREP framework. Put your wiki dump and hotpotqa data in `../datasets` before running anything.

Run `bash scripts/train.sh` to preprocess the data and build the FAISS/BM25 indices. Once that's done you can run `bash scripts/test.sh` to evaluate the retrieval performance. The main pipeline logic is in `src/pipeline.py` and the DPO training code is in `src/model/dpo_trainer.py`. Note that you might need to change model and dataset paths if your data layout is different.
