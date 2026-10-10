**5 min** of optional download setup: run [embedding_download.py] to get the pretrained embeddings as headstart. Otherwise remove #6 and update #8.


Here's the pipeline:

cnn.py is the main script

It:
1. loads the csvs
2. mapped the classes (models) to numbers, ie. claude -> 0, chatgpt -> 1
3. takes the response column of the dataset, turns into list (because the tokenizer implementation im using uses a list, could change the implementation)
4. tokenize responses [tokenize_encode.py]
5. encodes tokenized responses [tokenize_encode.py]
6. loads up embeddings [load_embeddingss.py]
7. creates dataloaders (supposedly optimizes speed/mem, not sure what it does) [dataloader.py]

8. calls initilize_model() from [initialize_model.py], which uses [conv.py]'s CNN model (which is itself made from pytorch's conv models)

9. Using everything we built up so far, train, and evaluate [train.py]