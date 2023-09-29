from typing import (List, Type, Optional)
from cog import BasePredictor, Input, Path, BaseModel, File
import os
import json
from transformers import AutoTokenizer, AutoModel
from optimum.bettertransformer import BetterTransformer
from collections import defaultdict
import torch
import time

current_script_directory = os.path.dirname(os.path.abspath(__file__))
cache_directory = os.path.join(current_script_directory, 'models')
os.environ['TRANSFORMERS_CACHE'] = cache_directory

INGEST_PATH = os.path.join(current_script_directory, "ingest_temp")
OUT_PATH ="/tmp"
CONTEXT_WINDOW_LIMIT = 512
MAX_CONCURRENT_DOWNLOADS = 10  # You can adjust this based on your 

async def fetch_content(session, url):
    async with session.get(url) as response:
        return await response.text()

class Output(BaseModel):
    query_embeddings: List[List[float]]
    extra_metrics: str

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.init_time = time.time()
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model = BetterTransformer.transform(self.model)
        self.model.to(self.device)
        self.instruction = "Represent this sentence for searching relevant passages:"
        
    #split up the array of embedding requests into batches that fit the max_token size
    def split_to_max(self, passages = {}, max_tokens=20000):
        batches = [[]]
        files= [[]]
        idx = 0
        sub_idx = 0
        current_len = 0
        current_max = 0
        for filename, original_strings in passages.items():
            for s in original_strings:
                next_tokens = self.tokenizer.tokenize(s)[:CONTEXT_WINDOW_LIMIT]      
                current_max = max(len(next_tokens), current_max)
                current_len = (sub_idx*current_max) + current_max
                if current_len + len(next_tokens) > max_tokens:
                    batches.append([])
                    files.append([])
                    current_len = len(next_tokens)
                    current_max = current_len
                    sub_idx = 0
                    idx += 1 
                batches[idx].append(s)
                files[idx].append(filename)
                sub_idx +=1

        tokenized_list = []
        for ss in batches: 
            tokenized_list.append(self.tokenizer(ss, padding=True, truncation=True, return_tensors='pt'))

        return tokenized_list, files, len(batches)
    
    def get_batched_embeddings(self, encoded_input):
        encoded_input.to(self.device)
        print("processing batch")
        with torch.no_grad():
            model_output = self.model(**encoded_input)                
            return model_output[0][:, 0]

    def process(self, tokens_lists, filenames, normalize):
        total_infr_time = 0
        file_embeddings = defaultdict(list)
        for i, batch in enumerate(tokens_lists):
            infrstart =time.time()
            embedding_result = self.get_batched_embeddings(encoded_input=batch)
            if(normalize):
                embedding_result = torch.nn.functional.normalize(embedding_result, p=2, dim=1)
            total_infr_time += time.time() - infrstart
            emb_as_list = embedding_result.tolist()
            for j, passagevec in enumerate(emb_as_list):
                file_embeddings[filenames[i][j]].append(passagevec)
        return file_embeddings, total_infr_time, embedding_result.dtype, embedding_result.device
    
    def predict(
        self,
        query_texts: str = Input(default="[]", description="A serialized JSON array of strings you wish to generate *retreival* embeddings for. (note, that you should keep this list short to avoid Replicate response size limitations). Use this to embed short text queries intended for comparison against document text. A vector will be returned corresponding to each line of text in the input array (in order of input). This endpoint will automatically format your query strings for retrieval, you do not need to preprocess them."),
        normalize: bool = Input(default=True, description="normalizes returned embedding vectors to a magnitude of 1. (default: true, as this model presumes cosine similarity comparisons downstream)"),
        batchtoken_max: float = Input(default=200, ge=CONTEXT_WINDOW_LIMIT/1024, description="You probably don't need to worry about this parameter if you're just getting the embeddings for a handful of queries. This parameter sets the maximumum number of kibiTokens (1 kibiToken = 1024 tokens) to try to stuff into a batch (to avoid out of memory errors but maximize throughput). If the total number of tokens across the flattened list of requested embeddings exceed this value, the list will be split internally and run across multiple forward passes. This will not affect the shape of your output, just the time it takes to run."),
        precision: str = Input(default="full", choices=["full", "half"], description="numerical precision for inference computations. Either full or half. Defaults to a paranoid value of full. You may want to test if 'half' is sufficient for your needs, though regardless you should probably prefer to use the same precision for querying as you do for archiving.")
    ) -> Output:
        
        compute_start = time.time()
        max_tokens = batchtoken_max*1024
        self.precision = precision
        print("model loaded, starting inference")
        if self.precision == 'half':
            self.model.half()
        else: 
            self.model.float()
        total_infr_time = 0
        dtype = None
        device = self.device
        query_embeddings = {'queries' : [[]]}

        if query_texts is not None:
            text_batches = json.loads(query_texts)
            if len(text_batches) > 0:
                query_texts = {'queries' : [self.instruction + q for q in text_batches]}
                query_tokens, filenames, m_batches_count = self.split_to_max(passages=query_texts, max_tokens=max_tokens)
                query_embeddings, infr_time, dtype, device = self.process(tokens_lists=query_tokens, filenames=filenames, normalize=normalize)
                total_infr_time = total_infr_time + infr_time
        
        compute_millis = time.time() - compute_start
        extra_metrics = json.dumps({'dtype' : f"{dtype}", 
                            'inference_milliseconds': int(total_infr_time*1000), 
                            'compute_milliseconds': int(compute_millis*1000),
                            'device' : f"{device}"})
        query_emb_results = query_embeddings['queries']
        if len(query_emb_results) == 0: 
            query_emb_results = [[0.0]]
        
        output = Output(query_embeddings= query_emb_results,
                        extra_metrics = extra_metrics)
        return output