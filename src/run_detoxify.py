from pathlib import Path
import pandas as pd
from pyspark.sql.functions import col, pandas_udf, from_unixtime, year, month, length
from pyspark.sql.types import StringType

from detoxify import Detoxify

from src.utils.data_loader import get_spark_session, load_data, ROOT

INPUT_PATH  = ROOT / "data" / "processed" / "messages_preprocessed"
OUTPUT_PATH = ROOT / "data" / "processed" / "toxicity"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# class DetoxifyClient:
#     def __inti__(self, model_type='original', device='cpu'):
#         print(f"Loading Detoxify {model_type} model on {device}...")
#         self.model = Detoxify(model_type, device=device)

#     def analyze_text(self,text):
#         predictions = self.model.predict(text)

def analyze_partition_iterator(iterator, batch_size=256):
    import torch
    torch.set_num_threads(1)
    
    from detoxify import Detoxify
    import pandas as pd
    import numpy as np

    model = Detoxify('original', device='cpu')

    for pdf in iterator:
        if pdf.empty:
            yield pdf
            continue

        texts = pdf['clean_text'].fillna("").astype(str).tolist()

        results = {
            'toxicity': [], 'severe_toxicity': [], 'obscene': [],
            'threat': [], 'insult': [], 'identity_attack': []
        }

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            if not batch or all(t.strip() == "" for t in batch):
                for k in results.keys():
                    results[k].extend([0.0] * len(batch))
                continue

            predictions = model.predict(batch

            for key in results.keys():
                results[key].extend(predictions[key])
        
        for key, values in results.items():
            pdf[key] = values
            
        yield pdf


if __name__ == "__main__":
    spark = get_spark_session("processDetoxify")
    df = load_data(spark, INPUT_PATH)
    
    #df.show()
    #print(df.count())
    #print(df.select('year').distinct().count())

    print("\nQuantidade de registros por ano:")
    df.groupBy("year").count().orderBy("year").show()

    df_result = df.mapInPandas(analyze_partition_iterator)
    
    