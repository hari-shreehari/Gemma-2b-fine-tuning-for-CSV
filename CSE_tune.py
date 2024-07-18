import csv
import keras
import keras_nlp
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

# Load and parse the CSV file
data = []
input_file = 'CSE_Dataset.csv'

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    header=next(reader)
    header=next(reader)
    for row in reader:
        template = "Question:\n{qns}\n\nAnswer:\n{ans}\n\nOptions:\n{opt}"
        data.append(template.format(qns=row["Question"], ans=row["Answer"], opt=[row["Option 1"], row["Option 2"], row["Option 3"], row["Option 4"]]))
data=data[:6]  
def preprocess_data(data, tokenizer, max_seq_length):
    preprocessed_data = []
    for sample in data:
        question_tokens = tokenizer.tokenize(sample[0])
        answer_tokens = tokenizer.tokenize(sample[1])
        options_tokens = [tokenizer.tokenize(option) for option in sample[2]]

        input_sequence = question_tokens + answer_tokens + options_tokens[0]

        input_sequence = [tokenizer.cls_token] + input_sequence + [tokenizer.sep_token]
        
        input_ids = tokenizer.convert_tokens_to_ids(input_sequence)

        input_tensor = pad_sequences([input_ids], maxlen=max_seq_length, padding='post')[0]
        preprocessed_data.append(input_tensor)

    preprocessed_data = np.array(preprocessed_data)
    return preprocessed_data

# Set environment variables
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Enable LoRA
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 512
# AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)

optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=1, batch_size=1)


# Save the fine-tuned model
gemma_lm.save("FineTuned.keras") #can be .keras or .h5
