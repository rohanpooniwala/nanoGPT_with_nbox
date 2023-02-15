import os, random, string, threading, traceback, time
from pydantic import BaseModel

from typing import List

from fastapi import FastAPI
from op_server import get_gpt2_output
from nbox import logger


# create `results` folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

DELIMITER = '~~~~~~~'
app = FastAPI()

### Parameters ###
class Prompt(BaseModel):
    prompt: str
    num_samples=4 # number of samples to draw
    max_new_tokens=100 # number of tokens generated in each sample
    temperature=0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k=200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    max_time=5 # maximum time to wait for the output

### Utils ###
def thread_it(func):
    """
    Decorator to run a function in a separate thread
    :param func: the function to run in a separate thread
    :return: the wrapper function
    """
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=func, args=args, kwargs=kwargs)
        t.setDaemon(True)
        t.start()
    return wrapper

def save_to_file(texts: List[str], filename: str, delimiter: str = DELIMITER):
    """
    Function to save the output to a file
    :param texts: the texts to save
    :param filename: the filename to save to
    :param delimiter: the delimiter to use
    """
    with open('results/'+filename, 'w') as f:
        f.write(delimiter.join(texts))

### Model ###
@thread_it
def get_output_to_file(prompt: Prompt, inference_id: str):
    """
    Function to get the output and save it to a file
    :param prompt: the prompt to use
    :param inference_id: the inference id
    """
    try:
        start_time = time.time()
        print(f'Starting inference {inference_id}: Prompt({prompt.prompt=}, {prompt.num_samples=}, {prompt.max_new_tokens=}, {prompt.temperature=}, {prompt.top_k=})')
        output = get_gpt2_output(
            start=prompt.prompt,
            num_samples=prompt.num_samples,
            max_new_tokens=prompt.max_new_tokens,
            temperature=prompt.temperature,
            top_k=prompt.top_k
        )
        print(f'Inference finished {inference_id}')
        save_to_file(output, inference_id)
        print(f'Saved to file {inference_id}, took {time.time() - start_time} seconds for inference')
    except Exception as e:
        print(f'Inference failed {inference_id}')
        print(traceback.format_exc())
        save_to_file(['Inference failed'], inference_id)

def check_and_get_inference_status(inference_id, delimiter: str = DELIMITER):
    print(f'Checking inference {inference_id}: results/'+inference_id)
    exist = os.path.isfile('results/'+inference_id)
    if exist:
        print(f'Inference finished {inference_id}')
        with open('results/'+inference_id, 'r') as f:
            output = f.read().split(delimiter)
        return output
    else:
        print(f'Inference not finished yet {inference_id}')
        return None

### API ###
@app.get("/")
def read_root():
  return '''<html>Hello World</html>'''

@app.post('/generate')
def generate(prompt: Prompt):
    # Starts the inference and returns the inference_id in json
    inference_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    get_output_to_file(prompt, inference_id)

    # wait for 5 seconds to check if inference is finished
    time.sleep(prompt.max_time)
    output = check_and_get_inference_status(inference_id)
    if output is not None:
        return {'output': output, 'inference_id': inference_id}
    else:
        return {'inference_id': inference_id}

@app.get('/result/{inference_id}')
def get_result(inference_id: str):
    # Returns the inference result in json
    output = check_and_get_inference_status(inference_id)
    if output is not None:
        return {'output': output}
    else:
        return {'output': 'Inference not finished yet'}