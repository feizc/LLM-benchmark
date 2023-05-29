from transformers import AutoTokenizer, AutoModelForCausalLM 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(): 
    model_path = './ckpt'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path) 

    


if __name__ == "__main__":
    main() 
