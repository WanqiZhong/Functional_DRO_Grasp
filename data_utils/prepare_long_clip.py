import torch
from transformers import CLIPTokenizer, CLIPTextModel
import clip


# 输入句子
input_sentence = "Handover a wrench. Thumb, index finger, middle finger, ring finger and pinky finger are contacting the wrench head."
def huggingface_clip():

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CLIP 的 tokenizer 和文本模型（以 openai/clip-vit-base-patch32 为例）
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()

    # 对输入文本进行 tokenize，确保序列长度为 77（padding 与 truncation 均生效）
    inputs = tokenizer(
        input_sentence,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 将输入转移到设备上
    input_ids = inputs["input_ids"].to(device)         # 形状：[batch, 77]
    attention_mask = inputs["attention_mask"].to(device) # 形状：[batch, 77]，1 表示真实 token，0 表示 PAD

    with torch.no_grad():
        # 获取 CLIP 文本模型的输出，last_hidden_state 的形状为 [batch, seq_len, hidden_size]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # 这里的 hidden_size 为 768

    print("Tokenized Text:", input_ids)
    print("Embedding Shape:", last_hidden_state.shape)  # (1, 77, 768)
    print("Argmx:", input_ids.argmax(dim=-1))
    print("Embedding:", last_hidden_state[0][input_ids.argmax(dim=-1)])
    print("Attention Mask:", attention_mask)

def openai_clip():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, download_root="/data/zwq/code/CLIP")
    model.eval()

    tokenized_text = clip.tokenize(input_sentence).to(device)
    mask = (tokenized_text != 0)
    with torch.no_grad():
        token_embeddings = model.token_embedding(tokenized_text)
        cls_embeddings = model.encode_text(tokenized_text)

    print("Tokenized Text:", tokenized_text)
    print("Embedding Shape:", token_embeddings.shape)  # (77, 512)
    print("CLS Embedding:", cls_embeddings)
    print("Embedding:", token_embeddings[0])
    print("Mask:", mask)


if __name__ == "__main__":
    huggingface_clip()
    print("="*50)
    openai_clip()