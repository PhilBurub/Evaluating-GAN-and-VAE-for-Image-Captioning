from tqdm import tqdm
import torch

break_symbol = ' [IMAGE] '

class SoftPromptTrainer:
    def __init__(self, trainable_tokens, qwen_model, qwen_tokenizer, device, lr=1e-4):
        self.trainable_tokens = trainable_tokens
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.lr = lr
        self.device = device
        
        break_tokens = self.qwen_tokenizer(break_symbol, return_tensors='pt')['input_ids'].to(self.device)
        
        with torch.no_grad():
            self.break_embeddings = qwen_model.model.embed_tokens(break_tokens).to(self.device)

        for param in self.qwen_model.parameters():
            param.requires_grad = False

    def get_soft_prompts(self, texts):

        qwen_tokens = self.qwen_tokenizer(
            texts,
            return_tensors='pt',
            padding=True
        ).to(self.device)

        text_embeddings = self.qwen_model.model.embed_tokens(
            qwen_tokens['input_ids']
        )

        fixed_embeddings = torch.cat(
            [
                self.break_embeddings.detach().repeat(len(texts), 1, 1),
                text_embeddings
            ],
            dim=1
        )

        attn_mask = torch.cat(
            [
                torch.ones((len(texts), self.trainable_tokens), device=self.device),
                qwen_tokens['attention_mask']
            ],
            dim=1
        )

        targets = torch.cat(
            [
                torch.full((len(texts), self.trainable_tokens + 1), -100, device=self.device),
                torch.where(
                    qwen_tokens['attention_mask']==1,
                    qwen_tokens['input_ids'],
                    -100
                )
            ],
            dim=1
        ).to(torch.int64)
        
        trainable_emeddings = torch.randn(
            (
                len(texts),
                self.trainable_tokens,
                self.qwen_model.config.hidden_size
            ),
            device=self.device,
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam(
            trainable_emeddings,
            lr=self.lr
        )
        
        cur_loss = None
        while True:
            all_embeddings = torch.cat(
                [trainable_emeddings, fixed_embeddings],
                dim=1
            )
            
            loss = self.qwen_model(
                inputs_embeds=all_embeddings,
                attention_mask=attn_mask,
                labels=targets
            ).loss
            
            if cur_loss is not None and cur_loss - loss.item() < 1e-4:
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return trainable_emeddings.detach().cpu()
    
    def generate(self, trainable_emeddings, max_tokens=20):
        with torch.no_grad():
            previous_embeddings = torch.concat(
                [
                  trainable_emeddings,
                  self.break_embeddings.detach().repeat(trainable_emeddings.shape[0], 1, 1).to(self.device)
                ],
                dim=1
            )
    
            attn_mask = torch.ones(previous_embeddings.shape[:2], device=self.device)
    
            fake_input_ids = torch.full(
                previous_embeddings.shape[:2],
                self.qwen_tokenizer.pad_token_id,
                device=self.device
            ).to(torch.int64)
    
            generated_ids = self.qwen_model.generate(
                input_ids=fake_input_ids,
                inputs_embeds=previous_embeddings,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
            )
    
            gen_text_ids = generated_ids[:, previous_embeddings.shape[1]:]

        return self.qwen_tokenizer.batch_decode(
            gen_text_ids,
            skip_special_tokens=True
        )
    
    def run_test(self, loader):
        out_df = {
            'references': [],
            'predictions': []
        }

        for batch in tqdm(loader, desc="Testing"):
            trainable_emeddings, references = batch
            predictions = self.generate(trainable_emeddings, max_tokens=100)
            out_df['references'].extend(references)
            out_df['predictions'].extend(predictions)
        return out_df

