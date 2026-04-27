from tqdm import tqdm
import torch
import evaluate

bertscore = evaluate.load("bertscore")
break_symbol = ' [IMAGE] '

class QwenImageDescriptionTrainer:
    def __init__(self, qwen_model, qwen_tokenizer, image_adapter, device, lr=1e-4):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.image_adapter = image_adapter
        self.device = device
        
        break_tokens = self.qwen_tokenizer(break_symbol, return_tensors='pt')['input_ids'].to(self.device)
        
        with torch.no_grad():
            self.break_embeddings = qwen_model.model.embed_tokens(break_tokens).to(self.device)

        for param in self.qwen_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.image_adapter.parameters(),
            lr=lr
        )

    def get_loss(self, image_inputs, texts, val=False):
        image_embeddings = self.image_adapter(
            image_inputs.permute((0, 2, 1)).to(self.device)
        ).permute((0, 2, 1))

        image_embeddings = torch.concat(
            [
              image_embeddings,
              self.break_embeddings.detach().repeat(image_embeddings.shape[0], 1, 1)
            ],
            dim=1
        )

        qwen_tokens = self.qwen_tokenizer(
            texts,
            return_tensors='pt',
            padding=True
        ).to(self.device)

        text_embeddings = self.qwen_model.model.embed_tokens(
            qwen_tokens['input_ids']
        )

        all_embeddings = torch.cat(
            [
                image_embeddings,
                text_embeddings
            ],
            dim=1
        )

        attn_mask = torch.cat(
            [
                torch.ones(image_embeddings.shape[:2], device=self.device),
                qwen_tokens['attention_mask']
            ],
            dim=1
        )

        targets = torch.cat(
            [
                torch.full(image_embeddings.shape[:2], -100, device=self.device),
                torch.where(
                    qwen_tokens['attention_mask']==1,
                    qwen_tokens['input_ids'],
                    -100
                )
            ],
            dim=1
        ).to(torch.int64)

        loss = self.qwen_model(
            inputs_embeds=all_embeddings,
            attention_mask=attn_mask,
            labels=targets
        ).loss
        
        if not val:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def run_epoch(self, loader, val=False):
        total_loss = 0.0
        total_examples = 0

        if val:
            self.image_adapter.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating"):
                    image_inputs, texts = batch
                    total_loss += self.get_loss(image_inputs, texts, val=True) * image_inputs.size(0)
                    total_examples += image_inputs.size(0)

        else:
            self.image_adapter.train()
            for batch in tqdm(loader, desc="Training"):
                image_inputs, texts = batch
                total_loss += self.get_loss(image_inputs, texts, val=False) * image_inputs.size(0)
                total_examples += image_inputs.size(0)
                
        torch.cuda.empty_cache()
        return total_loss / total_examples

    def generate(self, image_embeddings, max_tokens=20):
        self.image_adapter.eval()
        with torch.no_grad():            
            previous_embeddings = self.image_adapter(
                image_embeddings.permute((0, 2, 1))
            ).permute((0, 2, 1))
    
            previous_embeddings = torch.concat(
                [
                  previous_embeddings,
                  self.break_embeddings.detach().repeat(previous_embeddings.shape[0], 1, 1).to(self.device)
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
    
    def evaluate(self, loader):
        out_df = {
            'references': [],
            'predictions': [],
            'bert_scores': []
        }

        for batch in tqdm(loader, desc="Testing"):
            image_inputs, references = batch
            predictions = self.generate(image_embeddings=image_inputs, max_tokens=100)
            bert_scores = bertscore.compute(
                predictions=predictions,
                references=references,
                device=self.device,
                lang="en"
            )['f1']
            
            out_df['references'].extend(references)
            out_df['predictions'].extend(predictions)
            out_df['bert_scores'].extend(bert_scores)

        return out_df
