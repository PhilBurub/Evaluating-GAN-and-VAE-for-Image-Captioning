from tqdm import tqdm
import torch

break_symbol = ' [IMAGE] '

class VAEImageDescriptionTrainer:
    def __init__(self, vae_encoder, encoder_dim, qwen_model, qwen_tokenizer, image_adapter, device, lr=1e-4, kl_coef=0.1):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.image_adapter = image_adapter
        self.device = device
        self.encoder = vae_encoder
        self.encoder_dim = encoder_dim
        self.kl_coef = kl_coef
            
        break_tokens = self.qwen_tokenizer(break_symbol, return_tensors='pt')['input_ids'].to(self.device)
        with torch.no_grad():
            self.break_embeddings = qwen_model.model.embed_tokens(break_tokens).to(self.device)

        for param in self.qwen_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.image_adapter.parameters()},
                {'params': self.encoder.parameters()}
            ],
            lr=lr
        )
    
    def encode_texts(self, images, texts):
        qwen_tokens = self.qwen_tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            max_length=self.encoder.text_tokens,
            truncation=True
        ).to(self.device)
        
        text_embeddings = self.qwen_model.model(**qwen_tokens).last_hidden_state
        
        mu, log_var = self.encoder(images.permute((0, 2, 1)), text_embeddings.permute((0, 2, 1)))
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu, device=self.device)
        return eps * std + mu, mu, log_var
        
    def get_loss(self, image_inputs, texts, val=False):
        
        noise_sampled, mu, log_var = self.encode_texts(image_inputs, texts)
        
        kld_loss = - self.kl_coef * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        
        image_inputs = torch.concat(
            [
                image_inputs.to(self.device), 
                noise_sampled.unflatten(1, (image_inputs.shape[1], self.encoder_dim))
            ],
            dim=2
        ).permute((0, 2, 1))
        
        image_embeddings = self.image_adapter(image_inputs).permute((0, 2, 1))

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
        ).loss + kld_loss
        
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
            self.encoder.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating"):
                    image_inputs, texts = batch
                    total_loss += self.get_loss(image_inputs, texts, val=True) * image_inputs.size(0)
                    total_examples += image_inputs.size(0)

        else:
            self.image_adapter.train()
            self.encoder.train()
            for batch in tqdm(loader, desc="Training"):
                image_inputs, texts = batch
                total_loss += self.get_loss(image_inputs, texts, val=False) * image_inputs.size(0)
                total_examples += image_inputs.size(0)
                
        torch.cuda.empty_cache()
        return total_loss / total_examples

    def generate(self, image_embeddings, fixed_noise=None, max_tokens=20):
        self.image_adapter.eval()
        self.encoder.eval()
        with torch.no_grad():
            if fixed_noise is None:
                noise_sampled = torch.randn(
                    (
                        image_embeddings.shape[0],
                        image_embeddings.shape[1],
                        self.encoder_dim
                    ),
                    device=self.device
                )
            else:
                noise_sampled = torch.full(
                    (
                        image_embeddings.shape[0],
                        image_embeddings.shape[1],
                        self.encoder_dim
                    ),
                    fill_value=fixed_noise,
                    device=self.device
                )
            
            image_inputs = torch.concat(
                [
                    image_embeddings.to(self.device), 
                    noise_sampled
                ],
                dim=2
            ).permute((0, 2, 1))
            
            
            previous_embeddings = self.image_adapter(image_inputs).permute((0, 2, 1))
    
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
    
    def run_test(self, loader):
        out_df = {
            'references': [],
            'predictions': []
        }

        for batch in tqdm(loader, desc="Testing"):
            image_inputs, references = batch
            predictions = self.generate(image_embeddings=image_inputs, max_tokens=100)
            out_df['references'].extend(references)
            out_df['predictions'].extend(predictions)
        return out_df
