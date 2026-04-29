from tqdm import tqdm
from torch import autograd
import torch
import evaluate

bertscore = evaluate.load("bertscore")
break_symbol = ' [IMAGE] '

class GANImageDescriptionTrainer:
    def __init__(self, discriminator, qwen_model, qwen_tokenizer, image_adapter, device, lr=1e-4, lambda_gp=5):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.image_adapter = image_adapter
        self.device = device
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp
            
        break_tokens = self.qwen_tokenizer(break_symbol, return_tensors='pt')['input_ids'].to(self.device)
        with torch.no_grad():
            self.break_embeddings = qwen_model.model.embed_tokens(break_tokens).to(self.device)

        for param in self.qwen_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.image_adapter.parameters()},
                {'params': self.discriminator.parameters()}
            ],
            lr=lr
        )
        
    def compute_gradient_penalty(self, image_tokens, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        critic_interpolates = self.discriminator(image_tokens.permute((0, 2, 1)), interpolates.permute((0, 2, 1)))
        
        fake_grad_outputs = torch.ones_like(critic_interpolates, device=self.device)
        
        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty
        
    def get_loss(self, image_inputs, texts, val=False, generator=False):
        image_inputs_noisy = torch.concat(
            [
                image_inputs.to(self.device), 
                torch.randn(
                    (
                        image_inputs.shape[0], 
                        image_inputs.shape[1],
                        self.image_adapter.adapter[0].in_channels - image_inputs.shape[2]
                    ),
                    device=self.device
                )
            ],
            dim=2
        ).permute((0, 2, 1))
        
        image_embeddings = self.image_adapter(image_inputs_noisy).permute((0, 2, 1))

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
            padding='max_length',
            max_length=self.discriminator.text_tokens,
            truncation=True
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

        logits = self.qwen_model(
            inputs_embeds=all_embeddings,
            attention_mask=attn_mask,
            labels=targets
        ).logits[:, image_embeddings.shape[1]:]
        
        logits /= logits.sum(dim=-1, keepdim=True)
        
        pred_embeddings = logits @ self.qwen_model.model.embed_tokens.weight
        
        pred_score = self.discriminator(
            image_inputs.permute((0, 2, 1)),
            pred_embeddings.permute((0, 2, 1))
        )
        
        if generator:
            loss = - pred_score.mean()
        else:
            real_score = self.discriminator(
                image_inputs.permute((0, 2, 1)),
                text_embeddings.permute((0, 2, 1))
            )
            
            loss = pred_score.mean() - real_score.mean()
            loss =+ self.lambda_gp * self.compute_gradient_penalty(image_inputs, text_embeddings, pred_embeddings)
        
        if not val:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def run_epoch(self, loader, n_critic=5, val=False):
        generator_loss = critic_loss = 0.0
        generator_examples = critic_examples = 0

        if val:
            self.image_adapter.eval()
            self.encoder.eval()
            with torch.no_grad():
                step = 1
                for batch in tqdm(loader, desc="Validating"):
                    image_inputs, texts = batch
                    
                    if step == n_critic:
                        generator_loss += self.get_loss(image_inputs, texts, generator=True, val=True) * image_inputs.size(0)
                        generator_examples += image_inputs.size(0)
                        step = 1
                    else:
                        critic_loss += self.get_loss(image_inputs, texts, val=True) * image_inputs.size(0)
                        critic_examples += image_inputs.size(0)
                        step += 1

        else:
            step = 1
            for batch in tqdm(loader, desc="Training"):
                image_inputs, texts = batch
                
                if step == n_critic:
                    self.image_adapter.train()
                    self.discriminator.eval()
                    
                    generator_loss += self.get_loss(image_inputs, texts, generator=True, val=False) * image_inputs.size(0)
                    generator_examples += image_inputs.size(0)
                    step = 1
                else:
                    self.image_adapter.eval()
                    self.discriminator.train()
                    
                    critic_loss += self.get_loss(image_inputs, texts, val=False) * image_inputs.size(0)
                    critic_examples += image_inputs.size(0)
                    step += 1
                
        torch.cuda.empty_cache()
        return generator_loss / generator_examples, critic_loss / critic_examples

    def generate(self, image_embeddings, max_tokens=20):
        self.image_adapter.eval()
        self.discriminator.eval()
        with torch.no_grad():
            image_inputs_noisy = torch.concat(
                [
                    image_embeddings.to(self.device), 
                    torch.randn(
                        (
                            image_embeddings.shape[0], 
                            image_embeddings.shape[1],
                            self.image_adapter.adapter[0].in_channels - image_embeddings.shape[2]
                        ),
                        device=self.device
                    )
                ],
                dim=2
            ).permute((0, 2, 1))
            
            previous_embeddings = self.image_adapter(image_inputs_noisy).permute((0, 2, 1))
    
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
