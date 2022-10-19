from transformers import Seq2SeqTrainer, AutoTokenizer, AlbertForSequenceClassification

import torch

REGULARIZE = False

class FormalityReward(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_entail = AutoTokenizer.from_pretrained('textattack/albert-base-v2-snli')
        self.model_entail = AlbertForSequenceClassification.from_pretrained('textattack/albert-base-v2-snli')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for param in self.model_entail.parameters():
          param.requires_grad = False
        self.model_entail.to(self.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        gen_outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=56,
                                     output_scores=True, return_dict_in_generate=True)

        self_loss = -(gen_outputs['sequences_scores'])
        self_loss.requires_grad = True
        self_loss = torch.mean(self_loss)
        gen_outputs = gen_outputs['sequences']
        with torch.no_grad():
            gen_outputs = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            input_docs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            if REGULARIZE:
                gold_summaries = self.tokenizer.batch_decode(inputs["decoder_input_ids"], skip_special_tokens=True)

            reward = 0 # formality
            reward.to(self.device)

        loss = torch.sum(self_loss * reward)
        loss += base_loss
        return (loss, outputs) if return_outputs else loss


    def calculate_reward(self, inference_scores, indices, reward_type: str):
        indiv_rewards = []
        for i in range(len(indices) - 1):
            scores = inference_scores[indices[i]:indices[i+1]]
            if reward_type == "entailment_mean":
                indiv_rewards.append(1 - torch.mean(scores[:, 0]))
            elif reward_type == "entailment_max":
                indiv_rewards.append(1 - torch.max(scores[:, 0]))
            elif reward_type == "entailment_min":
                indiv_rewards.append(1 - torch.min(scores[:, 0]))
            elif reward_type == "contradiction_mean":
                indiv_rewards.append(torch.mean(scores[:, 2]))
            elif reward_type == "contradiction_max":
                indiv_rewards.append(torch.max(scores[:, 2]))
            elif reward_type == "contradiction_min":
                indiv_rewards.append(torch.min(scores[:, 2]))

        return torch.mean(torch.FloatTensor(indiv_rewards))


    def inference_score(self, sent_pairs):
        softmax = torch.nn.Softmax(dim=1)
        inputs = self.tokenizer_entail(sent_pairs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model_entail(**inputs)
        logits = outputs.logits

        return softmax(logits)


