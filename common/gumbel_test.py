import torch
import gumbel_utils as gu
probs = torch.tensor([0.8,0.2])
probs=probs.unsqueeze(0)
logits = torch.log(probs)
print('Probabilities: ' + str(probs.numpy()))
print('Samples with t=10:')
for i in range(10):
    samples = gu.gumbel_softmax_sample(logits, 10)
    print(samples)

print('Samples with t=0.1:')
for i in range(10):
    samples = gu.gumbel_softmax_sample(logits, 0.1)
    print(samples)