import torch

from constants import *
from Data import get_batch
from Model import BigramLanguageModel
from Tokenizer import decode


def main() -> None:
    """
    Main function of the program.
    :return:
    """
    m: BigramLanguageModel = BigramLanguageModel()

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer: torch.optim.AdamW = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for it in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = m.estimate_loss()
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

    return


if __name__ == "__main__":
    main()
