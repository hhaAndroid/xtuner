
if __name__ == "__main__":
    import torch
    device = "cuda:0"
    batch = 1
    seqlen=16
    seqlens = []
    for b in range(batch):
        nsplits = torch.randint(1, 5, (1,)).item()
        eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
        seqlens.append(torch.diff(torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])).tolist())
        assert sum(seqlens[-1]) == seqlen
        assert all(s > 0 for s in seqlens[-1])
    print(seqlens)
    # seqlens = [3, 5, 2]

    seq_idx= torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(seqlens[0])], dim=0)[None]
    print(seq_idx,seq_idx.shape)
