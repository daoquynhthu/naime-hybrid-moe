import torch

from naime_hybrid import NAIMEStateMoEConfig, build_model


def main() -> None:
    config = NAIMEStateMoEConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=64,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        stride=4,
        window=8,
        z_dim=16,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=96,
    )
    input_ids = torch.randint(1, config.vocab_size, (2, 17))

    for architecture in [
        "dense",
        "token_moe",
        "naime_state_moe",
        "naime_v4_state_moe",
        "naime_v41_state_moe",
        "naime_v42_state_moe",
    ]:
        model = build_model(architecture, config)
        out = model(input_ids)
        print(f"[{architecture}] logits:", tuple(out["logits"].shape))
        if architecture == "dense":
            continue
        last_aux = out["aux"][-1]
        moe = last_aux["moe"]
        print(f"[{architecture}] topk:", tuple(moe["topk_indices"].shape))
        print(f"[{architecture}] expert_load:", moe["token_load"].detach().tolist())
        if "semantic" in last_aux:
            semantic = last_aux["semantic"]
            print(f"[{architecture}] z:", tuple(semantic["z"].shape))
            print(f"[{architecture}] alpha_mean:", float(semantic["alpha"].detach().float().mean()))


if __name__ == "__main__":
    main()
