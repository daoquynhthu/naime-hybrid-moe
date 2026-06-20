import torch

from naime_hybrid.modules.attention import AttentionKVCache, GQAAttention, MLAAttention


def _assert_cached_attention_matches_full_sequence(attn_cls, **kwargs):
    torch.manual_seed(2050)
    attn = attn_cls(
        dropout=0.0,
        qk_norm=True,
        max_seq_len=32,
        **kwargs,
    ).eval()
    hidden = torch.randn(2, 9, kwargs["d_model"])

    with torch.no_grad():
        full = attn(hidden)
        assert isinstance(full, torch.Tensor)

        first, cache = attn(hidden[:, :4, :], use_cache=True)
        assert isinstance(first, torch.Tensor)
        assert isinstance(cache, AttentionKVCache)
        assert cache.key.shape[:3] == (2, kwargs["n_kv_heads"], 4)
        assert cache.value.shape == cache.key.shape

        second, cache = attn(hidden[:, 4:, :], past_key_value=cache, use_cache=True)
        assert isinstance(second, torch.Tensor)
        assert isinstance(cache, AttentionKVCache)
        assert cache.seq_len == hidden.size(1)

        cached = torch.cat([first, second], dim=1)

    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)


def test_v8_prereq_gqa_attention_kv_cache_matches_full_sequence():
    _assert_cached_attention_matches_full_sequence(
        GQAAttention,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
    )


def test_v8_prereq_mla_attention_kv_cache_matches_full_sequence():
    _assert_cached_attention_matches_full_sequence(
        MLAAttention,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_latent=12,
        d_rope_per_head=4,
    )


def test_v8_prereq_attention_cache_accepts_current_segment_mask():
    torch.manual_seed(2051)
    attn = GQAAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.0,
    ).eval()
    hidden = torch.randn(2, 7, 32)
    current_mask = torch.ones(2, 3, dtype=torch.bool)
    current_mask[1, -1] = False

    with torch.no_grad():
        _, cache = attn(hidden[:, :4, :], use_cache=True)
        out, next_cache = attn(
            hidden[:, 4:, :],
            attention_mask=current_mask,
            past_key_value=cache,
            use_cache=True,
        )

    assert out.shape == (2, 3, 32)
    assert isinstance(next_cache, AttentionKVCache)
    assert next_cache.seq_len == hidden.size(1)


def test_v8_prereq_attention_cache_validates_committed_memory_shape():
    attn = GQAAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.0,
    ).eval()
    hidden = torch.randn(2, 4, 32)
    bad_cache = AttentionKVCache(
        key=torch.randn(3, 2, 2, 8),
        value=torch.randn(3, 2, 2, 8),
    )

    try:
        attn(hidden, past_key_value=bad_cache)
    except ValueError as exc:
        assert "batch mismatch" in str(exc)
    else:
        raise AssertionError("expected cache batch mismatch to raise ValueError")
