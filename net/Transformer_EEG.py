import tensorflow.keras as kr
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model


def net(config):
    """Transformer-based EEG classifier.

    Expects input shaped as (CH, T, 1), keeps preprocessing unchanged.
    Tokenizes along time with patching, applies TransformerEncoder layers,
    global pooling, then Dense(2)+softmax output.
    """

    # Required fields are present across existing models
    assert (hasattr(config, 'data_format') and
            hasattr(config, 'fs') and
            hasattr(config, 'frame') and
            hasattr(config, 'CH') and
            hasattr(config, 'dropoutRate'))

    # Defaults pulled via getattr so DL_config.py need not change
    patch_size = getattr(config, 'transformer_patch_size', 16)
    patch_stride = getattr(config, 'transformer_patch_stride', 8)
    embed_dim = getattr(config, 'transformer_embed_dim', 64)
    num_heads = getattr(config, 'transformer_num_heads', 4)
    ff_dim = getattr(config, 'transformer_ff_dim', 128)
    num_layers = getattr(config, 'transformer_num_layers', 2)
    dropout_rate = getattr(config, 'transformer_dropout_rate', config.dropoutRate)

    # Input alignment: (CH, T, 1)
    input_shape = (config.CH, config.frame * config.fs, 1)
    i = Input(shape=input_shape)

    # Remove singleton channel dimension -> (CH, T)
    x = layers.Lambda(lambda t: kr.backend.squeeze(t, axis=-1))(i)
    # Permute to (T, CH) so time is sequence axis
    x = layers.Permute((2, 1))(x)

    # Project channels to embedding dim and perform time patching via Conv1D with stride
    # This acts as tokenization along the time axis
    # Output: (T_tokens, embed_dim)
    reg = {}
    if hasattr(config, 'l2'):
        reg = {
            'kernel_regularizer': kr.regularizers.l2(config.l2),
            'bias_regularizer': kr.regularizers.l2(config.l2)
        }

    x = layers.Conv1D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_stride,
        padding='valid',
        **reg
    )(x)

    # Positional embedding (learned)
    # Compute token length dynamically
    def get_len(t):
        return kr.backend.shape(t)[-2]

    def tile_positional_embedding(seq, pe):
        # pe shape: (1, max_len, embed_dim), seq: (B, L, D) -> slice to L
        l = kr.backend.shape(seq)[1]
        return pe[:, :l, :]

    max_pos_len = getattr(config, 'transformer_max_pos_len', 2048)
    pos_emb = layers.Embedding(
        input_dim=max_pos_len,
        output_dim=embed_dim
    )

    # Create positions [0..L-1]
    positions = layers.Lambda(lambda t: kr.backend.arange(0, get_len(t)))(x)
    positions = layers.Lambda(lambda p: kr.backend.expand_dims(p, axis=0))(positions)
    positions = layers.Lambda(lambda p: kr.backend.tile(p, (kr.backend.shape(x)[0], 1)))(positions)
    p = pos_emb(positions)
    x = layers.Add()([x, p])

    x = layers.Dropout(dropout_rate)(x)

    # Transformer Encoder blocks
    for _ in range(num_layers):
        # Self-attention
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(x, x)
        attn_out = layers.Dropout(dropout_rate)(attn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn_out]))

        # Feed-forward
        ffn = layers.Dense(ff_dim, activation='gelu', **reg)(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(embed_dim, **reg)(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ffn]))

    # Global average pooling over tokens
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head: 2-class softmax
    logits = layers.Dense(2, **reg)(x)
    out = layers.Activation('softmax')(logits)

    return Model(inputs=i, outputs=out)


