_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

# model settings
model = dict(
    pretrained='/home/ma-user/work/bowen/mmsegmentation-0621/pretrain/mit_b1_mm.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(
        type='SegDeformerHead3',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        num_heads=1,
        att_type="SelfAttention",
        trans_with_mlp=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
data = dict(samples_per_gpu=2, workers_per_gpu=2)
