"""Hugging Face 发布清单。发布源只有这里一份,避免脚本间路径和指标漂移。"""

RELEASES = {
    "Handwritten": {
        "checkpoint": "runs/mbv2_phase1_aug/best.pt",
        "ncnn_param": "save/output/mbv2_aug.int8.ncnn.param",
        "ncnn_bin": "save/output/mbv2_aug.int8.ncnn.bin",
        "charset": "data/processed/charset.json",
        "model": "mobilenet_v2",
        "classes": 3755,
        "metrics": {
            "top1": 0.9547,
            "top10": 0.9958,
        },
        "source": "https://github.com/Ismantic/Handwritten",
    },
}
