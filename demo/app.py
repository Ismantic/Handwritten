"""Tkinter HCCR demo:鼠标手写 → NCNN INT8 实时识别 → top-10 候选栏。

用法:
    cd <repo>
    python demo/app.py
    或:
    make -C demo run

操作:
    左键拖动绘制
    抬笔(release)自动识别,候选栏刷新
    "清空" 按钮 / Esc / Space:清空画板
    候选字点击:打印到 stdout(模拟 IME 上屏)
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path

from PIL import Image, ImageDraw

from demo.inference import HCCRRecognizer


CANVAS_SIZE = 360            # UI 上画板尺寸
STROKE_WIDTH = 18            # 笔画粗细(像素)
TOPK = 10
RECOGNIZE_DELAY_MS = 150     # 抬笔后延迟识别,允许连写下一笔(超时再识别)


class HCCRApp:
    def __init__(self, root: tk.Tk, recognizer: HCCRRecognizer) -> None:
        self.recognizer = recognizer
        self.root = root
        root.title("HCCR demo (V2 + INT8 NCNN)")

        # 选 CJK 字体显示候选(Tk 里 Tkinter 默认字体可能不显示中文)
        cjk = self._pick_cjk_font()
        self.cjk_font = tkfont.Font(family=cjk, size=22)
        self.score_font = tkfont.Font(family=cjk, size=10)
        self.label_font = tkfont.Font(family=cjk, size=11)

        # 主布局
        outer = tk.Frame(root, padx=8, pady=8)
        outer.pack(fill="both", expand=True)

        # 画板
        self.canvas = tk.Canvas(
            outer, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", cursor="crosshair", highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # PIL 画布(白底 RGB),与 UI canvas 同步,用于 PIL → numpy 推理
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        # 候选栏 + 状态
        right = tk.Frame(outer)
        right.grid(row=0, column=1, sticky="nw")

        tk.Label(right, text="top-10 候选(点击上屏)", font=self.label_font).pack(anchor="w")
        cand_frame = tk.Frame(right)
        cand_frame.pack(anchor="w", pady=(4, 6))

        self.cand_buttons: list[tk.Button] = []
        for i in range(TOPK):
            col = tk.Frame(cand_frame, padx=2)
            col.pack(side="left")
            b = tk.Button(
                col, text=" ", width=2, font=self.cjk_font,
                command=lambda idx=i: self._on_pick(idx),
            )
            b.pack()
            score = tk.Label(col, text="", font=self.score_font, fg="#888")
            score.pack()
            b._score_label = score   # type: ignore[attr-defined]
            self.cand_buttons.append(b)

        # 控制按钮
        ctrl = tk.Frame(right)
        ctrl.pack(anchor="w")
        tk.Button(ctrl, text="清空 (Esc)", command=self._clear, font=self.label_font).pack(side="left")
        self.status = tk.Label(right, text="(尚未识别)", font=self.label_font, fg="#666")
        self.status.pack(anchor="w", pady=(8, 0))

        tk.Label(
            outer,
            text=(
                "左键拖动写字 · 抬笔自动识别 · 点击候选字会打印到终端\n"
                f"模型:{Path(self.recognizer.net.param_path if hasattr(self.recognizer.net, 'param_path') else '').name or 'NCNN INT8'}    "
                f"类数:{self.recognizer.num_classes}"
            ),
            font=self.label_font,
            fg="#666",
            justify="left",
        ).grid(row=1, column=1, sticky="sw")

        root.bind("<Escape>", lambda _: self._clear())
        root.bind("<space>", lambda _: self._clear())

        # 笔画状态
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._pending_after: str | None = None

    @staticmethod
    def _pick_cjk_font() -> str:
        """从 Tk 可用字体里选支持 CJK 的。Linux 上 Noto Sans CJK 通常装了。"""
        candidates = [
            "Noto Sans CJK SC", "Noto Sans CJK", "Noto Serif CJK SC",
            "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans CN",
            "SimHei", "Microsoft YaHei", "AR PL UMing CN",
        ]
        available = set(tkfont.families())
        for c in candidates:
            if c in available:
                return c
        return "TkDefaultFont"

    # ---------- 笔画事件 ----------

    def _on_press(self, event: tk.Event) -> None:
        self._last_x, self._last_y = event.x, event.y
        # 单点也画一个小圆,否则点击不出现痕迹
        r = STROKE_WIDTH // 2
        self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill="black", outline="black",
        )
        self.draw.ellipse(
            (event.x - r, event.y - r, event.x + r, event.y + r),
            fill=0,
        )
        if self._pending_after is not None:
            self.root.after_cancel(self._pending_after)
            self._pending_after = None

    def _on_drag(self, event: tk.Event) -> None:
        if self._last_x is None:
            return
        self.canvas.create_line(
            self._last_x, self._last_y, event.x, event.y,
            fill="black", width=STROKE_WIDTH, capstyle="round", smooth=True,
        )
        self.draw.line(
            [(self._last_x, self._last_y), (event.x, event.y)],
            fill=0, width=STROKE_WIDTH, joint="curve",
        )
        self._last_x, self._last_y = event.x, event.y

    def _on_release(self, _event: tk.Event) -> None:
        self._last_x = self._last_y = None
        # 延迟识别,如果用户继续下笔,取消重排
        if self._pending_after is not None:
            self.root.after_cancel(self._pending_after)
        self._pending_after = self.root.after(RECOGNIZE_DELAY_MS, self._recognize)

    # ---------- 识别 ----------

    def _recognize(self) -> None:
        self._pending_after = None
        try:
            candidates = self.recognizer.predict(self.image, k=TOPK)
        except Exception as e:
            self.status.config(text=f"err: {e}", fg="red")
            return
        if not candidates:
            self.status.config(text="(画布空)", fg="#666")
            for b in self.cand_buttons:
                b.config(text=" ", state="disabled")
                b._score_label.config(text="")  # type: ignore[attr-defined]
            return
        for i, b in enumerate(self.cand_buttons):
            if i < len(candidates):
                ch, p = candidates[i]
                b.config(text=ch, state="normal")
                b._score_label.config(text=f"{p:.0%}")  # type: ignore[attr-defined]
            else:
                b.config(text=" ", state="disabled")
                b._score_label.config(text="")  # type: ignore[attr-defined]
        top1, p1 = candidates[0]
        self.status.config(text=f"top-1: {top1} ({p1:.1%})", fg="#000")

    # ---------- 控件 ----------

    def _on_pick(self, idx: int) -> None:
        ch = self.cand_buttons[idx].cget("text")
        if ch.strip():
            print(ch, flush=True)
            self._clear()

    def _clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)
        for b in self.cand_buttons:
            b.config(text=" ", state="disabled")
            b._score_label.config(text="")  # type: ignore[attr-defined]
        self.status.config(text="(画板已清空)", fg="#666")
        if self._pending_after is not None:
            self.root.after_cancel(self._pending_after)
            self._pending_after = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ncnn-param",
        type=Path,
        default=Path("save/output/mbv2_aug.int8.ncnn.param"),
    )
    ap.add_argument(
        "--ncnn-bin",
        type=Path,
        default=Path("save/output/mbv2_aug.int8.ncnn.bin"),
    )
    ap.add_argument(
        "--charset",
        type=Path,
        default=Path("data/processed/charset.json"),
    )
    args = ap.parse_args()

    if not args.ncnn_param.exists():
        print(f"模型 param 找不到:{args.ncnn_param}", file=sys.stderr)
        print("先跑 `make -C save all` 生成 NCNN 模型", file=sys.stderr)
        sys.exit(1)

    rec = HCCRRecognizer(args.ncnn_param, args.ncnn_bin, args.charset)
    print(f"[demo] loaded NCNN INT8 model, {rec.num_classes} classes", flush=True)

    root = tk.Tk()
    app = HCCRApp(root, rec)
    root.mainloop()


if __name__ == "__main__":
    main()
