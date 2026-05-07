package com.shiyu.hccr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * 笔迹采集 View:白底,接收 touch 画黑笔画,提供 renderToBitmap() 给推理。
 *
 * 关键设计 —— 解耦视觉 ↔ 模型输入:
 *  - DISPLAY_STROKE_RATIO:视觉细笔画(用户看着舒服)
 *  - RENDER_STROKE_RATIO:渲染给模型的 stroke 粗细。匹配 Python demo
 *    (16px / 360 canvas = 4.4%),缩到 64×64 后约 ~2.5px,跟 HWDB 训练分布对齐
 *  - 路径用 lineTo(直线段),跟 Python PIL.ImageDraw.line 一致;不用 Path.quadTo
 *    的贝塞尔平滑,避免跟 HWDB 数位板的直线段训练分布错位
 */
public class HandwritingView extends View {

    public interface StrokeListener {
        void onStrokeStart();
        void onStrokeEnd();
    }

    private static final float DISPLAY_STROKE_RATIO = 0.018f;   // 视觉:画板宽 1.8%(细)
    private static final float RENDER_STROKE_RATIO  = 0.044f;   // 模型:画板宽 4.4%(同 Python)

    private final List<Path> strokes = new ArrayList<>();
    private Path current;
    private final Paint displayPaint = new Paint();
    private final Paint renderPaint  = new Paint();
    private float lastX, lastY;
    private StrokeListener listener;

    public HandwritingView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        initPaint(displayPaint);
        initPaint(renderPaint);
        setBackgroundColor(Color.WHITE);
    }

    private static void initPaint(Paint p) {
        p.setColor(Color.BLACK);
        p.setAntiAlias(true);
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeCap(Paint.Cap.ROUND);
        p.setStrokeJoin(Paint.Join.ROUND);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        if (w > 0) {
            displayPaint.setStrokeWidth(w * DISPLAY_STROKE_RATIO);
            renderPaint.setStrokeWidth(w * RENDER_STROKE_RATIO);
        }
    }

    public void setStrokeListener(StrokeListener l) {
        this.listener = l;
    }

    public void clear() {
        strokes.clear();
        current = null;
        invalidate();
    }

    public void undoLastStroke() {
        if (!strokes.isEmpty()) {
            strokes.remove(strokes.size() - 1);
            invalidate();
        }
    }

    public boolean isEmpty() {
        return strokes.isEmpty();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (Path p : strokes) {
            canvas.drawPath(p, displayPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent ev) {
        float x = ev.getX();
        float y = ev.getY();
        switch (ev.getAction()) {
            case MotionEvent.ACTION_DOWN:
                current = new Path();
                current.moveTo(x, y);
                strokes.add(current);
                lastX = x; lastY = y;
                if (listener != null) listener.onStrokeStart();
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                if (current != null) {
                    // Android MotionEvent 可以 batch 多个采样点,必须用 getHistorical 才不丢
                    // 中间样本(快速滑动时尤其多)。Python Tkinter 没有这个 batching。
                    int n = ev.getHistorySize();
                    for (int i = 0; i < n; i++) {
                        current.lineTo(ev.getHistoricalX(i), ev.getHistoricalY(i));
                    }
                    current.lineTo(x, y);
                    lastX = x; lastY = y;
                    invalidate();
                }
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                if (current != null) {
                    current.lineTo(x, y);
                    invalidate();
                }
                if (listener != null) listener.onStrokeEnd();
                return true;
        }
        return super.onTouchEvent(ev);
    }

    /** 渲染给模型用的 bitmap 长边目标 px(跟 Python demo 360 一致),不一次 bilinear 缩 23x。 */
    private static final int RENDER_TARGET_SIZE = 360;

    /**
     * 渲染给模型的 bitmap:渲染到 360px 目标尺寸(画布矩阵缩放),
     * 不在原始 hi-DPI 画布(1298px)上画完再 bilinear 缩 23x —— 那个 downscale 因子
     * 太大,bilinear 2x2 邻域采样丢大量笔画细节。
     *
     * 等价于"直接在 360 上画 16px stroke",跟 Python PIL.ImageDraw.line 完全一致。
     * 后续 MainActivity 的 bbox 裁切 + 长边缩 56 工作量大幅减少(~6x downscale 而非 23x)。
     */
    public Bitmap renderToBitmap() {
        int viewW = Math.max(1, getWidth());
        int viewH = Math.max(1, getHeight());
        float scale = (float) RENDER_TARGET_SIZE / Math.max(viewW, viewH);
        int dstW = Math.max(1, Math.round(viewW * scale));
        int dstH = Math.max(1, Math.round(viewH * scale));

        Bitmap bmp = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);
        c.drawColor(Color.WHITE);
        c.scale(scale, scale);  // 让 paths 和 strokeWidth 一并按 scale 缩
        for (Path p : strokes) {
            c.drawPath(p, renderPaint);
        }
        // 此时 stroke 的实际渲染宽度 = renderPaint.strokeWidth * scale
        //                           = (viewW * 0.044) * (360 / viewW) = 360 * 0.044 = 15.8 px
        // 跟 Python 的 16 px ★完全一致★
        return bmp;
    }
}
