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
 * 笔迹采集 View:白底,接收 touch 画黑笔画,提供 toBitmap() 给推理。
 *
 * 设计:
 * - 内部维护 Path 列表(每笔一个 Path),便于 undo/clear
 * - onDraw 直接重画,简单
 * - 抬笔时通过 listener 通知上层(通常用 idle 计时器后再触发识别)
 */
public class HandwritingView extends View {

    public interface StrokeListener {
        void onStrokeStart();
        void onStrokeEnd();
    }

    private final List<Path> strokes = new ArrayList<>();
    private Path current;
    private final Paint paint = new Paint();
    private float lastX, lastY;
    private StrokeListener listener;

    // 笔画粗细需要按"缩到 64×64 后还有 ~2-3 px"反推。
    // 高分屏 1080+ 像素画板,缩 ~20 倍 → 现场粗细要 40-60 px。
    // 用 dp 不够通用(高 DPI 越细),改用画布宽度比例。
    private static final float STROKE_RATIO = 0.038f;  // ~画布宽度 4%

    public HandwritingView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        paint.setColor(Color.BLACK);
        paint.setAntiAlias(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeJoin(Paint.Join.ROUND);
        // strokeWidth 在 onSizeChanged 里按画布宽度算
        paint.setStrokeWidth(20f);  // 占位,真值见 onSizeChanged
        setBackgroundColor(Color.WHITE);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        if (w > 0) {
            paint.setStrokeWidth(w * STROKE_RATIO);
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
            canvas.drawPath(p, paint);
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
                    // quad 平滑(用上一次中点作为控制)
                    float mx = (x + lastX) / 2f;
                    float my = (y + lastY) / 2f;
                    current.quadTo(lastX, lastY, mx, my);
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

    /**
     * 把当前所有笔画渲染到指定大小的灰度 bitmap(白底黑字),用于喂模型。
     * 渲染流程:
     *   1. View 内容画到 viewSize × viewSize 大 bitmap(白底,strokes 黑色)
     *   2. 上层负责后续:bbox 裁剪 → 长边缩到 56 → 居中放 64x64
     *      (这里只产出原始大尺寸 bitmap,归一化由 HCCRRecognizer 的 native 端做)
     */
    public Bitmap renderToBitmap() {
        int w = Math.max(1, getWidth());
        int h = Math.max(1, getHeight());
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);
        c.drawColor(Color.WHITE);
        for (Path p : strokes) {
            c.drawPath(p, paint);
        }
        return bmp;
    }
}
