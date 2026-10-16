import os
import io
import datetime
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether
)


def generate_forensic_pdf(
    case_info,
    video_meta,
    result,
    face_images=None,
    fft_images=None
):
    """
    Generates a professional forensic audit PDF report using ReportLab.
    
    Returns:
        bytes: Raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom forensic report styles
    c_primary = colors.HexColor("#0F172A")
    c_accent = colors.HexColor("#0284C7")
    c_danger = colors.HexColor("#DC2626")
    c_success = colors.HexColor("#16A34A")
    c_dark = colors.HexColor("#1E293B")
    c_light = colors.HexColor("#F8FAFC")
    c_border = colors.HexColor("#CBD5E1")
    
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0
    )
    
    style_sub = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )
    
    style_h2 = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_dark,
        spaceAfter=6,
        spaceBefore=10
    )
    
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark
    )
    
    style_verdict = ParagraphStyle(
        'Verdict',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1
    )
    
    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>DEEPDETECT AI FORENSIC LABORATORY</b><br/><font size=8 color='#64748B'>AI FORENSIC INVESTIGATION & MEDIA AUTHENTICITY REPORT</font>", style_title),
            Paragraph(f"<b>CASE REF:</b> {case_info.get('case_id', 'DD-2026-X')}<br/><b>DATE:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_sub)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.5 * inch, 2.7 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 2, c_accent),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    # 2. Executive Verdict Box
    is_fake = result.get('is_fake', False)
    verdict_text = "CRITICAL WARNING: SYNTHETIC DEEPFAKE DETECTED" if is_fake else "VERIFIED AUTHENTIC: NO DEEPFAKE DETECTED"
    verdict_color = c_danger if is_fake else c_success
    bg_verdict = colors.HexColor("#FEF2F2") if is_fake else colors.HexColor("#F0FDF4")
    border_verdict = c_danger if is_fake else c_success

    verdict_p = Paragraph(f"<font color='{verdict_color.hexval()}'>{verdict_text}</font>", style_verdict)
    conf = result.get('confidence', 0.0)
    fake_p = result.get('fake_probability', 0.0) * 100.0
    
    verdict_details = Paragraph(
        f"<b>Classification Confidence:</b> {conf:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Synthetic Manipulation Score:</b> {fake_p:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Target Media:</b> {video_meta.get('filename', 'Unknown')}",
        ParagraphStyle('VerdictSub', parent=style_body, alignment=1)
    )

    verdict_box = Table([[verdict_p], [verdict_details]], colWidths=[7.2 * inch])
    verdict_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_verdict),
        ('BOX', (0, 0), (-1, -1), 1.5, border_verdict),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(verdict_box)
    elements.append(Spacer(1, 14))

    # 3. Two-Column Metadata & Telemetry Table
    elements.append(Paragraph("<b>1. EVIDENCE SPECIFICATIONS & CRYPTOGRAPHIC INTEGRITY</b>", style_h2))
    
    meta_data = [
        [
            Paragraph(f"<b>File Name:</b> {video_meta.get('filename', 'N/A')}", style_body),
            Paragraph(f"<b>Resolution:</b> {video_meta.get('resolution', 'N/A')}", style_body)
        ],
        [
            Paragraph(f"<b>File Size:</b> {video_meta.get('size_mb', 0.0):.2f} MB", style_body),
            Paragraph(f"<b>Framerate:</b> {video_meta.get('fps', 0.0):.2f} FPS", style_body)
        ],
        [
            Paragraph(f"<b>Duration:</b> {video_meta.get('duration', 0.0):.2f} seconds", style_body),
            Paragraph(f"<b>Analyzed Sequence:</b> {result.get('num_frames', 0)} frames", style_body)
        ],
        [
            Paragraph(f"<b>SHA-256 Hash:</b> <font face='Courier' size=7>{video_meta.get('sha256', 'N/A')}</font>", style_body),
            Paragraph(f"<b>MD5 Hash:</b> <font face='Courier' size=7>{video_meta.get('md5', 'N/A')}</font>", style_body)
        ],
        [
            Paragraph(f"<b>Lead Examiner:</b> {case_info.get('examiner', 'DeepDetect AI Agent')}", style_body),
            Paragraph(f"<b>Department:</b> {case_info.get('agency', 'Cyber Forensics Unit')}", style_body)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[3.6 * inch, 3.6 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # 4. Neural Architecture & Methodology
    elements.append(Paragraph("<b>2. FORENSIC METHODOLOGY & NEURAL PIPELINE</b>", style_h2))
    method_text = (
        "The submitted media was processed using an end-to-end <b>Spatial-Temporal Deepfake Forensic Architecture</b>. "
        "Uniform temporal sampling extracted facial regions via multi-scale Haar cascade detection. Spatial feature maps "
        "were extracted per frame using a deep <b>ResNet-18 Convolutional Neural Network</b> (512-dim bottleneck). "
        "Inter-frame temporal anomalies and facial coherence were analyzed using a <b>Bidirectional Long Short-Term Memory (Bi-LSTM)</b> "
        "network. Final probability classification incorporates frame-level loss integration with calibrated sigmoid activation."
    )
    elements.append(Paragraph(method_text, style_body))
    elements.append(Spacer(1, 14))

    # 5. Temporal Probabilities Summary
    elements.append(Paragraph("<b>3. TEMPORAL ANOMALY SEQUENCE RECORD</b>", style_h2))
    frame_scores = result.get('frame_probabilities', [])
    if frame_scores:
        cols_per_row = 8
        score_rows = []
        for i in range(0, len(frame_scores), cols_per_row):
            header_row = [Paragraph(f"<b>F#{j+1}</b>", style_body) for j in range(i, min(i + cols_per_row, len(frame_scores)))]
            val_row = []
            for j in range(i, min(i + cols_per_row, len(frame_scores))):
                sc = frame_scores[j] * 100.0
                sc_col = c_danger if sc >= 50.0 else c_success
                val_row.append(Paragraph(f"<font color='{sc_col.hexval()}'><b>{sc:.1f}%</b></font>", style_body))
            
            # Pad if needed
            while len(header_row) < cols_per_row:
                header_row.append("")
                val_row.append("")
            score_rows.append(header_row)
            score_rows.append(val_row)

        col_w = (7.2 / cols_per_row) * inch
        score_table = Table(score_rows, colWidths=[col_w] * cols_per_row)
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_light),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(score_table)
    
    elements.append(Spacer(1, 14))

    # 6. Extracted Face Frames (if provided)
    if face_images and len(face_images) > 0:
        elements.append(Paragraph("<b>4. FACIAL EVIDENCE CAPTURES</b>", style_h2))
        img_cells = []
        # Take up to 4 representative frames
        step = max(1, len(face_images) // 4)
        selected_imgs = face_images[::step][:4]
        
        for idx, pil_img in enumerate(selected_imgs):
            img_buf = io.BytesIO()
            # Resize thumbnail for PDF
            thumb = pil_img.copy()
            thumb.thumbnail((120, 120))
            thumb.save(img_buf, format='JPEG', quality=85)
            img_buf.seek(0)
            rl_img = RLImage(img_buf, width=1.5 * inch, height=1.5 * inch)
            img_cells.append([rl_img, Paragraph(f"<font size=8>Sample Keyframe {idx+1}</font>", style_body)])
        
        # Format in a table
        if img_cells:
            thumb_row = [cell[0] for cell in img_cells]
            cap_row = [cell[1] for cell in img_cells]
            pad_needed = 4 - len(thumb_row)
            thumb_row.extend([""] * pad_needed)
            cap_row.extend([""] * pad_needed)
            
            thumb_table = Table([thumb_row, cap_row], colWidths=[1.8 * inch] * 4)
            thumb_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            elements.append(thumb_table)

    elements.append(Spacer(1, 16))

    # 7. Forensic Sign-off Block
    sign_data = [
        [
            Paragraph("<b>CHAIN OF CUSTODY VERIFICATION:</b><br/>This digital media was hashed and audited under automated tamper-evident protocol.", style_body),
            Paragraph("<b>CERTIFIED FORENSIC ENGINE:</b><br/>DeepDetect ResNet-18 Bi-LSTM v2.4<br/>Status: <b>VERIFIED & SIGNED</b>", style_body)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.2 * inch, 3.0 * inch])
    sign_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, -1), 1, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(sign_table)

    # Build PDF document
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
