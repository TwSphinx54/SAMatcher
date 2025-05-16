import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()

def box_xyxy_to_cxywh(xyxy, max_h, max_w):
    # cxywh: [N, 4]
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x1 = x1.clamp(min=0.0, max=max_w)
    x2 = x2.clamp(min=0.0, max=max_w)
    y1 = y1.clamp(min=0.0, max=max_h)
    y2 = y2.clamp(min=0.0, max=max_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    b = [cx, cy, w, h]
    return torch.stack(b, dim=-1)

def bbox_oiou(target, pred, eps=1e-7):
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

    # IoU
    ious = overlap / ap
    return ious

def draw_bbox(img, box, score=None, gt_box=None, tag=True,
              pred_box_color=(255, 100, 0),  # Orange-Blue for Predicted box
              gt_box_color=(0, 200, 50),      # Green for GT box
              text_color=(255, 255, 255),     # White text
              box_thickness=2,
              font_face=cv.FONT_HERSHEY_SIMPLEX,
              font_scale=0.6,                 # Adjusted for better fit
              text_thickness=1,               # Adjusted for cleaner text
              tag_padding=4):                 # Padding around text within its background
    
    draw_img = img.copy() # Work on a copy to avoid modifying the original
    img_h, img_w = draw_img.shape[:2]

    # Helper function to draw a text tag with a background
    def _draw_tag_on_image(current_image, text_content, box_corner_pt, tag_bg_color):
        (text_w, text_h), baseline = cv.getTextSize(text_content, font_face, font_scale, text_thickness)

        # Position tag background and text just above the box_corner_pt (top-left of a box)
        # Text baseline (y-coordinate for cv.putText)
        tag_text_y = box_corner_pt[1] - tag_padding - baseline // 2 
        # Top of the background rectangle for the tag
        tag_bg_y1 = tag_text_y - text_h - tag_padding + baseline // 2


        # If tag goes off screen (top), move it inside the box, just below the top line
        if tag_bg_y1 < tag_padding:
            tag_text_y = box_corner_pt[1] + text_h + tag_padding + baseline // 2
            tag_bg_y1 = box_corner_pt[1] + tag_padding - baseline // 2
        
        tag_bg_x1 = box_corner_pt[0]
        tag_bg_x2 = box_corner_pt[0] + text_w + tag_padding * 2
        tag_bg_y2 = tag_text_y + tag_padding + baseline // 2


        # Clip background rectangle to image boundaries
        final_bg_x1 = max(0, tag_bg_x1)
        final_bg_y1 = max(0, tag_bg_y1)
        final_bg_x2 = min(img_w - 1, tag_bg_x2)
        final_bg_y2 = min(img_h - 1, tag_bg_y2)

        # Adjust text position based on clipped background (simple horizontal adjustment)
        final_text_x = final_bg_x1 + tag_padding
        final_text_y = tag_text_y 
        # Ensure text baseline is within the visible part of the (potentially clipped) background
        if final_text_y > final_bg_y2 - tag_padding - baseline // 2 :
             final_text_y = final_bg_y2 - tag_padding - baseline // 2
        if final_text_y < final_bg_y1 + text_h + tag_padding - baseline // 2:
             final_text_y = final_bg_y1 + text_h + tag_padding - baseline // 2


        if final_bg_x1 < final_bg_x2 and final_bg_y1 < final_bg_y2: # Only draw if valid rect
            cv.rectangle(current_image, (final_bg_x1, final_bg_y1), (final_bg_x2, final_bg_y2), tag_bg_color, -1)
        
        cv.putText(current_image, text_content, (final_text_x, final_text_y),
                   font_face, font_scale, text_color, text_thickness, cv.LINE_AA)

    # 1. Draw GT Box and its Tag
    if gt_box is not None:
        # Ensure gt_box coordinates are integers for drawing
        pt1_gt = tuple(np.array(gt_box[:2]).astype(int))
        pt2_gt = tuple(np.array(gt_box[2:]).astype(int))
        cv.rectangle(draw_img, pt1_gt, pt2_gt, gt_box_color, box_thickness)

        if tag:
            _draw_tag_on_image(draw_img, "Ground Truth", pt1_gt, gt_box_color)

    # 2. Draw Predicted Box and its Tag
    if box is not None and len(box) == 4:
        # Ensure box coordinates are integers for drawing
        pt1_pred = tuple(np.array(box[:2]).astype(int))
        pt2_pred = tuple(np.array(box[2:]).astype(int))
        cv.rectangle(draw_img, pt1_pred, pt2_pred, pred_box_color, box_thickness)

        if tag:
            pred_label_text = "Ours" if score is None else f"Ours: {score:.3f}"
            _draw_tag_on_image(draw_img, pred_label_text, pt1_pred, pred_box_color)
    
    return draw_img
