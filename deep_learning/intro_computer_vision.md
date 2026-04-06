# Computer Vision for ML Interviews

## Convolutional Neural Networks (CNNs)

### Why Convolutions?

Fully connected layers applied to images have two problems:
1. **Parameter explosion** — a 224×224×3 image with a 1000-unit FC layer = 150M parameters
2. **Spatial invariance** — FC layers don't exploit local structure or translation invariance

Convolutions solve this via:
- **Local connectivity** — each neuron sees a small spatial region (kernel)
- **Weight sharing** — the same kernel slides over all positions (reduces parameters drastically)
- **Equivariance** — if a feature shifts in the image, its activation shifts proportionally

### Convolution Operation

```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m,n] + bias
```

**Output spatial size:**
```
H_out = floor((H_in + 2×padding - kernel_size) / stride) + 1
```

**Parameter count per conv layer:**
```
params = (kernel_h × kernel_w × C_in + 1) × C_out
# +1 for bias; 1 bias per output channel
```

Example: 3×3 conv, 64→128 channels = (3×3×64 + 1) × 128 = 74,368 params

### Receptive Field

The receptive field of a neuron is the region of the input that can influence it.

```
RF_L = RF_{L-1} + (kernel_size - 1) × stride_product_{1..L-1}
```

Deeper layers have larger receptive fields. This is why deep networks learn hierarchical features:
- Layer 1: edges, colors
- Layer 3-5: textures, patterns
- Layer 7+: object parts
- Final layers: full objects

---

## Landmark CNN Architectures

### AlexNet (2012) — The Deep Learning Revolution
- 5 conv layers + 3 FC, ~60M params
- First to use ReLU activations, dropout, data augmentation at scale
- Won ImageNet 2012 by 10% margin (top-5 error: 15.3% vs 26.2%)

### VGGNet (2014) — Depth via Small Kernels
- Key insight: Stack 3×3 convolutions instead of large kernels
- Two 3×3 convs have same receptive field as one 5×5 but 28% fewer params and more non-linearity
- VGG-16: 16 weight layers, 138M params (heavy FC layers)

### ResNet (2015) — Residual Connections

The problem with very deep networks: **vanishing/exploding gradients** make training degrade.

**Residual block:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity   # skip connection
        return self.relu(out)
```

Skip connections let gradients flow directly through identity mappings, enabling training of 50, 101, 152+ layer networks. ResNet-50 achieves 24.2% top-1 error on ImageNet.

**Why skip connections help:**
- Gradient highway: ∂L/∂x = ∂L/∂F + 1 (identity term prevents vanishing)
- Each block learns a *residual* F(x) rather than a full transformation H(x)
- At initialization, blocks behave like identity → training starts with a shallow-like network

### EfficientNet (2019) — Compound Scaling

Rather than scaling only depth, width, or resolution independently, EfficientNet scales all three jointly using a compound coefficient φ:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ
subject to: α × β² × γ² ≈ 2 (FLOP constraint)
```

Best values found by NAS: α=1.2, β=1.1, γ=1.15. EfficientNet-B7 achieves 84.3% top-1 with 66M params vs ResNet-152's 78.3% with 60M params.

### MobileNet — Efficient Inference

**Depthwise separable convolution** factorizes a standard convolution into:
1. **Depthwise conv** — 1 filter per input channel (spatial filtering)
2. **Pointwise conv** — 1×1 conv to combine channels

```python
# Standard 3×3 conv: D_k × D_k × M × N operations  
# Depthwise separable: D_k × D_k × M + M × N operations
# Reduction factor: 1/N + 1/D_k² ≈ 8-9× for 3×3 conv

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch)   # depthwise
        self.pw = nn.Conv2d(in_ch, out_ch, 1)          # pointwise
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.pw(self.relu(self.bn1(self.dw(x))))))
```

---

## Batch Normalization

BatchNorm normalizes layer activations across the batch dimension:

```
μ_B = (1/m) Σ x_i          # batch mean
σ²_B = (1/m) Σ (x_i - μ_B)² # batch variance
x̂_i = (x_i - μ_B) / √(σ²_B + ε)   # normalize
y_i = γ x̂_i + β             # scale and shift (learned)
```

**Benefits:**
- Reduces internal covariate shift — each layer sees normalized inputs
- Acts as regularization (slightly noisy estimates of population stats)
- Allows higher learning rates
- Makes network less sensitive to weight initialization

**At inference:** uses running mean/variance accumulated during training (not batch stats).

**Placement in ResNet:** Conv → BN → ReLU (pre-activation ResNet uses BN → ReLU → Conv, shown to perform slightly better).

---

## Object Detection

### Two-Stage Detectors (R-CNN Family)

**R-CNN pipeline:**
1. Region proposals (Selective Search) ~2000 regions
2. Warp each region, run CNN → features
3. SVM classification + bounding box regression

**Faster R-CNN** replaced slow proposal generation with **Region Proposal Network (RPN):**

```
Input Image
    ↓
Backbone CNN (e.g., ResNet-50) → Feature Map
    ↓                                ↓
  RPN                          RoI Pooling
(proposes ~300 boxes)               ↓
                           Classification head
                           BBox regression head
```

RPN slides over the feature map, predicts objectness + box offsets at each location for k anchor boxes (different scales/aspect ratios). Anchors are the key idea: pre-defined reference boxes.

### Single-Stage Detectors

**YOLO (You Only Look Once):**
- Divide image into S×S grid
- Each cell predicts B bounding boxes + class probabilities
- Single forward pass → predictions at multiple scales (YOLOv3+)
- Faster than two-stage but historically lower accuracy on small objects

**YOLOv8 in practice:**
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano variant

# Training
results = model.train(
    data="coco128.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda"
)

# Inference
results = model("image.jpg")
for r in results:
    boxes = r.boxes.xyxy    # bounding boxes
    scores = r.boxes.conf   # confidence scores
    classes = r.boxes.cls   # class indices
```

**SSD (Single Shot Detector):**
- Predicts at multiple feature map scales in one forward pass
- Default boxes (anchors) at different aspect ratios and scales
- Better than YOLO for small objects due to multi-scale predictions

### Key Metrics

**IoU (Intersection over Union):**
```python
def iou(box_a, box_b):
    # boxes: [x1, y1, x2, y2]
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (box_a[2]-box_a[0])*(box_a[3]-box_a[1]) + \
                 (box_b[2]-box_b[0])*(box_b[3]-box_b[1]) - inter_area
    return inter_area / union_area
```

**Mean Average Precision (mAP):**
- For each class, compute Precision-Recall curve by varying confidence threshold
- Average Precision (AP) = area under PR curve
- mAP = mean AP over all classes
- mAP@0.5: IoU threshold 0.5 (PASCAL VOC)
- mAP@[0.5:0.95]: average over IoU thresholds 0.5–0.95 (COCO standard)

**Non-Maximum Suppression (NMS):**
```python
def nms(boxes, scores, iou_threshold=0.5):
    # Sort by score descending
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Compute IoU of best box vs rest
        ious = [iou(boxes[i], boxes[j]) for j in order[1:]]
        # Keep boxes with IoU < threshold
        order = order[1:][np.array(ious) < iou_threshold]
    return keep
```

---

## Image Segmentation

### Semantic vs Instance Segmentation

- **Semantic segmentation** — classify every pixel (no distinction between instances): person/car/road
- **Instance segmentation** — detect each object instance + its pixel mask: person₁, person₂

### U-Net Architecture

Originally for medical image segmentation, now widely used:

```
Encoder (downsampling):  224→112→56→28→14 (feature maps grow)
         ↓          with max pooling at each step
Bottleneck:              14×14 feature map
         ↓
Decoder (upsampling):    14→28→56→112→224
         ↑           with transposed conv / bilinear upsampling
Skip connections: encoder features concatenated to decoder at each scale
```

**Key insight:** Skip connections preserve fine-grained spatial details that the encoder compresses away, enabling precise pixel-level predictions.

```python
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = UNetBlock(128 + 64, 64)  # +64 from skip
        self.out  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up(e2), e1], dim=1))
        return self.out(d1)
```

---

## Vision Transformers (ViT)

ViT applies the standard Transformer architecture to images:

1. Split image into fixed-size patches (16×16)
2. Linearly embed each patch → sequence of tokens
3. Add positional embeddings + [CLS] token
4. Feed through standard Transformer encoder
5. Classify from [CLS] token

```python
# Key: an image becomes a sequence of patch tokens
# 224×224 image, 16×16 patches → (224/16)² = 196 tokens
# Each token is a 16×16×3 = 768-dim vector

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed
```

**ViT vs CNN trade-offs:**
| Aspect | CNN | ViT |
|--------|-----|-----|
| Data efficiency | Better (inductive biases) | Needs large datasets |
| Scalability | Limited | Scales with data/compute |
| Global context | Via deep layers | From layer 1 |
| Transfer learning | Strong | Even stronger at scale |
| Computation | Linear in image size | Quadratic in patches |

---

## Data Augmentation

Augmentation is critical for CV — prevents overfitting and improves robustness.

```python
import torchvision.transforms as T
from torchvision.transforms import v2  # torchvision v2 API

train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.RandomGrayscale(p=0.2),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                 std=[0.229, 0.224, 0.225]),
])

# Advanced: CutMix and MixUp (standard in modern training)
cutmix = v2.CutMix(num_classes=1000)
mixup  = v2.MixUp(num_classes=1000)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
```

**Advanced augmentations:**
- **Cutout/Random Erasing** — mask out random patches
- **RandAugment** — randomly apply N augmentations from a fixed policy
- **MixUp** — linearly interpolate two images and their labels
- **CutMix** — cut a patch from one image, paste into another, mix labels by area

---

## Transfer Learning in Practice

```python
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Option 1: Feature extraction — freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for new task
model.fc = nn.Linear(model.fc.in_features, n_classes)  # only this trains

# Option 2: Fine-tuning — use differential learning rates
optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # backbone: low lr
    {'params': model.fc.parameters(),     'lr': 1e-3},  # head: high lr
])
```

**Transfer learning tips:**
- If dataset is small + similar to ImageNet: freeze backbone, train head only
- If dataset is large + different: unfreeze all, use low learning rate throughout
- Always normalize inputs with the pretrained model's statistics (ImageNet mean/std)

---

## Common Interview Questions

**Q: Why do we use 3×3 kernels so often in modern CNNs?**
Two 3×3 convolutions have the same receptive field as one 5×5 but use fewer parameters (2×3×3 = 18 vs 25 weights) and more non-linearity. Three 3×3s match a 7×7. This was the key insight from VGGNet and became universal.

**Q: Explain the vanishing gradient problem and how ResNet solves it.**
In deep networks, gradients are multiplied through many layers via chain rule. If those values are < 1, gradients shrink exponentially. ResNet's skip connections create gradient highways: ∂L/∂x includes an identity term (∂L/∂F + I), ensuring gradients don't vanish even in 100+ layer networks.

**Q: What is the difference between detection, segmentation, and classification?**
Classification: whole image → one label. Detection: image → multiple bounding boxes + labels. Semantic segmentation: each pixel → label (no instance distinction). Instance segmentation: each pixel → label + instance ID. Panoptic = semantic + instance combined.

**Q: How would you handle a highly imbalanced class dataset in CV?**
- Oversample minority class (SMOTE-like augmentation, duplicate with augmentation)
- Class-weighted loss: weight minority class higher
- Focal loss: down-weight easy negatives, focus on hard examples
- Use mAP or F1 per class rather than accuracy for evaluation
- Consider two-stage approach: detect all objects first, then classify

**Q: When would you choose a CNN over a ViT?**
CNNs are better for small datasets (strong inductive biases) and when you need computational efficiency on edge devices. ViTs excel with large datasets, benefit more from scaling, and are better at capturing global context. For most production tasks with sufficient data, a ViT (or hybrid like ConvNeXt) tends to outperform pure CNNs.

**Q: What is Focal Loss and when would you use it?**
Focal Loss = -(1-p_t)^γ × log(p_t). The modulating factor (1-p_t)^γ down-weights easy examples (high p_t) and focuses training on hard, misclassified examples. Used in object detection (RetinaNet) where the background class massively outnumbers foreground, making cross-entropy training dominated by easy negatives.
