"""ResNet-18 students with every width times f (plan 2026-11-25 "distill the whole of A").

make_student(f)          ResNet-18 with stage widths round(f * (64, 128, 256, 512)). Same module names as torchvision,
                         so A's tensors slice straight in. Inner width of a block = its stage width (as in A).
channel_scores(A, x)     Minitron activation scores, written before the run:
                           inner[s][b] = mean ReLU output of block b's inner channels (between conv1 and conv2)
                           stream[s]   = mean |stream| at every block output of stage s, averaged over its 2 blocks
cut_from_a(A, f, sc)     the P start: keep the top round(f * width) channels by score in each set (kept in A's order),
                         copy A's weights for them. One stream set per stage, so the skip path stays legal.
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

BASE = (64, 128, 256, 512)
STRIDES = (1, 2, 2, 2)


def widths(f):
    return tuple(max(1, round(f * c)) for c in BASE)


class Student(nn.Module):
    def __init__(self, w, num_classes=1000):
        super().__init__()
        self.w = tuple(w)
        self.conv1 = nn.Conv2d(3, w[0], 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        inp = w[0]
        for i, (c, s) in enumerate(zip(w, STRIDES)):
            ds = None
            if s != 1 or inp != c:
                ds = nn.Sequential(nn.Conv2d(inp, c, 1, s, bias=False), nn.BatchNorm2d(c))
            setattr(self, f"layer{i + 1}", nn.Sequential(BasicBlock(inp, c, s, ds), BasicBlock(c, c)))
            inp = c
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w[3], num_classes)
        for m in self.modules():                       # torchvision's ResNet init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def make_student(f, seed=0):
    torch.manual_seed(seed)
    return Student(widths(f))


def n_neurons(m):
    """Every conv output channel (stem, inner, block outputs, downsample) = the network's neurons."""
    return sum(mod.out_channels for mod in m.modules() if isinstance(mod, nn.Conv2d))


def layers(m):
    return [m.layer1, m.layer2, m.layer3, m.layer4]


@torch.no_grad()
def channel_scores(A, x, bs=100):
    """x: normalized photos (N, 3, 224, 224) on A's device. Returns {'inner': [[4 x 2 tensors]], 'stream': [4 tensors]}."""
    inner = [[0.0, 0.0] for _ in range(4)]
    stream = [0.0] * 4
    n = 0
    for a in range(0, len(x), bs):
        h = A.maxpool(A.relu(A.bn1(A.conv1(x[a:a + bs]))))
        for s, layer in enumerate(layers(A)):
            for b, bl in enumerate(layer):
                i = torch.relu(bl.bn1(bl.conv1(h)))
                inner[s][b] = inner[s][b] + i.sum((0, 2, 3)) / (i.shape[2] * i.shape[3])
                skip = h if bl.downsample is None else bl.downsample(h)
                h = torch.relu(bl.bn2(bl.conv2(i)) + skip)
                stream[s] = stream[s] + h.abs().sum((0, 2, 3)) / (h.shape[2] * h.shape[3]) / 2
        n += len(x[a:a + bs])
    return dict(inner=[[(v / n).cpu() for v in row] for row in inner], stream=[(v / n).cpu() for v in stream])


def keep_idx(score, k):
    return torch.sort(torch.topk(score, k).indices).values


def _conv(dst, src, out_idx, in_idx):
    dst.weight.copy_(src.weight[out_idx][:, in_idx])


def _bn(dst, src, idx):
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(dst, name).copy_(getattr(src, name)[idx])
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


@torch.no_grad()
def cut_from_a(A, f, sc):
    """The P start: A cut to width f by the scores sc. Returns (student, kept index sets)."""
    w = widths(f)
    S = Student(w).to(next(A.parameters()).device)
    st = [keep_idx(sc["stream"][s], w[s]).to(S.fc.weight.device) for s in range(4)]
    inn = [[keep_idx(sc["inner"][s][b], w[s]).to(S.fc.weight.device) for b in range(2)] for s in range(4)]
    _conv(S.conv1, A.conv1, st[0], torch.arange(3, device=st[0].device))
    _bn(S.bn1, A.bn1, st[0])
    for s, (ls, la) in enumerate(zip(layers(S), layers(A))):
        prev = st[s - 1] if s > 0 else st[0]
        for b, (bs_, ba) in enumerate(zip(ls, la)):
            src_in = prev if b == 0 else st[s]
            _conv(bs_.conv1, ba.conv1, inn[s][b], src_in)
            _bn(bs_.bn1, ba.bn1, inn[s][b])
            _conv(bs_.conv2, ba.conv2, st[s], inn[s][b])
            _bn(bs_.bn2, ba.bn2, st[s])
            if ba.downsample is not None:
                _conv(bs_.downsample[0], ba.downsample[0], st[s], src_in)
                _bn(bs_.downsample[1], ba.downsample[1], st[s])
    S.fc.weight.copy_(A.fc.weight[:, st[3]])
    S.fc.bias.copy_(A.fc.bias)
    kept = dict(stream=[v.cpu() for v in st], inner=[[v.cpu() for v in row] for row in inn])
    return S, kept
