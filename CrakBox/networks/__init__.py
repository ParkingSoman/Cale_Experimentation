"""Network adapters. get(name) loads one only when it is needed, so the core never imports torch."""
import os
import sys


def get(name):
    if name == "resnet18":
        from networks.resnet18.adapter import ResNet18
        return ResNet18()
    if name == "convnext_tiny":
        from networks.convnext_tiny.adapter import ConvNeXtTiny
        return ConvNeXtTiny()
    if name == "fake" and os.environ.get("MAPPER_TEST"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
        from fake import FakeNet
        return FakeNet()
    raise ValueError(f"no adapter for {name}")
