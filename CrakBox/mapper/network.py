"""One interface for a network. It speaks of examples, and of items for what a judge sees, so that a language model fits later.

A pole is "pos" or "neg". Values on either pole are returned as magnitudes, so a larger value is a stronger firing on that pole.

A named gap: render returns images today, and the judge draws them on one sheet. A language model needs a text form of the sheet. It gets built when that adapter arrives."""
import sys
from abc import ABC, abstractmethod

SPLITS = ("read", "sealed")


class Network(ABC):
    name = "network"

    @abstractmethod
    def layers(self):
        """Layer names from the output down to the input."""

    @abstractmethod
    def nodes(self, layer):
        """Node names in a layer."""

    @abstractmethod
    def layer_of(self, node):
        """The layer that holds a node."""

    @abstractmethod
    def poles(self, node):
        """Which poles exist for this node: ("pos", "neg"), or ("pos",) when the architecture stores no negative side."""

    @abstractmethod
    def peaks(self, node, pole, split):
        """(value, position) per example: the strongest firing on this pole, and where it is."""

    @abstractmethod
    def value_at(self, node, pole, split, examples, positions):
        """The magnitude on this pole at given places. Zero when the firing is on the other side."""

    @abstractmethod
    def inputs(self, node, split, examples, positions):
        """Exact inputs: (D examples x inputs, input names, base, true). true is signed. base + D.sum(1) equals true, within float error."""

    @abstractmethod
    def readers(self, node, split, count):
        """[(reader name, signed change)], the strongest first. Each change is what this node does to that reader on its own strong examples."""

    @abstractmethod
    def render(self, split, node, examples, positions, size):
        """One image per example, as a judge sees it. It must hold no text."""

    @abstractmethod
    def similar(self, node, split, count):
        """Nodes of the same layer that act most like this node, the most similar first."""

    @abstractmethod
    def output_effect(self, node, split, examples):
        """KL on the outputs when this node carries only its mean, over the given examples."""

    @abstractmethod
    def evidence(self, tool, args):
        """(command line, working folder) for one evidence tool. The core logs the call and counts it as a step."""

    @abstractmethod
    def n_positions(self, node):
        """How many positions a node has per example."""

    def judged_poles(self, node):
        """The poles whose test has passed its control on this network. Others stay reported, and they are not a pass (CONSTITUTION section 6 rule 8)."""
        return self.poles(node)

    def largest_in_layer(self, layer, split):
        """The largest firing of each node in a layer, so that a node's strength can be read against its neighbours."""
        return [float(self.peaks(n, "pos", split)[0].max()) for n in self.nodes(layer)]

    def example_labels(self, split, examples):
        """A label per example when the data has them. None when it does not."""
        return None

    def canonical(self, name):
        """The node behind an input's name, without any mark an adapter adds."""
        return name

    def output_scores(self, examples, split):
        """{node of the output layer: what it writes to these examples' own outputs}. It starts a scope's trace."""
        raise NotImplementedError

    def goal_examples(self, goal, split):
        """The examples of a goal."""
        raise NotImplementedError

    def battery(self):
        """[(name, image)] fixed in advance, for the dial test. None when this network has none."""
        return None

    def battery_values(self, node):
        """The node's firing on each battery item, as a share of its largest firing on the reading split."""
        raise NotImplementedError

    def python(self):
        """The interpreter that runs this network's tools."""
        return sys.executable

    def evidence_help(self):
        """The evidence tools, in words for a reader."""
        return "see the adapter"

    def has_inputs(self, node):
        """False for a node that reads the raw input. It has no parent, so no parent test applies."""
        return True

    def input_kept(self, name):
        """True for an input that a parent claim keeps at the network's values, whatever the set Q. An architecture's bypass is the usual case."""
        return False

    def leak_patterns(self):
        """Regular expressions for words that would tell a judge where a node sits."""
        return []

    def words_of(self, name):
        """Plain words for a node that has no card by nature, such as an output. None for an ordinary node."""
        return None

    def output_names(self):
        """Names of output readers, for a test of what reads the last layer."""
        return []
