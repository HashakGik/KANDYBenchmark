import json
import numpy as np
import colorsys
import re
import random
import math

from PIL import Image, ImageDraw

# Class encoding a single task. It takes the task parameters, the sets of positive and negative samples, a list of past tasks, and optionally a logger to print debug informations.
class Task:
    def __init__(self, name, samples, alpha, beta, gamma, noisy_color, noisy_size, positive_samples, negative_samples, past_tasks,  canvas_size, padding, logger=None):
        self.name = name # Task name.
        self.alpha = alpha # Sampling probability for previous tasks.
        self.beta = beta # Minimum supervision probability.
        self.gamma = gamma # Maximum supervision probability.
        self.total = samples # Number of samples to generate for this task.
        self.noisy_color = noisy_color # Should the color be altered with noise?
        self.noisy_size = noisy_size # Should the size be altered with noise?
        

        # Color noise parameters: the RGB color is converted to HSV and then 0-mean gaussian noise is injected into each coordinate. These parameters are the standard deviation of each gaussian.
        # Values were chosen by trial and error to preserve perceptual semantics for up to 16 well-separed colors (ie. a human can classify a noisy yellow still as "yellow").
        self.h_sigma = 0.01
        self.s_sigma = 0.2
        self.v_sigma = 0.2

        # Size noise parameter. A uniform random value from -noise to +noise is added to size. (eg. a base size of 25 pixels with a noise of 5 can range from 20 to 30 pixels).
        self.size_noise = 5

        # Available base shapes. Extending this dictionary requires to rewrite sample_base_object() as well.
        self.shapes = ["triangle", "square", "circle"]

        # Available colors. This dictionary can be extended arbitrarily, but no guarantees are given for "perceptual semantics" when injected with noise.
        self.colors = {
           "red": "#ff0000",
           "green": "#00ff00",
           "blue": "#0000ff",
           "magenta": "#ff00ff",
           "cyan": "#00ffff",
           "yellow": "#ffff00"
        }

        # Available sizes. This dictionary can be extended arbitrarily (as long as each size can be drawn within the canvas and as long as injecting noise does not make classes overlap).
        self.sizes = {
            "small": 10,
            "large": 25
        }

        # Available compositions. Extending this list requires to write new composition methods and to modify sample().
        self.compositions = ["in", "stack", "side_by_side", "diagonal_ul_lr", "diagonal_ll_ur", "grid", "random"]

        self.positive_samples = self.parse_rules(positive_samples)
        self.negative_samples = self.parse_rules(negative_samples)
        self.past_tasks = past_tasks

        self.canvas_size = canvas_size
        self.padding = padding
        self.logger = logger

    # Simple logging method.
    def log(self, message, level="debug"):
        if self.logger is not None:
            if level == "debug":
                self.logger.debug(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.error(message)

    # JSON validation: base rules. It also expands not_ and | operators.
    def parse_base_rule(self, r):

        tmp = {}

        if r["shape"] == "any":
            tmp["shape"] = self.shapes
        elif r["shape"].startswith("not_"):
            tmp["shape"] = list(set(self.shapes).difference([r["shape"][len("not_"):]]))
        else:
            tmp["shape"] = r["shape"].split("|")
        if r["color"] == "any":
            tmp["color"] = list(self.colors.keys())
        elif r["color"].startswith("not_"):
            tmp["color"] = list(set(self.colors.keys()).difference([r["color"][len("not_"):]]))
        else:
            tmp["color"] = r["color"].split("|")
        if r["size"] == "any":
            tmp["size"] = list(self.sizes.keys())
        elif r["size"].startswith("not_"):
            tmp["size"] = list(set(self.colors.keys()).difference([r["size"][len("not_"):]]))
        else:
            tmp["size"] = r["size"].split("|")

        tmp["shape"] = list(set(tmp["shape"]))
        tmp["color"] = list(set(tmp["color"]))
        tmp["size"] = list(set(tmp["size"]))

        assert len(set(tmp["shape"]).intersection(self.shapes)) > 0 and len(set(tmp["shape"]).union(self.shapes)) == len(
            self.shapes), "Shapes should be a subset of {}, received {}".format(self.shapes, tmp["shape"])
        assert len(set(tmp["color"]).intersection(self.colors.keys())) > 0 and len(set(tmp["color"]).union(self.colors.keys())) == len(
            self.colors.keys()), "Colors should be a subset of {}, received {}".format(self.colors.keys(), tmp["color"])
        assert len(set(tmp["size"]).intersection(self.sizes.keys())) > 0 and len(set(tmp["size"]).union(self.sizes.keys())) == len(
            self.sizes.keys()), "Sizes should be a subset of {}, received {}".format(self.sizes.keys(), tmp["size"])

        return tmp

    # JSON validation: recursive rules.
    def parse_rules(self, rules):
        out = []

        for r in rules:
            assert len(set(r.keys()).intersection(self.compositions)) <= 1, "There can be at most one composition operator at this level, found {}".format(set(r.keys()).intersection(self.compositions))

            if len(set(r.keys()).intersection(self.compositions)) == 0:
                # Base case: the rule under consideration is a base shape.
                out.append(self.parse_base_rule(r))
            else:
                # Inductive case: the rule under consideration is a composition.
                c = list(set(r.keys()).intersection(self.compositions))[0]
                tmp = {c : self.parse_rules(r[c])}
                if "shuffled" in r.keys():
                    tmp["shuffled"] = r["shuffled"]
                else:
                    tmp["shuffled"] = False

                out.append(tmp)

        return out

    # Compute random (uniform) size noise.
    def inject_size_noise(self, size):
        if self.size_noise > 0:
            rnd = np.random.uniform(-self.size_noise, self.size_noise)
        else:
            rnd = 0
        return max(1, size + rnd)

    # colorsys requires lists, PIL requires #rrggbb strings. These utilities handle conversions.

    def _rgb_to_list(self, string):
        assert re.match("#[0-9a-f]{6}", string, flags=re.IGNORECASE) is not None
        rgb = re.search("#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", string, flags=re.IGNORECASE).groups()
        rgb = [int(x, 16) for x in rgb]
        return rgb

    def _list_to_rgb(self, rgb):
        assert len(rgb) == 3
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Inject gaussian noise into HSV coordinates.
    def inject_color_noise(self, rgb):
        hsv = colorsys.rgb_to_hsv(*self._rgb_to_list(rgb))
        h = hsv[0]  # h in [0.0, 1.0]
        s = hsv[1]  # s in [0.0, 1.0]
        v = hsv[2]  # v in [0, 255]
        if self.h_sigma > 0.0:
            rnd = np.random.normal(0, self.h_sigma)
            h = max(0, min(1, h + rnd * 1.0))
        if self.s_sigma > 0.0:
            rnd = np.random.normal(0, self.s_sigma)
            s = max(0, min(1, s + rnd * 1.0))
        if self.v_sigma > 0.0:
            rnd = np.random.normal(0, self.v_sigma)
            v = max(0, min(255, int(v + rnd * 255)))

        return self._list_to_rgb(colorsys.hsv_to_rgb(h, s, v))

    # Randomly sample a base object from the sample list and draw it on a canvas_size image.
    def sample_base_object(self, sample, canvas_size):
        shape = np.random.choice(sample["shape"])
        color_name = np.random.choice(sample["color"])
        size_name = np.random.choice(sample["size"])

        if self.noisy_color:
            color = self.inject_color_noise(self.colors[color_name])
        else:
            color = self.colors[color_name]
        if self.noisy_size:
            size = self.inject_size_noise(self.sizes[size_name])
        else:
            size = self.sizes[size_name]

        if canvas_size[0] < max(self.sizes.values()) + self.size_noise or canvas_size[1] < max(self.sizes.values()) + self.size_noise:
            self.log("Canvas size too small. Defaulting to ({},{})".format(max(self.sizes.values()) + self.size_noise, max(self.sizes.values()) + self.size_noise), "warning")
            canvas_size = (max(self.sizes.values()) + self.size_noise, max(self.sizes.values()) + self.size_noise)

        logstring = "({}-{}-{})".format(shape, color_name, size_name)
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(bitmap)

        bounding_box = (
        (canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 - size / 2),
        (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 + size / 2))

        # PIL regular_polygon wants the coordinates of the inscribed circle, making polygons too large for the canvas size. Therefore we manually draw our own.

        if shape == "circle":
            draw.ellipse(bounding_box, fill=color, outline=None)
        elif shape == "triangle":
            xy = ((canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 0 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 0 - np.pi / 2)),
                  (canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 1 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 1 - np.pi / 2)),
                  (canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 2 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 2 - np.pi / 2))
                  )
            draw.polygon(xy, fill=color, outline=None)
        elif shape == "square":
            xy = ((canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 - size / 2), (canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 + size / 2), (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 + size / 2), (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 - size / 2))
            draw.polygon(xy, fill=color, outline=None)

        return (bitmap, logstring)

    # "in" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_in(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        logstrings = []

        if sample["shuffled"]:
            random.shuffle(sample["in"])

        for s in sample["in"]:
            bmp, ls = self.sample(s, canvas_size)
            bitmap.paste(bmp, mask=bmp)
            logstrings.append(ls)

        return bitmap, "in[{}]".format("-".join(logstrings))

    # "stack" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_stack(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0]),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["stack"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            random.shuffle(sample["stack"])

        for s in sample["stack"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = canvas_size[1] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            x0 = int(0)
            y0 = int((i + 1) * step - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "stack[{}]".format("-".join(logstrings))

    # "side_by_side" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_sbs(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["side_by_side"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1]) )

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            random.shuffle(sample["side_by_side"])

        for s in sample["side_by_side"]:
            bmp, ls = self.sample(s, child_canvas )
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = canvas_size[0] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            y0 = int(0)
            x0 = int((i + 1) * step - child_canvas[0] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "side_by_side[{}]".format("-".join(logstrings))

    # "diagonal_ul_lr" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_ullr(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["diagonal_ul_lr"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["diagonal_ul_lr"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            random.shuffle(sample["diagonal_ul_lr"])

        for s in sample["diagonal_ul_lr"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (i + 1) * step
            x0 = int(d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "diag1[{}]".format("-".join(logstrings))

    # "diagonal_ll_ur" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_llur(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["diagonal_ll_ur"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["diagonal_ll_ur"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            random.shuffle(sample["diagonal_ll_ur"])

        for s in sample["diagonal_ll_ur"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (i + 1) * step
            x0 = int(canvas_size[0] - d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "diag2[{}]".format("-".join(logstrings))

    # "grid" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_grid(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        n = math.ceil(math.sqrt(len(sample["grid"])))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // n),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // n))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            random.shuffle(sample["grid"])

        for s in sample["grid"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)


        for i in range(len(bitmaps)):
            j = (i % n)
            k = i // n

            x0 = int(j * canvas_size[0] / n)
            y0 = int(k * canvas_size[1] / n)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "grid[{}]".format("-".join(logstrings))

    # "random" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_random(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        logstrings = []

        for s in sample["random"]:
            bmp, ls = self.sample(s, canvas_size)
            m = max(self.sizes.values()) + self.size_noise
            x0 = np.random.randint(m // 2, canvas_size[0] - m // 2) - canvas_size[0] // 2
            y0 = np.random.randint(m // 2, canvas_size[1] - m // 2) - canvas_size[1] // 2
            x1 = x0 + canvas_size[0]
            y1 = y0 + canvas_size[1]

            bitmap.paste(bmp, (x0, y0, x1, y1), mask=bmp)
            logstrings.append(ls)

        return bitmap, "random[{}]".format("-".join(logstrings))

    # Wrapper for the recursive sampling procedure.
    def sample(self, sample_set, canvas_size):
        if isinstance(sample_set, list):
            sample = np.random.choice(sample_set)
        else:
            sample = sample_set


        if "in" in sample.keys():
            return self.sample_in(sample, canvas_size)
        elif "stack" in sample.keys():
            return self.sample_stack(sample, canvas_size)
        elif "side_by_side" in sample.keys():
            return self.sample_sbs(sample, canvas_size)
        elif "diagonal_ul_lr" in sample.keys():
            return self.sample_ullr(sample, canvas_size)
        elif "diagonal_ll_ur" in sample.keys():
            return self.sample_llur(sample, canvas_size)
        elif "grid" in sample.keys():
            return self.sample_grid(sample, canvas_size)
        elif "random" in sample.keys():
            return self.sample_random(sample, canvas_size)
        else:
            return self.sample_base_object(sample, canvas_size)

    # Upper level wrapper. Draws the sample and then removes transparency and adds a padding.
    # It returns both the image and a logstring describing the objects in the image (useful for logging or symbolic processing).
    def draw_sample(self, sample_set):
        w = self.canvas_size[0] - self.padding
        h = self.canvas_size[1] - self.padding
        bitmap, logstring = self.sample(sample_set, (w,h))

        out = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
        out.paste(bitmap, (self.padding // 2, self.padding // 2), mask=bitmap)

        return out.convert('RGB'), logstring

    # Outputs a supervised triple (anchor, positive, negative).
    def sample_supervised(self):
        return self.draw_sample(self.positive_samples), self.draw_sample(self.positive_samples), self.draw_sample(self.negative_samples)

    # Outputs an unsupervised sample (anchor, None, None).
    def sample_unsupervised(self):
        return self.draw_sample(self.positive_samples), None, None

    # Produces size supervised triples (anchor, positive, negative) as a list of tuples. If converted to numpy the shape would be (size, apn=3, width, height, rgb=3).
    def get_test_set(self, size):
        return [self.sample_supervised() for _ in range(size)]

    # Teacher generator. It samples a decision and generates images accordingly. Sampling alternates old tasks and the current one, to avoid starvation.
    # The number of samples generated is guaranted to be at least self.total (in case it never samples from past tasks) and at most 2*self.total (in case at each step it decides to sample from past tasks).
    def get_decision(self):
        self.log("BEGINNING TASK {}".format(self.name))
        self.log("Number of samples: {}, alpha: {}, beta: {}, gamma: {}".format(self.total, self.alpha, self.beta, self.gamma))
        i = 0
        scale = (np.log(self.gamma / self.beta)) / (self.gamma * self.total) # Guarantee a minimum probability of beta and a maximum of gamma. See appendix for an explanation.


        while i < self.total:
            # 1. Extract from a geometric distribution alpha * (1 - alpha)**j the probability of sampling from old task -j.
            # It stops at the first successful decision. If every decision fails, it extracts no samples from the past.
            for j in range(len(self.past_tasks)):
                if np.random.random() < self.alpha * (1 - self.alpha) ** j:
                    (s_a, logstring_a),(s_p, logstring_p),(s_n, logstring_n) = self.past_tasks[len(self.past_tasks) - j - 1].sample_supervised()
                    self.log("SUPERVISED SAMPLE FROM OLD TASK {}: {}".format(self.past_tasks[len(self.past_tasks) - j - 1].name, (logstring_a, logstring_p, logstring_n)))
                    yield s_a, s_p, s_n
                    break # Stop at the first success.

            # 2. Extract from an exponential distribution the decision of providing a supervised or unsupervised sample.
            # Each sample is guaranteed to be supervised with at least a probability of beta.
            t = i * scale
            if np.random.random() < self.gamma * np.exp(-self.gamma * t):
                (s_a, logstring_a),(s_p, logstring_p),(s_n, logstring_n) = self.sample_supervised()
                self.log("SUPERVISED SAMPLE: {}".format((logstring_a, logstring_p, logstring_n)))
                yield s_a, s_p, s_n
            else:
                (s_a, logstring_a),_, _ = self.sample_unsupervised()
                self.log("UNSUPERVISED SAMPLE: {}".format((logstring_a)))
                yield s_a, None, None

            i += 1

        self.log("END OF TASK {}".format(self.name))


# Curriculum generator class. It wraps multiple tasks into a single object.
class CurriculumGenerator:
    def __init__(self, config, canvas_size, padding, logger=None):
        self.tasks = []
        self.current_task = 0

        self.canvas_size = canvas_size
        self.padding = padding
        self.logger = logger

        self.parse_config(json.loads(config))

    # Parse the curriculum JSON.
    def parse_config(self, config):
        for i, c in enumerate(config):
            self.tasks.append(Task(c["task_name"], c["samples"], c["alpha"], c["beta"], c["gamma"], c["noisy_color"], c["noisy_size"], c["positive_samples"], c["negative_samples"], self.tasks[0:i], self.canvas_size, self.padding, self.logger))

    # Reset the generator.
    def reset(self):
        self.current_task = 0

    # Generator for the entire curriculum. It visits each task in order.
    def generate_curriculum(self):
        while self.current_task < len(self.tasks):
            for sample in self.tasks[self.current_task].get_decision():
                yield sample
            self.current_task += 1

    # Returns a test set for task i, composed of size samples.
    def get_test_set(self, i, size):
        if i < len(self.tasks):
            return self.tasks[i].get_test_set(size)
        else:
            raise IndexError()

    # Returns a test set for the current task, composed of size samples.
    def get_current_test_set(self, size):
        return self.get_test_set(self.current_task, size)




"""
Computing the scale value for the exponential distribution:

We cannot use directly a geometric distribution in virtue of the fact that the number of samples is large and probabilities quickly drop to zero.
We use an exponential distribution opportunely limited between 0 and a value such that the minimum probability is beta.

We obtain a continuous variable by normalizing task progression in [0,1]: t = i / self.total, then we rescale this value by:
scale = log(gamma / beta) / gamma.

This scale factor comes from solving the equation:
beta = gamma * e **(-gamma * scale)

This can be easily rewritten as:
log(beta) = log(gamma) - gamma * scale * log(e)
log(beta) = log(gamma) - gamma * scale
log(beta) - log(gamma) = -gamma * scale
(log(gamma) - log(beta)) / gamma = scale
log(gamma / beta) / gamma = scale

gamma * e ** (-gamma * scale * i / total) is guaranteed to have a maximum gamma in i=0 and a minimum beta in i=total.
"""